import numpy as np
import scipy.constants as spconst
from scipy.interpolate import interp1d
# Now various astropy modules
from astropy.io import fits
from astropy import wcs   
import astropy.units as u
from astropy.units import Quantity, UnitBase
import astropy.constants as const
from astropy.coordinates import Angle #
from astropy.cosmology import FlatLambdaCDM

from numpy.typing import NDArray
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union

# Now project/repo modules
import gcfluct.utils.utility_functions as uf
import gcfluct.utils.numerical_integration as ni
import gcfluct.gc.tsz_spectrum as tsz
import gcfluct.gc.ksz_spectrum as ksz
from gcfluct.spectra.spectra2d import ImagesFromPS
from gcfluct.spectra.spectra2d import PSfromImages

class Cluster:

    """

    
    Attributes
    ----------
    arcminutes500 : np.floating
        Angular size of R_500 for the assumed cosmology.
    d_ang : Quantity
        Angular distance, with units of length.
    dens_crit : Quantity
        Critical density (mass per volume) of the Universe.
    z : np.floating
        Redshift of the cluster.
    m500 : Quantity
        Mass at density contrast of 500 of a cluster.
    r500 : Quantity
        Radius within which the average cluster density is 500 times the critical density (at the specified redshift).
    
    Methods
    -------
    get_cosmo()
        A method to return the otherwise "private" cosmo class.
    m2r_delta()
        At the redshift given by attribute z, provides a method to go from a density contrast mass to its corresponding
        radius. If no mass is specified, assumes m500 (and delta=500).
    r2m_delta()
        At the redshift given by attribute z, provides a method to go from a density contrast radius to its corresponding
        masss. If no radius is specified, assumes r500 (and delta=500).
    mdelta_from_ydelta()
        At the redshift given by attribute z, provides a method to go from an integrated Compton Y parameter (Y_sph) to 
        its corresponding density contrast mass. Default is delta=500.
    ydelta_from_mdelta()
        At the redshift given by attribute z, provides a method to go from a density contrast mass to its corresponding
        integrated Compton Y (Y_sph) parameter. Default is delta=500.
    """

    def __init__(self,
                 z: np.floating,
                 m500: Optional[Quantity] = None,
                 r500: Optional[Quantity] = None,
                 H0: np.floating = 70.0,
                 Om0: np.floating = 0.3,
                 Tcmb0: np.floating = 2.725):
        """
        Initialize the Cluster class. Set redshift, mass (or corresponding radius), and underlying cosmological parameters.
        
        Parameters
        ----------
        z : np.floating
            The redshift of the target cluster
        m500 : Optional[Quantity]
            The mass (M_500) of the cluster as a quantity with units of mass.
            If not specified then r500 must be specified!.
        r500 : Optional[Quantity]
            The radius (R_500) of the cluster as a quantity with units of length.
            If not specified then m500 must be specified!.
        H0 : np.floating
            The Hubble parameter at z=0 in km/s/Mpc. Default is 70.
        Om0 : np.floating
            The matter density fraction (Omega_matter) at z=0. Default is 0.3
        Tcmb0 : np.floating
            The temperature of the CMB at z=0 (in Kelvin). Default is 2.725
        """
        
        self._cosmo = FlatLambdaCDM(H0=H0,Om0=Om0,Tcmb0=Tcmb0)
        self.z = z
        self.d_ang = self._get_d_ang()
        self.dens_crit = self._cosmo.critical_density(z)
        self._h70 = (self._cosmo.H(0) / (70.0*u.km / u.s / u.Mpc))
        self._h = self._cosmo.H(z)/self._cosmo.H(0)

        if not m500 is None:
            self.m500 = m500
            self.r500 = self.m2r_delta()
        else:
            if r500 is None:
                raise TypeError("No m500 or r500 specified.")
            else:
                self.r500 = r500
                self.m500 = self.r2m_delta()

        self._p500 = self._p500_from_m500()
        self._ang500 = self._calc_theta500()  #r500 on the sky (in radians)
        self.arcminutes500 = self._ang500 * 60 * 180 / np.pi # And in arcseconds
        self.kpcperas = self.d_ang.to("kpc").value * np.pi / (3600 * 180)

    def get_cosmo():

        return self._cosmo
    
    def _get_d_ang(self) -> Quantity :
        """
        Calculates the angular distance from the redshift.

        Returns
        -------
        d_ang : Quantity
            The angular distance, with units of distance.
        """

        d_ang = self._cosmo.comoving_distance(self.z) / (1.0 + self.z)

        return d_ang
        
    def _calc_theta500(self):
        """
        Calculates the R_500 as an angle on the sky, in radians

        Returns
        -------
        r500ang : np.floating
            R_500 as an angle on the sky, in radians.
        """
    
        r500ang   = (self.r500/self.d_ang).decompose()
    
        return r500ang.value

    def _p500_from_m500(self):
        """
        Calculates the P_500 the self-similar pressure normalization according to Arnaud et al (2010)

        Returns
        -------
        p500 : Quantity
            P_500 in units of pressure.
        """
        p500 = (1.65 * 10**-3) * ((self._h)**(8./3)) * ((
            self.m500 * self._h70)/ ((3*10**14 * self._h70**(-1)) * const.M_sun)
            )**(2./3+0.11) * self._h70**2 * u.keV / u.cm**3
        return p500
                                       
    def m2r_delta(self,
                  mass: Optional[Quantity] = None,
                  delta: Union[np.floating,int] = 500
                  ) -> Optional[Quantity]:
        """
        Calculates the R_delta for a given overdensity, delta, with respect to the critical density of the universe (at the cluster redshift).

        Parameters
        ----------
        mass : Optional[Quantity]
            If no mass is specified, uses attribute m500.
        delta : Union[np.floating,int]
            The density contrast of interest (Valid for any value so long as the input mass matches that overdensity).
        
        Returns
        -------
        r_delta : Quantity
            r_delta in units of kpc.
        """

        m_delta = self.m500 if mass is None else mass
        r_delta = (3 * m_delta/(4 * np.pi  * delta * self.dens_crit))**(1/3.)
        r_delta = r_delta.to('kpc')
                                       
        return r_delta

    def r2m_delta(self,
                  radius: Optional[Quantity] = None,
                  delta: Union[np.floating,int]=500
                  ) -> Optional[Quantity]:
        """
        Calculates the M_delta for a given overdensity, delta, with respect to the critical density of the universe (at the cluster redshift).

        Parameters
        ----------
        radius : Optional[Quantity]
            If no radius is specified, uses attribute r500.
        delta : Union[np.floating,int]
            The density contrast of interest (Valid for any value so long as the input radius matches that overdensity).
        
        Returns
        -------
        mdelta : Quantity
            m_delta in units of kpc.
        """

        r_delta = self.r500 if radis is None else radius
        m_delta = 4 * np.pi / 3 * (r_delta)**3 * delta * self.dens_crit
        m_delta = m_delta.to('M_sun')

        return m_delta
                                       
    def mdelta_from_ydelta(self,
                           y_delta,
                           delta=500,
                           ):
        """
        Converts Y_delta to M_delta based on indicated Y-M relation (attribute ym_rel) for a given overdensity, delta,
        with respect to the critical density of the universe (at the cluster redshift).

        Parameters
        ----------
        y_delta : np.foating
            The Y_sph value at the given overdensity, in units of square Mpc.
        delta : Union[np.floating,int]
            The density contrast of interest (Valid for any value so long as the input radius matches that overdensity).
        
        Returns
        -------
        m_delta : np.floating
            m_delta in units of M_sun.
        """
        d_a = self.d_ang.to('Mpc').value
        iv = self._h**(-1./3)*d_a
        my_ydelta = y_delta * (iv**2)
        aaa,bbb = get_aaa_bbb(self.ym_rel,delta,ycyl=ycyl,h70=self._h70)
        m_delta = ( my_ydelta.value / 10**bbb )**(1./aaa)

        return m_delta
                                       
    def ydelta_from_mdelta(self,m_delta,delta=500,ycyl=False):
        """
        Converts M_delta to Y_delta based on indicated Y-M relation (attribute ym_rel) for a given overdensity, delta,
        with respect to the critical density of the universe (at the cluster redshift).

        Parameters
        ----------
        m_delta : np.foating
            The Y_sph value at the given overdensity, in units of square Mpc.
        delta : Union[np.floating,int]
            The density contrast of interest (Valid for any value so long as the input radius matches that overdensity).
        
        Returns
        -------
        y_delta : np.floating
            m_delta in units of M_sun.
        """

        d_a = self.d_ang.to('Mpc').value
        iv = self._h**(-1./3)*d_a
        aaa,bbb = get_aaa_bbb(self.ym_rel,delta,ycyl=ycyl,h70=self._h70)
        y_delta = m_delta**aaa * 10**bbb / (iv.value**2)
                                       
        return y_delta

class SS_Model(Cluster):
    """
    Child/subclass to Cluster class. Where Cluster only gives bulk quantities (at some overdensity) and a specified
    redshift, this class extends the attributes to include simple, self-similar thermodynamic (and corresponding
    surface brightness) profiles. This class simply adopts some standard parameterized models and is not intended to
    be used for fitting models to data. (Certainly no fitting infrastructure is provided here).
    
    Attributes
    ----------
    arcminutes500 : np.floating
        Angular size of R_500 for the assumed cosmology.
    d_ang : Quantity
        Angular distance, with units of length.
    dens_crit : Quantity
        Critical density (mass per volume) of the Universe.
    z : np.floating
        Redshift of the cluster.
    m500 : Quantity
        Mass at density contrast of 500 of a cluster.
    r500 : Quantity
        Radius within which the average cluster density is 500 times the critical density (at the specified redshift).

    ym_rel : str
        A short string with letter and last two digits of the year as a shorthand for the paper in which the
        relation is presented. Default is "A10".
    gnfw_pars : dict
        A dictionary which contains the gNFW parameters (p0, c500, a, b, and c).
    rads : NDArray[np.floating]
        Array of radial distances (in kpc)
    pixarc : np.floating
    npix : int

    rmat : NDArray[np.floating]
    y_map : NDArray[np.floating]
    xrsb_map : NDArray[np.floating]
    smoothedy_maps : NDArray[np.floating]
    smoothedy_profile : NDArray[np.floating]

    Methods
    -------
    get_cosmo()
        A method to return the otherwise "private" cosmo class.
    m2r_delta()
        At the redshift given by attribute z, provides a method to go from a density contrast mass to its corresponding
        radius. If no mass is specified, assumes m500 (and delta=500).
    r2m_delta()
        At the redshift given by attribute z, provides a method to go from a density contrast radius to its corresponding
        masss. If no radius is specified, assumes r500 (and delta=500).
    mdelta_from_ydelta()
        At the redshift given by attribute z, provides a method to go from an integrated Compton Y parameter (Y_sph) to 
        its corresponding density contrast mass. Default is delta=500.
    ydelta_from_mdelta()
        At the redshift given by attribute z, provides a method to go from a density contrast mass to its corresponding
        integrated Compton Y (Y_sph) parameter. Default is delta=500.
    """
    
    def __init__(self,
                 z: np.floating,
                 m500: Optional[Quantity] = None,
                 r500: Optional[Quantity] = None,
                 H0: np.floating = 70.0,
                 Om0: np.floating = 0.3,
                 Tcmb0: np.floating = 2.725,
                 rads: Optional[NDArray] = None,
                 npts: np.floating = 500,
                 r_max: np.floating = 20,
                 lg_rmin: np.floating = -0.5,
                 ym_rel='A10'):
        """
        Initialize the Cluster class. Set redshift, mass (or corresponding radius), and underlying cosmological parameters.
        
        Parameters
        ----------
        z : np.floating
            The redshift of the target cluster
        m500 : Optional[Quantity]
            The mass (M_500) of the cluster as a quantity with units of mass.
            If not specified then r500 must be specified!.
        r500 : Optional[Quantity]
            The radius (R_500) of the cluster as a quantity with units of length.
            If not specified then m500 must be specified!.
        H0 : np.floating
            The Hubble parameter at z=0 in km/s/Mpc. Default is 70.
        Om0 : np.floating
            The matter density fraction (Omega_matter) at z=0. Default is 0.3
        Tcmb0 : np.floating
            The temperature of the CMB at z=0 (in Kelvin). Default is 2.725
        rads : Optional[Quantity]
            If specified, an array of radii at which to calculate 3D quantities (pressure and emissivity), in kpc.
        npts : int
            If rads is not specified, this determines the number of elements in the rads array that will be generated.
        r_max : Union[np.floating,int]
            What factor of R_500 to treat as the maximum radial extent for 3D profiles. Default is 20.
        lg_rmin : np.floating
            The log (base 10) of the minimum radius (in kpc) to use in constructing a radial profile. Default is -0.5.
        ym_rel : str
            A short string with letter and last two digits of the year as a shorthand for the paper in which the
            relation is presented. Default is "A10".
        """

        Cluster.__init__(self,z,m500=m500,r500=r500,H0=H0,Om0=Om0,Tcmb0=Tcmb0)
        self.ym_rel = ym_rel
                                       
        Thom_cross = (spconst.value("Thomson cross section") *u.m**2).to("cm**2")
        mec2 = (const.m_e *const.c**2).to("keV") # Electron mass times speed of light squared, in keV
        self._pdl2y = Thom_cross * self.d_ang.to("cm") / mec2 * u.cm**3
        self._p2invkpc = Thom_cross * u.kpc.to("cm") / mec2 * u.cm**3
        lg_rmax = np.log10((r_max*self.r500/u.kpc).decompose().value) # when using kpc
        if rads is None:
            rads = np.logspace(lg_rmin,lg_rmax,npts) * u.kpc # 1 Mpc ~ r500, usually
        self.rads = rads
        self.radians = (rads / self.d_ang).decompose().value
        ###########################################################################################
        self.set_gnfw_profile() # Will assign A10 UPP pars at initialization.
        self.set_xr_USBP()

        ###########################################################################################
        # We can make maps of our cluster, but we'll want to know image properties to do so.
        # For now, set the maps to None.
        self.rmat = None
        self.y_map = None
        self.xrsb_map = None

        # Private variables
        self._imsz = None
        self._npix = None
        self._xymat = None

        
        self.smoothedy_maps = None # (re)set this to None; I want the user to specify smoothing kernels   
        self.smoothedy_profile = None
        # Currently no XRSB smoothing (beta models generally just absorb/include it... not always in the most
        # accurate manner.) In any case, smoothing the beta model isn't really an improvement all things
        # considered.

    def _set_xyrmat_from_imagefromps(self,
                                     imagefromps : ImagesFromPS):

        self._imsz = imagefromps._imsz
        is_length = imagefromps.pixunits.is_equivalent(u.kpc)
        if is_length:
            pix_arcsec = imagefromps.pixsize * imagefromps.pixunits.to("kpc") / self.kpcperas
        else:
            pix_arcsec = imagefromps.pixsize * imagefromps.pixunits.to("arcsec")
        self.pixarc = pix_arcsec
        self.pixunits = imagefromps.pixunits
        self._npix = self._imsz[0]
        self._xymat = ( imagefromps._xmat, imagefromps._ymat ) # Tuple of two 2D arrays (maps)
        self.rmat = imagefromps.rmat        
        
    def set_ss_maps(self,
                    n_r500: Union[np.floating,int] = 3.0,
                    pix_arcsec: Union[np.floating,int] = 1.0,
                    cx: Optional[Union[np.floating,int]] = None,
                    cy: Optional[Union[np.floating,int]] = None,
                    force_integer: bool = False,
                    imagefromps: Optional[ImagesFromPS] = None):

        if imagefromps is None:
            self._set_xyrmat(n_r500=n_r500,pix_arcsec=pix_arcsec,cx=cx,cy=cy,force_integer=force_integer)
        else:
            self._set_xyrmat_from_imagefromps(imagefromps)
        self.set_ymap_from_gnfw()
        self.set_xrsb_map()
        self.smoothedy_maps = None # (re)set this to None; I want the user to specify smoothing kernels   
        self.smoothedy_profile = None
        self.smoothedxrsb_maps = None # (re)set this to None; I want the user to specify smoothing kernels   
        self.smoothedxrsb_profile = None
        
    def set_gnfw_profile(self,
                       c500: np.floating = 1.177,
                       p0: np.floating = 8.403,
                       a: np.floating = 1.0510,
                       b: np.floating = 5.4905,
                       c: np.floating = 0.3081):
        """
        Here is a block of methods. I might want to change the functionality so that I could make a    
        Compton-y profile (set_yprof) in one command after updating gnfw pars.                   
        I probably will just write a new method to do that.
        .. math::

            P(r) = \\frac{p0 P_{500}}{((c_{500} r)/R_{500})^c [1 + ((c_{500} r)/R_{500})^a]^(b-a)/c}


        Parameters
        ----------
        c500 : np.floating
            The concentration parameter
        p0 : np.floating
            The pressure normalization, on top of the unversal (scaled) pressure
        a : np.floating
            The turnover rate in (seen as :math:`\\alpha`in some references)
        b : np.floating
            The outer pressure profile slope (seen as :math:`\\beta`in some references)
        c : np.floating
            The inner pressure profile slope (seen as :math:`\\gamma`in some references)
        """
        self._set_gnfw_pars()
        self.set_pressure_profile_gnfw()
        self.set_ulPprof()
        self.set_yprof()
        
    def _set_gnfw_pars(self,
                       c500: np.floating = 1.177,
                       p0: np.floating = 8.403,
                       a: np.floating = 1.0510,
                       b: np.floating = 5.4905,
                       c: np.floating = 0.3081):
        """
        The user should generally not access this by itself, as one could update the gNFW parameters without
        updating other relevant model products. set_gnfw_model() groups all the products together. Recall the form of
        the gNFW profile:
        .. math::

            P(r) = \\frac{p0 P_{500}}{((c_{500} r)/R_{500})^c [1 + ((c_{500} r)/R_{500})^a]^(b-a)/c}

        
        Parameters
        ----------
        c500 : np.floating
            The concentration parameter
        p0 : np.floating
            The pressure normalization, on top of the unversal (scaled) pressure
        a : np.floating
            The turnover rate in (seen as :math:`\\alpha`in some references)
        b : np.floating
            The outer pressure profile slope (seen as :math:`\\beta`in some references)
        c : np.floating
            The inner pressure profile slope (seen as :math:`\\gamma`in some references)
        """

        self.gnfw_pars = {"c500":c500, "p0":p0, "a":a, "b":b, "c":c}        
        
    def _set_xyrmat(self,
                    n_r500: Union[np.floating,int] = 3.0,
                    pix_arcsec: Union[np.floating,int] = 1.0,
                    pixunits : UnitBase = None,
                    cx: Optional[Union[np.floating,int]] = None,
                    cy: Optional[Union[np.floating,int]] = None,
                    force_integer: bool = False):
        """
        Defines grids for maps to be made.

        Parameters
        ----------
        n_r500 : Union[np.floating,int]
            How many factors of R500 to extend to in radius, along a given axis. Default is 3.0 such that the entire image
            will be 6 R_500 on a side (and sqrt(2) more along the diagonal)
        pix_arcsec : np.floating
        pixunits : BaseUnit
        cx : Optional[Union[np.floating,int]]
            If provided, the center of the target, in pixel coordinates, along axis=0
        cy : Optional[Union[np.floating,int]]
            If provided, the center of the target, in pixel coordinates, along axis=1
        force_integer : bool
            If cx and cy are not provided, this would for cx and cy to be an integer. Default is False.
        """
                                       
        self.pixarc = pix_arcsec
        mapsize = np.round(self.arcminutes500*2*n_r500)
        self._npix = int(np.round((mapsize*60)/self.pixarc))
        if cx is None:
            cx   = self._npix//2 if force_integer else self._npix/2.0
        if cy is None:
            cy   = self._npix//2 if force_integer else self._npix/2.0
        x1   = (np.arange(self._npix)-cx)*self.pixarc
        y1   = (np.arange(self._npix)-cy)*self.pixarc
        x    = np.outer(x1,np.ones(self._npix))
        y    = np.outer(np.ones(self._npix),y1)
        
        if isinstance(pixunits,UnitBase):
            is_angle = pixunits.is_equivalent(u.deg)
            is_length = pixunits.is_equivalent(u.kpc)
            if is_angle or is_length:
                self.pixunits = pixunits
        else:
            if pixunits is None:
                # Maybe the user is just playing around and doesn't care about units.
                if not no_warn:
                    warnings.warn("No pixel units were input! Using u.arcsec; proceed with caution.")
                self.pixunits = u.arcsec
            else:
                raise AttributeError("Pixel units must either be a length or angle.")
        
        self._xymat = (x,y) # Tuple of two 2D arrays (maps)
        self.rmat = np.sqrt(x**2 + y**2)

    def gnfw(self, radii : Quantity = None) -> Quantity:
        """
        Computes the gNFW profile:
        .. math::

            P(r) = \\frac{p0 P_{500}}{((c_{500} r)/R_{500})^c [1 + ((c_{500} r)/R_{500})^a]^(b-a)/c}

        based on gNFW parameters which have been set as attributes.
        
        Parameters
        ----------
        radii : Optional[Quantity]
            Radii, as a quantity with units of length, if provided. Otherwise adopts attribute rads.

        Returns
        -------
        pressure : Quantity
            Output has units of pressure with same data type as input radii (or attribute rads) if no radii provided.
        """

        if radii is None:
            radii = self.rads
        p_norm = self._p500 * self._h70**-1.5 * self.gnfw_pars["p0"]   # self.p0
        r_p = self.r500 / self.gnfw_pars["c500"]
        r_scaled =  (radii/r_p).decompose().value
        #pressure = (p_norm / (((r_scaled)**self.c)*((1 + (r_scaled)**self.a))**((self.b - self.c)/self.a)))

        core_pressure = r_scaled**self.gnfw_pars["c"] 
        outer_exponent = (self.gnfw_pars["b"] - self.gnfw_pars["c"])/self.gnfw_pars["a"] # (b-c)/a
        bulk = ( 1 + r_scaled**self.gnfw_pars["a"] )**outer_exponent
        pressure = p_norm / bulk        

        return pressure

    def set_pressure_profile_gnfw(self):
        """
        Sets the gNFW pressure profile based on attributes.
        """

        self.pressure_prof = self.gnfw()
    
    def set_ulPprof(self):
        """
        Rescales attribute pressure_prof to a unitless value and assigns this to attribute unitless_pressure_profile.
        """

        self.unitless_pressure_profile = (self.pressure_prof * self._pdl2y).decompose().value

    def set_yprof(self,nr500max=10):
        """
        Calculates and sets y_prof

        Parameters
        ----------
        nr500max : Union[np.floating,int]
            To what depth, along the line of sight (LOS) do we use to calculate the Compton y profile, relative to R500?
            The default is 10. (cf. Arnaud et al. 2010 used 5).
        """

        self.y_prof = ni.int_profile(self.radians, self.unitless_pressure_profile,self.radians,zmax=self._ang500*nr500max)
    
    def set_ymap_from_gnfw(self):
        """
        From attribute y_prof, interpolates and sets attribute ymap.
        """

        self.set_yprof()
        flatymap = uf.grid_profile(self.radians,self.y_prof,self._xymat)
        self.y_map = flatymap.reshape((self._npix,self._npix))

    def set_gnfw_beam_smoothed_map(self,
                               fwhm : np.floating = 10.0):
        """
        Smooths the Compton y map (y_map) by a single Gaussian, i.e. an instrument's beam (or PSF).
        Sets attribute smoothedy_maps, itself a dictionary; keys are a string-formating of FWHM

        Paramters
        ---------
        fwhm : np.floating
            The FWHM of a 2D Gaussian that approximates an instrument's beam.
        """
        
        y_prof = self.get_gnfw_yprof()
        pixfwhm = fwhm/self.pixarc
        smoothed = imf.fourier_filtering_2d(self.y_map,"gauss",pixfwhm)     # in Compton y
        dictkey = "{:.2f}".format(fwhm).replace('.','p')
        if self.smoothedy_maps is None:
            self.smoothedy_maps = {dictkey:smoothed}
        else:
            self.smoothedy_maps[dictkey] = smoothed

    def set_gnfw_beam_smoothed_y_prof(self,fwhm=10.0):
        """
        Creates a Compton-y profile of the same array size (same radii) as the unsmoothed profile,
        but for a given beam (taken as a single, circular Gaussian, characterized by its FWHM).
        Sets attribute smoothedy_profile, itself a dictionary; keys are a string-formating of FWHM.

        Paramters
        ---------
        fwhm : np.floating
            The FWHM of a 2D Gaussian that approximates an instrument's beam.
        """

        dictkey = "{:.2f}".format(fwhm).replace('.','p')
        if self.smoothedy_maps is None:
            self.set_gnfw_beam_smoothed_map(fwhm=fwhm)
        else:
            if not (dictkey in self.smoothedy_maps):
                self.set_gnfw_beam_smoothed_map(fwhm=fwhm)
        smoothed = self.smoothedy_maps[dictkey]
        rbin,ybin,yerr,ycnts = uf.bin_two2Ds(self.rmat,self,binsize=self.pixarc*2.0)
        fint = interp1d(rbin,ybin, bounds_error = False, fill_value = "extrapolate")
        radarcsec = self.radians * 3600 * 180/np.pi
        if self.smoothedy_profile is None:
            self.smoothedy_profile = {dictkey:fint(radarcsec)}
        else:
            self.smoothedy_profile[dictkey] = fint(radarcsec)

    #####################################################################################
            
    def calculate_y_cyl_prof(self) -> NDArray[np.floating]:
        """
        While the user could do their own calculations by hand, let's provide an avenue for accurate calculations.
        
        User can choose a beam-smoothed y-profile. Otherwise (if no y-profile is specified), the method will use
        the unsmoothed Compton y profile.            
        
        Returns
        -------
        y_cyl : Quantity
            An array, expressed as a Quantity, containing Y_cylindrical values of the same shape as input y-profile.
            Units are square arcminutes.
        """

        yratios = self.y_prof[:-1] / self.y_prof[1:]
        rratios = self.radians[:-1] / self.radians[1:]
        alpha = np.log(yratios) / np.log(rratios) # i.e. spectral index

        # For int_a^b f(x) = F(b) - F(a) =  k (F(b)/k - F(a)/k), here is: (F(b)/k - F(a)/k)
        integral_bounds = (self.radians[1:])**(2.0-alpha) - (self.radians[:-1])**(2.0-alpha)

        # The common factors, k are provided here:
        integral_norm = (self.y_prof[:-1]*(self.y_prof[:-1]/u.rad)**alpha) / (2.0 - alpha)

        # OK, with the additon of 2*pi, as we integrated over 2 pi r dr
        tint  = 2.0*np.pi * np.cumsum(integral_bounds*integral_norm) * u.sr

        # Add a tenth of the first entry to the front to keep the same size array.
        y_cyl  = np.hstack([tint[0].to("arcmin2")]/10.0,tint.to("arcmin2")) 

        return y_cyl

    def y_sphere(self):
        """
        While the user could do their own calculations by hand, let's provide an avenue for accurate calculations.
        
        User can choose a beam-smoothed y-profile. Otherwise (if no y-profile is specified), the method will use
        the unsmoothed Compton y profile.            
        
        Returns
        -------
        y_sphere : Quantity
            An array of Y_spherical values of the same shape as input y-profile (and shape as self.radians),
            with units of square Mpc
        """
    
        pratios = self.unitless_pressure_profile[:-1] / self.unitless_pressure_profile[1:]
        rratios = self.radians[:-1] / self.radians[1:]
        alpha = np.log(pratios) / np.log(rratios)
     
        # For int_a^b f(x) = F(b) - F(a) =  k (F(b)/k - F(a)/k), here is: (F(b)/k - F(a)/k)
        integral_bounds = (self.rads[1:]/u.kpc)**(3.0-alpha) - (self.rads[:-1]/u.kpc)**(3.0-alpha)

        # The common factors, k are provided here:
        integral_norm = (self.unitless_pressure_profile[:-1]*(self.rads[:-1]/u.kpc)**alpha) / (3.0 - alpha)
        
        # OK, with the additon of 4*pi, as we integrated over 4 pi r**2 dr
        tint  = 4.0*np.pi * np.cumsum(parint) * (u.kpc)**2

        # Add a tenth of the first entry to the front to keep the same size array.
        y_sphere = np.hstack([tint[0].to("Mpc2")]/10.0,tint.to("Mpc2"))
    
        return y_sphere

    def _set_xr_universal_beta_pars(self,soft_only: bool = True):
        """
        Sets "universal" beta model parameters for XMM-like observations of galaxy clusters. The parameters would define images
        of counts rates (cnts/s/arcmin**2)

        Parameters
        ----------
        soft_only : bool
            Get the brightness for 0.4-1.25 keV band (soft) or a fuller spectrum (0.4-1.25 + 2.0-7.0 keV)
        """
        beta = 2.0/3.0
        x = 0.1
        theta_c = x * self.arcminutes500 # in arcminutes
        #self.d_ang.to("Mpc").value
        if soft_only:
            universal_I = 5.9e-4 / (u.Mpc)
            m_pivot = 9.5e14 * u.M_sun
            m_scale = (self.m500/m_pivot).decompose().value
            z_scale = self.r500 * m_scale * self._h**2 / (1 + self.z)**3
        else:
            universal_I = 7.7e-4 / (u.Mpc**2)
            z_scale = self.r500**2 * self._h**(7.0/6.0) / (1 + self.z)**3
            
        I_xscale = (2*np.pi * x**2 * (np.sqrt(1+x**2) - 1)) / (np.sqrt(1+x**2))
        I_not = (universal_I * z_scale / I_xscale).decompose().value

        self.xr_I_0 = I_not
        self.xr_theta_c = theta_c
        self.xr_beta = beta

    def set_xr_USBP(self,soft_only: bool = True):
        """
        Sets "universal" beta model profile (attribute xr_sb_prof)

        Parameters
        ----------
        soft_only : bool
            Get the brightness for 0.4-1.25 keV band (soft) or a fuller spectrum (0.4-1.25 + 2.0-7.0 keV)
        """
    
        radarcmins = self.radians * 60 * 180/np.pi
        self._set_xr_universal_beta_pars(soft_only=soft_only)  # rc in arcminutes
        sb_prof = self.xr_I_0 / ( (1 + (radarcmins/self.xr_theta_c)**2)**(3*self.xr_beta - 0.5) )
        self.xr_sb_prof = sb_prof

    def set_xrsb_map(self):
        """
        From attribute y_prof, interpolates and sets attribute ymap.
        """

        flatxrsbmap = uf.grid_profile(self.radians,self.xr_sb_prof,self._xymat)
        self.xrsb_map = flatxrsbmap.reshape((self._npix,self._npix))
        
##########################################################################################################################
######                                                                                                               #####
######        Functions that contain relations from the literature or otherwise not tied to objects                  #####
######                                                                                                               #####
##########################################################################################################################

def get_aaa_bbb(ym_rel: str,
                delta:Union[np.floating,int],
                h70: np.floating=1.0
                ) -> Tuple[np.floating,np.floating]:
    """
    Basically just a repository (look-up table) of Y-M relations.
    ym_rel must be either:
       (1) 'A10' (Arnaud 2010)
       (2) 'A11' (Anderson 2011)
       (3) 'M12' (Marrone 2012)
       (4) 'P14' (Planck 2014), or
       (5) 'P17' (Planelles 2017)

    All are converted to Y = 10^bbb * M^aaa; mass (M) is in units of solar masses; Y is in Mpc^2 (i.e. with D_A^2 * E(z)^-2/3)

    Parameters
    ----------
    ym_rel : str
        One of the strings as identified in the function description.
    delta : Union[np.floating,int]
        Must be either 500 or 2500. (No other relations are currently stored.)
    h70 : np.floating
        The scaled (to 70 km/s/Mpc) Hubble parameter at z=0. Default is 1.0

    Returns
    -------
    expr : Tuple[np.floating,np.floating]
        A two-element tuple of parameters aaa,bbb for the relation provided in the description
    """

    if delta == 2500:
        if ym_rel == 'A10':   # Taken from Comis+ 2011
            aaa = 1.637
            bbb = -28.13  
        if ym_rel == 'A11':
            raise ValueError("This relation does not exist")
        elif ym_rel == 'M12':
            bbb = -30.669090909090908
            aaa = 1.0 / 0.55
        elif ym_rel == 'M12-SS':
            bbb = -28.501666666666667
            aaa = 5.0/3.0
        elif ym_rel == 'P14':
            raise ValueError("This relation does not exist")
        elif ym_rel == 'P17':     # Planelles 2017
            aaa = 1.755
            bbb = -29.6833076    # -4.585
        elif ym_rel == 'H20':    
            raise ValueError("This relation does not exist")
        else:
            raise ValueError("No such relation found")
            
    elif delta == 500:
        if ym_rel == 'A10':
            aaa   = 1.78
            iofx  = 0.6145  
            Bofx  = 2.925e-5 * iofx * h70**(-1) / (3e14/h70)**aaa
            bbb = np.log10(Bofx)
        if ym_rel == 'A11':
            b_tabulated = 14.06 # But this is some WEIRD Y_SZ (M_sun * keV) - M relation
            b_conversion = -18.855
            a_conversion = -24.176792495381836
            aaa = 1.67
            bbb = b_tabulated + b_conversion + a_conversion  # Anderson+ 2011, Table 6
        elif ym_rel == 'M12':
            bbb = -37.65227272727
            aaa = 1.0 / 0.44
        elif ym_rel == 'M12-SS':
            bbb = -28.735
            aaa = 5.0/3.0
        elif ym_rel == 'P14':
            aaa = 1.79
            bbb = -30.6388907     
        elif ym_rel == 'P17':
            aaa = 1.685
            bbb = -29.0727644    
        elif ym_rel == 'H20':
            aaa = 1.790
            bbb = -30.653047     
        else:
            raise ValueError("No such relation found")
    else:
        raise ValueError("Only density contrasts (currently) allowed are 500 and 2500.")

    return aaa,bbb

def _get_YM_sys_err(logy:np.floating,
                    ym_rel: str,
                    delta:Union[np.floating,int] = 500,
                    h70: np.floating = 1.0
                    ) -> np.floating:
    
    """
    Calculates the fractional uncertainty in a mass estimate due to the relation uncertainties.
    Accounts for pivot points where given.
    
    Parameters
    ----------
    logy : np.floating
        Log10(Y_sphere) value (when Y is expressed in square Mpc).
    ym_rel : str
        One of the strings as identified in the function description.
    delta : Union[np.floating,int]
        Must be either 500 or 2500. (No other relations are currently stored.)
    h70 : np.floating
        The scaled (to 70 km/s/Mpc) Hubble parameter at z=0. Default is 1.0

    Returns
    -------
    xer : np.floating
        The fractional uncertainty in the mass value based on reported Y-M relation uncertainties.
    """
    if delta == 500:
        if ym_rel == 'A10':
            pivot = 3e14
            iofx  = 0.6145
            norm  = 2.925e-5 * iofx * h70**(-1)
            aaa = 1.78
            t1   = 0.024 / aaa
            t2   = ((np.log10(norm) - logy)/aaa**2)*0.08
            xer  = np.sqrt(t1**2 + t2**2) * np.log(10)
            #import pdb;pdb.set_trace()
        elif ym_rel == 'A11':
            t1   = 0.29 # Fixed slope
            t2   = 0.1
            xer  = np.sqrt(t1**2 + t2**2) * np.log(10)
            print(xer)
        elif ym_rel == 'M12':
            t1   = np.array([1.0,logy+5])
            #t1   = np.array([0.367,0.44])
            tcov = np.array([[0.098**2,-0.012],[-0.012,0.12**2]])
            #tcov = np.array([[0.098**2,-(0.012**2)],[-(0.012**2),0.12**2]])
            #tcov = np.array([[0.098**2,0],[0,0.12**2]])
            t2   = np.abs(np.matmul(t1,np.matmul(tcov,t1)))
            xer  = np.sqrt(t2) * np.log(10)
            print(xer)
        elif ym_rel == 'M12-SS':
            t1   = 0.0 # Fixed slope
            t2   = 0.036
            xer  = np.sqrt(t1**2 + t2**2) * np.log(10)
            print(xer)
        elif ym_rel == 'P14':
            t1   = 0.06
            t2   = 0.079
            xer  = np.sqrt(t1**2 + t2**2) * np.log(10)
            print(xer)
        elif ym_rel == 'P17':
            norm = -4.305
            aaa = 1.685
            t1   = 0.009 / aaa
            t2   = ((norm - logy)/aaa**2)*0.013
            xer  = np.sqrt(t1**2 + t2**2) * np.log(10)
            #xer = 0.104
        elif ym_rel == 'H20':
            pivot = 3e14; 
            norm  = 10**(-4.739); aaa = 1.79
            #t1   = ((logy - 1)/aaa )*0.024
            t1   = 0.003 / aaa
            t2   = ((np.log10(norm) - logy)/aaa**2)*0.015
            xer  = np.sqrt(t1**2 + t2**2) * np.log(10)
        else:
            print('No match!')
            import pdb;pdb.set_trace()
    elif delta == 2500:
        if ym_rel == 'A10': # Taken from Comis+ 2011
            lognorm = -28.13; aaa = 1.637
            t1   = 0.88 / aaa 
            t2   = ((logy - lognorm)/aaa**2)*0.062
            xer  = np.sqrt(t1**2 + t2**2) * np.log(10)
            #xer  = np.log(1 + 0.23)
        elif ym_rel == 'A11':
            raise ValueError("This relation does not exist")
        elif ym_rel == 'M12':
            t1   = np.array([1.0,logy+5])
            tcov = np.array([[0.063**2,-(0.008**2)],[-(0.008**2),0.14**2]])
            t2   = np.abs(np.matmul(t1,np.matmul(tcov,t1)))
            xer  = np.sqrt(t2) * np.log(10)
            print(xer)
        elif ym_rel == 'M12-SS':
            t1   = 0.0 # Fixed slope
            t2   = 0.033
            xer  = np.sqrt(t1**2 + t2**2) * np.log(10)
            print(xer)
        elif ym_rel == 'P14':
            raise ValueError("This relation does not exist")
        elif ym_rel == 'P17':
            norm = -4.5855; aaa = 1.755
            t1   = 0.014 / aaa
            t2   = ((norm - logy)/aaa**2)*0.020
            xer  = np.sqrt(t1**2 + t2**2) * np.log(10)
        elif ym_rel == 'H20':
            raise ValueError("This relation does not exist")
        else:
            raise ValueError("No such relation found")
    else:
        raise ValueError("Only density contrasts (currently) allowed are 500 and 2500.")

    return xer


