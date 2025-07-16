import numpy as np
import scipy.constants as spconst
from scipy.interpolate import interp1d
# Now various astropy modules
from astropy.io import fits
from astropy import wcs   
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import Angle #
from astropy.cosmology import FlatLambdaCDM
import warnings

# Now project/repo modules
import utlity_functions as uf
import numerical_integration as ni
import tsz_spectrum as tsz
import ksz_spectrum as ksz

class Cluster:


    
    def __init__(self,z,m500=None,r500=None,H0=70.0, Om0=0.3, Tcmb0=2.725):
        """
        Initializes the Cluster class. Fundamental quantities are a redshift (z) and either M_500 or R_500.
        Additionally, a cosmology must be assumed.

        Parameters
        ----------
        z : np.floating
            Redshift of the cluster.
        m500 : Optional[Quantity]
            M_500 of the cluster (with mass units)
            NB, either m500 or r500 *must* be supplied!
        r500 : Optional[Quantity]
            R_500 of the cluster (with length units)
            NB, either m500 or r500 *must* be supplied!
        H0 : np.floating
            The Hubble parameter at z=0 (in units of km/s/Mpc). Default is 70.0
        Om0 : np.floating
            Omega_matter at z=0. Default is 0.3
        Tcmb0 : np.floating
            Temperature of the CMB at z=0 in Kelvin. Default is 2.725
        """
        self._set_cosmology(h0=h0,Om0=Om0,Tcmb0=Tcmb0) 
        self.z = z
        self.d_ang = get_d_ang(self)
        self.dens_crit = self.cosmo.critical_density(z)
        self.h70 = (self.cosmo.H(0) / (70.0*u.km / u.s / u.Mpc))
        self.h = self.cosmo.H(z)/self.cosmo.H(0) # Sometimes called E in the literature.

        if not m500 is None:
            self.m500 = m500
            self.r500 = self.m2r_deltar(delta=500)
        else:
            if r500 is None:
                raise TypeError("No m500 or r500 specified.")
            else:
                self.r500 = r500
                self.m500 = self.r2m_deltam(delta=500)

        self.ang500 = theta500_from_m500()  #r500 on the sky (in angular units)
        self.arcminutes500 = self.ang500 * 60 * 180 / np.pi # And in arcseconds
        arcseconds500 = self.arcminutes500 * 60
        self.scale = self.r500.to("kpc").value / arcseconds500 # kpc / arcsecond

    def _set_cosmology(self,H0=70.0, Om0=0.3, Tcmb0=2.725):
        """
        Sets the cosmo attribute (a class from astropy).
        """
        
        self.cosmo = FlatLambdaCDM(H0=H0, Om0=Om0, Tcmb0=Tcmb0)

    def get_d_ang(self) -> Quantity:
        """
        Calculates the angular distance from the redshift.

        Returns
        -------
        d_ang : Quantity
            A quantity with units of distance (e.g. kpc)
        """

        d_ang = self.cosmo.comoving_distance(self.z) / (1.0 + self.z)

        return d_ang
        
    def return_theta500(self)-> np.floating:
        """
        Finds the angular extent (on the sky) of R_500 

        Returns
        -------
        r500ang : np.floating
            Angular_extent of R500, in radians.
        """
    
        r500ang = (self.r500/self.d_ang).decompose()
    
        return r500ang.value

    def p500_from_m500(self,mass=None) -> Quantity:
        """
        Uses relation offered in Arnaud et al. (2010) to calculate P_500 as a function of M_500.

        Parameters
        ----------
        mass : Optional[Quantity]
            If None (default), uses attribute m500; otherwise input as quantity with mass units.

        Returns
        -------
        p500 : Quantity
            Corresponding pressure (normalizations) of the cluster, with corresponding units.
        """
        m500 = self.m500 if mass is None else mass * u.Msun
        p500 = (1.65 * 10**-3) * ((E)**(8./3)) * ((
            m500 * self.h70)/ ((3*10**14 * h70**(-1)) * const.M_sun)
            )**(2./3+0.11) * self.h70**2 * u.keV / u.cm**3
        return p500

    def m2r_delta(self,mass=None,delta=500):
        """
        Convert from M_delta to R_delta (at the specified delta).
        
        Parameters
        ----------
        mass : Optional[Quantity]
            If None (default), uses attribute m500. If specifying, the value(s) should be in solar masses.
        delta : Union[np.floating,int]
            Any (single) density contrast is allowed (but it should match the density contrast of the radius specified.

        Returns
        -------
        r_delta : Quantity
            Corresponding radius(radii) of the cluster, in units of kpc. A solitary value if using attribute value
            of m500 or input scalar; otherwise matches input dimensions.
        """

        m_delta = self.m500 if mass is None else mass
        r_delta = (3 * m_delta/(4 * np.pi  * delta * self.dens_crit))**(1/3.)
        r_delta = r_delta.to('kpc')

        return r_delta

    def r2m_delta(self,
                  radius: Optional[Quantity] = None,
                  delta=500
                  ) -> Quantity:
        """
        Convert from R_delta to M_delta (at the specified delta).
        
        Parameters
        ----------
        radius : Optional[Quantity]
            If None (default), uses attribute r500. If specifying, the value(s) should be in kpcs.
        delta : Union[np.floating,int]
            Any (single) density contrast is allowed, but it should match the density contrast of the radius specified.

        Returns
        -------
        m_delta : Quantity
            Corresponding mass(es) of the cluster, in units of solar masses.
            of r500 or input scalar; otherwise matches input dimensions.
        """

        r_delta = self.r500 if radius is None else radius
        m_delta = 4 * np.pi / 3 * (r_delta)**3 * delta * self.dens_crit
        m_delta = m_delta.to('M_sun')

        return m_delta

    def mdelta_from_ydelta(self,y_delta,delta=500):
        """
        Uses the relation:
        .. math::

            Y = 10 ^ bbb * M ^ aaa

        with mass, M, in units of solar masses. Y is in units of Mpc^2. Calls another function as a "look-up table" for
        values of aaa and bbb based on the literature reference (indicated by attribute ym_rel).

        Parameters
        ----------
        y_delta : np.floating
            Y_sph value, with units of square Mpc at the indicated density contrast (relative to critical density)
        delta : Union[np.floating,int]
            Currently only density constrast of 500 and 2500 are supported. (Options at 2500 are limited).

        Returns
        -------
        m_delta : np.floating
            Corresponding mass of the cluster, in units of solar masses.

        """
        d_a = self.d_ang.to('Mpc').value
        iv = self.h**(-1./3)*d_a
        myYdelta = y_delta * (iv**2)
        aaa,bbb = get_aaa_bbb(self.ym_rel,delta,h70=self.h70)
        m_delta = ( myYdelta.value / 10**bbb )**(1./aaa)

        return m_delta

    def ydelta_from_mdelta(self,m_delta,delta=500):
        """
        Uses the relation:
        .. math::

            Y = 10 ^ bbb * M ^ aaa

        with mass, M, in units of solar masses. Y is in units of Mpc^2. Calls another function as a "look-up table" for
        values of aaa and bbb based on the literature reference (indicated by attribute ym_rel).

        Parameters
        ----------
        m_delta : np.floating
            Mass of the cluster at the indicated density contrast (relative to critical density), in units of solar masses.
        delta : Union[np.floating,int]
            Currently only density constrast of 500 and 2500 are supported. (Options at 2500 are limited).

        Returns
        -------
        y_delta : np.floating
            Returns the corresponding Y_sph value, with units of square Mpc.
        """

        d_a = self.d_ang.to('Mpc').value
        iv = self.h**(-1./3)*d_a
        aaa,bbb = get_aaa_bbb(self.ym_rel,delta,h70=self.h70)
        y_delta = m_delta**aaa * 10**bbb / (iv.value**2)

        return y_delta

class SS_Model(Cluster):
    
    def __init__(self,
                 z: np.floating,
                 m500: Optional[Quantity[np.floating]] = None,
                 r500: Optional[Quantity[np.floating]] = None,
                 rads: Optional[NDArray[Quantity[np.floating]]] = None,
                 ym_rel: str = 'A10',
                 npts: int = 500,
                 r_max: Union[np.floating, int] = 8,
                 lg_rmin: np.floating = -0.5,
                 pix_arcsec: Union[np.floating, int] = 1.0):
        """

        Parameters:
        z : np.floating
            The redshift of the galaxy cluster
        m500 : Optional[Quantity[np.floating]]
            The mass of the galaxy cluster at density contrast, relative to critical density, of 500.
            NB, either m500 or r500 *must* be supplied!
        r500 : Optional[Quantity[np.floating]]
            The radius of the galaxy cluster at density contrast, relative to critical density, of 500.
            NB, either m500 or r500 *must* be supplied!
        rads : Optional[NDArray[Quantity[np.floating]]]
            An array of radii with units of length. If not specified, a logarthmically spaced array will
            be generated based on lg_rmin and r_max.
        ym_rel : str
            A string denoting which Y-M relation to use. (See function get_aaa_bbb). Defaults to 'A10', for
            the relation presented in Arnaud et al. (2010).
        npts : int
            Number of points (elements) to use in the array of rads, if a user-supplied array is not provided.
        r_max : Union[np.floating, int]
            Out to what factor of R_500 to generate pressure profiles.
        lg_rmin : np.floating
            Log10 value of the minimum radius to use, in kpc. Defaults to -0.5 (roughly a third of a kpc).
        pix_arcsec : Union[np.floating, int]
            Size of a pixel (same on each side), in arcseconds. Default is 1.
        """
        Cluster.__init__(self,z,m500=m500,r500=r500)
        self.ym_rel = ym_rel

        Thom_cross = (spconst.value("Thomson cross section") *u.m**2).to("cm**2")
        mec2 = (const.m_e *const.c**2).to("keV") # Electron mass times speed of light squared, in keV
        boltzmann = spconst.value("Boltzmann constant in eV/K")/1000.0 # keV/K  
        planck = spconst.value("Planck constant in eV s")/1000.0 # keV s
        self.Pdl2y = Thom_cross*self.d_ang.to("cm")/mec2 * u.cm**3 # Scaled by angular distance
        self.P2invkpc = Thom_cross*u.kpc.to("cm") / mec2 * u.cm**3 # Scaled by 1 kpc.
        #r500, p500       = r500_p500_from_m500_z(m500,z)
        lg_rmax = np.log10((r_max*self.r500/u.kpc).decompose().value) # when using kpc
        if rads is None:
            rads = np.logspace(lg_rmin,lg_rmax,npts) * u.kpc # 1 Mpc ~ r500, usually
        self.rads = rads
        self.radians = (rads / self.d_ang).decompose().value
        self.pixrad = pix_arcsec * u.arcsec.to('rad').value
        self.pixarc = pix_arcsec
        ###########################################################################################
        self.set_gnfw_model() # Will assign A10 gnfw pars at initialization.

    def set_gnfw_model(self,c500=1.177, p0=8.403, a=1.0510, b=5.4905, c=0.3081):
        """
        Sets the underlying gNFW parameters and creates associated outputs. I congregate them so that
        if one updates the parameters, then all the associate outputs are updated correspondingly.
        A reminder, the gNFW profile is denoted as:

        .. math::

            P(r) =  \\frac{p500 p}{\\left( (r c_{500}/r_{500})^{c} (1 + r c_{500}/r_{500})^a \\right) ^{(b - c)/a} }

        where :math:`r` is the radius).

        Parameters
        ----------
        c500 : np.floating
            Concentration parameter.
            Defaults to 1.177, the value found in Arnaud et al. (2010) for the entire sample.
        p0 : np.floating
            Pressure normalization.
            Defaults 8.403, the value found in Arnaud et al. (2010) for the entire sample.
        a : np.floating
            Rate of turnover; denoted as :math:`\\alpha` in many references.
            Defaults to 1.051, the value found in Arnaud et al. (2010) for the entire sample.
        b : np.floating
            Outer (logarithmic) pressure slope; denoted as :math:`\\beta` in many references.
            Defaults to 5.4905, the value found in Arnaud et al. (2010) for the entire sample.
        c : np.floating
            Inner (logarithmic) pressure slope; denoted as :math:`\\gamma` in many references.
            Defaults to 0.3081, the value found in Arnaud et al. (2010) for the entire sample.
        """
        self._set_gnfw_pars()
        self.set_pressure_profile_gnfw()
        self.set_ulPprof()
        self.set_yprof()
        self.set_xyrmap()
        self.set_ymap_from_gnfw()
        self.smoothy_map = None # (re)set this to None; I want the user to specify smoothing kernels   
        self.smoothy_profile = None
        
    def _set_gnfw_pars(self,
                       c500: np.floating = 1.177,
                       p0: np.floating = 8.403,
                       a: np.floating = 1.0510,
                       b: np.floating = 5.4905,
                       c: np.floating = 0.3081):
        """
        The user should generally not access this by itself, as one could update the gNFW parameters without
        updating other relevant model products. set_gnfw_model() groups all the products together.
        The parameters correspond to the generalized NFW profile:
    
        .. math::

            P(r) =  \\frac{p500 p}{\\left( (r c_{500}/r_{500})^{c} (1 + r c_{500}/r_{500})^a \\right) ^{(b - c)/a} }

        where :math:`r` is the radius (a stand-in for the array, radii).

        Parameters
        ----------
        c500 : np.floating
            Concentration parameter.
            Defaults to 1.177, the value found in Arnaud et al. (2010) for the entire sample.
        p0 : np.floating
            Pressure normalization.
            Defaults 8.403, the value found in Arnaud et al. (2010) for the entire sample.
        a : np.floating
            Rate of turnover; denoted as :math:`\\alpha` in many references.
            Defaults to 1.051, the value found in Arnaud et al. (2010) for the entire sample.
        b : np.floating
            Outer (logarithmic) pressure slope; denoted as :math:`\\beta` in many references.
            Defaults to 5.4905, the value found in Arnaud et al. (2010) for the entire sample.
        c : np.floating
            Inner (logarithmic) pressure slope; denoted as :math:`\\gamma` in many references.
            Defaults to 0.3081, the value found in Arnaud et al. (2010) for the entire sample.
        """
        self.c500 = c500
        self.p0 = p0
        self.a = a
        self.b = b
        self.c = c
        
    def set_xyrmap(self,
                   n_r500: Union[np.floating, int] = 3.0,
                   cx: Optional[Union[np.floating, int] = None,
                   cy: Optional[Union[np.floating, int] = None,
                   force_integer: bool = False):
        """
        Sets maps of x-, y-, and r- coordinates in units of arcseconds.

        Parameters
        ----------
        n_r500 : Union[np.floating, int]
            The sides of the maps will be twice this length (relative to R_500). The default is 3.0;
            therefore the sides of the map are 6 R_500.
        cx : Optional[Union[np.floating, int]
            The center x-coordinate (in pixels, taken along axis=0) of the map (target).
            Defaults to the center of the map (x_length / 2)
        cy : Optional[Union[np.floating, int]
            The center y-coordinate (in pixels, taken along axis=1) of the map (target).
            Defaults to the center of the map (y_length / 2)
        force_integer : bool
            Forces cx and cy to be integers (using floor division)
        """

        mapsize = np.round(self.arcminutes500*2*n_r500)
        xpix = int(np.round((mapsize*60)/self.pixarc))
        ypix = int(np.round((mapsize*60)/self.pixarc))
        if cx is None:
            cx   = xpix//2 if force_integer else xpix/2.0
        if cy is None:
            cy   = ypix//2 if force_integer else ypix/2.0
        x1   = (np.arange(xpix)-cx)*self.pixarc
        y1   = (np.arange(ypix)-cy)*self.pixarc
        x    = np.outer(x1,np.ones(ypix))
        y    = np.outer(np.ones(xpix),y1)
    
        self.xymap = (x,y) # Tuple of two 2D arrays (maps)
        self.rmap = np.sqrt(x**2 + y**2)

    def set_pressure_profile_gnfw(self):
        """
        Sets the pressure profile from the gNFW parameters set, in units of pressure, at the radii indicated in attribute rads.
        """
        
        self.pressure_prof = self.gnfw_pres_radii(self.rads)
    
    def gnfw_pres_radii(self,radii: Quantity[NDArray[np.floating]]) -> : Quantity[NDArray[np.floating]]:
        """
        Returns the pressure profile from the gNFW parameters set, in units of pressure.
        (Allows the user access to a pressure profile at radii of their own choosing).

        Parameters
        ----------
        radii : Quantity[NDArray[np.floating]]
            An array of radii, with units of length.

        Returns
        -------
        pressure_prof : Quantity[NDArray[np.floating]]
            The corresponding pressure, in units of pressure.
        """

        pressure_prof = gnfw(self.r500,self.p500,radii,h70=self.h70,
                          c500=self.c500, p=self.p0, a=self.a, b=self.b, c=self.c)
        
        return pressure_prof
    
    def set_ulPprof(self):
        """
        Creates the "unitless" pressure profile from the gNFW parameters set. Paired with radii in radians,
        integrations can be done without concern for units (the units are all accounted for, up front).
        Assigns the unitless profile to attribute unitless_pressure_profile
        """

        self.unitless_pressure_profile = (self.pressure_prof * self.Pdl2y).decompose().value

    def set_yprof(self,nr500max: Union[int,np.floating] = 5):
        """
        Creates the Compton y profile from the gNFW parameters set.
        Assigns this to attribute y_prof.
        Parameters
        ----------
        nr500max : Union[int,np.floating]
            Out to what factor of R_500 should the Compton y profile be created? Default is 5 (i.e. out to 5 R_500)
        """

        self.y_prof = ni.int_profile(self.radians, self.unitless_pressure_profile,self.radians,zmax=self.Ang500*nr500max)
    
    def set_ymap_from_gnfw(self):
        """
        Creates the (sky) map of Compton y values based on the Compton y profile from the gNFW parameters set.
        Assigns this to attribute ymap.
        """

        self.set_gnfw_yprof()
        flatymap = UF.grid_profile(self.radians,self.y_prof,self.xymap)
        ymap = flatymap.reshape((nx,ny))
        self.y_map = ymap

    def make_gnfw_beam_smoothed_map(self,fwhm=10.0):
        """
        Assigns a dictionary (or adds an entry to it) of the map, using the smoothing kernel (transform into a string)
        as the dictionary key.
        
        Parameters
        ----------
        fwhm : np.floating
            The value of the FWHM of the beam, in arcseconds. The default is 10 (the FWHM of MUSTANG-2).
        """

        # Note: There may be a desire to extend this for more complex beams in the future.
        # I could let the user specify a key based on the instrument, e.g. "MUSTANG-2", and they could supply their own smoothed maps.
        # Let's see when we get there. (The get_gnfw_smoothed_y_prof method could easily be adapted to this.)
        y_prof = self.get_gnfw_yprof()
        pixfwhm = fwhm/self.pixarc
        beam_map = imf.fourier_filtering_2d(self.y_map,"gauss",pixfwhm)     # in Compton y
        fwhmstr = "{:.2f}".format(fwhm).replace('.','p')
        dictkey = "yMap_"+fwhmstr
        if self.smoothy_map is None:
            self.smoothy_map = {dictkey:beam_map}
        else:
            self.smoothy_map[dictkey] = beam_map

    def get_gnfw_smoothed_y_prof(self,fwhm=10.0):
        """
        Creates a Compton-y profile of the same array size (same radii) as the unsmoothed profile,
        but for a given beam (taken as a single, circular Gaussian, characterized by its FWHM). 

        Parameters
        ----------
        fwhm : np.floating
            The value of the FWHM of the beam, in arcseconds. The default is 10 (the FWHM of MUSTANG-2).
       """

        fwhmstr = "{:.2f}".format(fwhm).replace('.','p')
        dictkey = "yMap_"+fwhmstr
        if self.smoothy_map is None:
            self.make_gnfw_beam_smoothed(fwhm=fwhm)
        else:
            if not (dictkey in self.smoothy_map):
                self.make_gnfw_beam_smoothed(fwhm=fwhm)
        beam_ymap = self.smoothy_map[dictkey]
        rbin,ybin,yerr,ycnts = UF.bin_two2Ds(self.rmap,beam_ymap,binsize=self.pixarc*2.0)
        fint = interp1d(rbin,ybin, bounds_error = False, fill_value = "extrapolate")
        radarcsec = self.radians * 3600 * 180/np.pi
        if self.smoothedy_profile is None:
            smoothy_profile = {dictkey:fint(radarcsec)}
            self.smoothy_profile = smoothedy_profile
        else:
            self.smoothy_profile[dictkey] = fint(radarcsec)

    #####################################################################################
            
    def calculate_Y_cyl_prof(self,yprof=None) -> NDArray[np.floating]:
        """
        Calculates and returns :math:`Y_{cyl}(r)` as a profile, either from a user-specified Compton y profile,
        or (if no y-profile is specified), the method will use the unsmoothed Compton y profile, which is the
        attribute yprof.
        
        Returns
        -------
        y_cyl : NDArray[np.floating]
            An array of :math:`Y_{cyl}(r)` which has units of square arcmin.
        """

        if yprof is None:
            yprof = self.y_prof
        yratios = yprof[:-1] / yprof[1:]
        rratios = self.radians[:-1] / self.radians[1:]
        alpha = np.log(yratios) / np.log(rratios) # Logarthmic slope
        
        # For int_a^b { f(x) dx } = k* ( F(b) - F(a)), the following notes F(b) - F(a):
        int_bounds = (self.radians[1:])**(2.0-alpha) - (self.radians[:-1])**(2.0-alpha)

        # And the following term corresponds to k, where common factors were "extracted".
        int_norm = (yprof[:-1]*(yprof[:-1]/u.rad)**alpha) / (2.0 - alpha)
        
        # Since we integrated over 2 pi r dr, we have a factor of 2 pi:
        tint  = 2.0*np.pi * np.cumsum(parint) * u.sr

        # To return an array of the same dimensions, add something "small" in numerical value to the front.
        y_cyl  = np.hstack([tint[0].to("arcmin2")]/10.0,tint.to("arcmin2")) # Stack to return same dimensionality

        return y_cyl

    def calc_y_sphere_profile(self) -> NDArray[np.floating]:
        """
        Calculates and returns :math:`Y_{sph}(r)` as a profile from attributes unitless_pressure_profile    

        Returns
        -------
        y_sphere : NDArray[np.floating]
            An array of :math:`Y_{sph}(r)` which has units of square Mpc.
        """
    
        pratios = self.unitless_pressure_profile[:-1] / self.unitless_pressure_profile[1:]
        rratios = self.radians[:-1] / self.radians[1:]
        alpha = np.log(pratios) / np.log(rratios) # Logarthmic slope

        # For int_a^b { f(x) dx } = k* ( F(b) - F(a)), the following notes F(b) - F(a):
        int_bounds = (self.rads[1:]/u.kpc)**(3.0-alpha) - (self.rads[:-1]/u.kpc)**(3.0-alpha)
        
        # And the following term corresponds to k, where common factors were "extracted".
        int_norm = (self.unitless_pressure_profile[:-1]*(self.rads[:-1]/u.kpc)**alpha) / (3.0 - alpha)
        
        # Since we integrated over 4 pi r**2 dr, we have a factor of 4 pi:
        integrated = 4.0*np.pi * np.cumsum(int_norm * int_bounds) * (u.kpc)**2

        # To return an array of the same dimensions, add something "small" in numerical value to the front.
        y_sphere = np.hstack([tint[0].to("Mpc2")]/10.0,tint.to("Mpc2"))
    
        return y_sphere

    def _set_xr_univ_beta_pars(self,soft_only: bool = True):
        """
        Sets :math:`I_0`, :math:`\\theta_c`, and :math:`\\beta` as the attributes xr_I_0, xr_theta_c, and xr_beta,
        respectively, for a classic Beta model:
        
        .. math::

            I(r) = I_0 \\left(1 + (\\theta/\\theta_c)^2\\right) ^ {-3 \\beta + 0.5}       

        Parameters
        ----------
        soft_only : bool
            Are you using the soft-band only? The soft band is taken as 0.4-1.25 keV.
            If not, assumes 0.5-7.0 keV, excluding 1.25 to 2.5 keV. Default is True.
        """
       
        beta = 2.0/3.0
        x = 0.1
        self.xr_r_c = x * self.r500
        theta_c = x * self.Ang500 *60 * 180 / (D_a_mpc * np.pi) # arcminutes
        if SoftOnly:
            Universal_I = 5.9e-4 / (u.Mpc)
            Zscale = self.r500*self.E**2 / (1 + z)**3
        else:
            Universal_I = 7.7e-4 / (u.Mpc**2)
            Zscale = self.r500**2 * self.E**(7.0/6.0) / (1 + self.z)**3
            I_xscale = (2*np.pi * x**2 * (np.sqrt(1+x**2) - 1)) / (np.sqrt(1+x**2))
            I_not = (Universal_I * Zscale / I_xscale).decompose().value

        self.xr_I_0 = I_not
        self.xr_theta_c = theta_c
        self.xr_beta = beta

    def set_xr_usbp(self,SoftOnly: bool=True):
        """
        Method sets the X_ray "universal surface brightness profile" (USBP), which is really an approximate form for count rate
        as observed with XMM. It seems to scale reasonably well across redshift - certainly well enough for toy calculations.

        Parameters
        ----------
        soft_only : bool
            Are you using the soft-band only? The soft band is taken as 0.4-1.25 keV.
            If not, assumes 0.5-7.0 keV, excluding 1.25 to 2.5 keV. Default is True
        """
    
        radarcmins = self.radians * 60 * 180/np.pi
        self._set_xr_univ_beta_pars(soft_only = soft_only)  # rc in arcminutes
        sbprof = self.xr_I_0 / ( (1 + (radarcmins/self.xr_theta_c)**2)**(3*self.xr_beta - 0.5) )
    
        self.xr_sb_prof = sbprof

##########################################################################################################################
######                                                                                                               #####
######        Functions that contain relations from the literature or otherwise not tied to objects                  #####
######                                                                                                               #####
##########################################################################################################################

def gnfw(r500, p500, radii, c500= 1.177, p=8.403, a=1.0510, b=5.4905, c=0.3081,h70=1.0):
    """
    Function returns the generalized NFW profile:
    
    .. math::

        P(r) =  \\frac{p500 p}{\\left( (r c_{500}/r_{500})^{c} (1 + r c_{500}/r_{500})^a \\right) ^{(b - c)/a} }

    where :math:`r` is the radius (a stand-in for the array, radii).

    Parameters
    ----------
    r500 : Quantity
        :math:`R_{500}`, with length units.
    p500 : Quantity
        :math:`P_{500}`, with pressure units.
    radii : NDArray[Quantity]
        Array of radii, with length units.
    c500 : np.floating
        Concentration parameter.
        Defaults to 1.177, the value found in Arnaud et al. (2010) for the entire sample.
    p : np.floating
        Pressure normalization.
        Defaults 8.403, the value found in Arnaud et al. (2010) for the entire sample.
    a : np.floating
        Rate of turnover; denoted as :math:`\\alpha` in many references.
        Defaults to 1.051, the value found in Arnaud et al. (2010) for the entire sample.
    b : np.floating
        Outer (logarithmic) pressure slope; denoted as :math:`\\beta` in many references.
        Defaults to 5.4905, the value found in Arnaud et al. (2010) for the entire sample.
    c : np.floating
        Inner (logarithmic) pressure slope; denoted as :math:`\\gamma` in many references.
        Defaults to 0.3081, the value found in Arnaud et al. (2010) for the entire sample.

    Returns
    -------
    prof : NDArray[Quantity]
        Array of pressure, with units of pressure.
    """
    
    p0 = p500 * p * h70**-1.5
    rp = r500 / c500
    rf =  (radii/rp).decompose().value
    prof = (p0 / (((rf)**c)*((1 + (rf)**a))**((b - c)/a)))

    return prof

def get_aaa_bbb(ym_rel,
                delta,
                h70=1.0):
    """
    A repository of Y-M (specifically Y_{sph}-M) relations. Relations at Delta=500 exist for all,
    but relations do not exist for all at Delta=2500.
    ym_rel must be either:
       (1) 'A10' (Arnaud 2010)
       (2) 'A11' (Anderson 2011)
       (3) 'M12' (Marrone 2012)
       (4) 'P14' (Planck 2014), or
       (5) 'P17' (Planelles 2017)

    All are converted to:
    
    .. math::

        Y = 10 ^ bbb * M ^ aaa

    with mass, M, in units of solar masses. Y is in units of Mpc^2 (i.e. with D_A^2 * E(z)^-2/3)

    Parameters
    ----------
    ym_rel : str
        A string from of letter (initial) and last 2 digits of the year, which serves as
        a shorthand for the paper from which the relation originates. (See function description).
    delta : np.floating
        At which overdensity factor do you desire the relation. Only available for 500 or 2500.
    h70 : np.floating
        The Hubble parameter at z=0, scaled to 70 km/s/Mpc. Default is 1.
    
    Returns
    -------
    aaa : np.floating
        The exponent (power-law) dependence of Y on M
    bbb : np.floating
        The log_10 normalization for the relation.
    """

    if delta == 2500:
        if ym_rel == 'A10': # Extension from Comis+ 2011
            aaa = 1.637
            bbb = -28.13     
        if ym_rel == 'A11': # No relation provided
            raise ValueError("A11 did not supply a relation at Delta=2500")
        elif ym_rel == 'M12':
            bbb = -30.669090909090908
            aaa = 1.0 / 0.55
        elif ym_rel == 'M12-SS':
            bbb = -28.501666666666667
            aaa = 5.0/3.0
        elif ym_rel == 'P14':     # No relation provided
            raise ValueError("P14 did not supply a relation at Delta=2500")
        elif ym_rel == 'P17':     # Planelles 2017
            aaa = 1.755
            bbb = -29.6833076    
        elif ym_rel == 'H20':     # No relation provided
            raise ValueError("H20 only supplied a relation at Delta=500")
        else:
            aaa = 1.637           # The A10 extension from Comis+ 2011
            bbb = -28.13          # 
            warnings.warn('No relation specified; using Arnaud+ 2010 (Comis+ 2011) values')
            
    elif delta == 500:
        if ym_rel == 'A10':
            aaa = 1.78
            #iofx = 0.7398 if ycyl else 0.6145  # Deprecated. Only relation also provided for Y_cyl
            iofx = 0.6145  # I(x)
            bofx = 2.925e-5 * iofx * h70**(-1) / (3e14/h70)**aaa #B(x)
            bbb = np.log10(bofx)
        if ym_rel == 'A11':
            b_tabulated = 14.06          # Non-cannonical Y-M relation
            b_conversion = -18.855
            a_conversion = -24.176792495381836
            aaa = 1.67
            bbb = b_tabulated + b_conversion + a_conversion  # cf. Anderson+ 2011, Table 6
        elif ym_rel == 'M12':
            bbb = -37.65227272727
            aaa = 1.0 / 0.44
        elif ym_rel == 'M12-SS':
            bbb = -28.735
            aaa = 5.0/3.0
        elif ym_rel == 'P14':
            aaa = 1.79
            bbb = -30.6388907    # 
        elif ym_rel == 'P17':
            aaa = 1.685
            bbb = -29.0727644    # -4.585
        elif ym_rel == 'H20':
            aaa = 1.790
            bbb = -30.653047     # -4.739
        else:
            aaa = 1.78
            iofx = 0.6145  # I(x)
            bofx = 2.925e-5 * iofx * h70**(-1) / (3e14/h70)**aaa #B(x)
            bbb = np.log10(bofx)
            warnings.warn('No relation specified; using Arnaud+ 2010 values')
    else:
        raise ValueError("Currently supported Delta values: 500, 2500.")

    return aaa,bbb

def _get_ym_sys_err(logy,ym_rel,delta=500,h70=1.0):
    """
    For a repository of Y-M (specifically Y_{sph}-M) relations, the calculates the systematic errors
    according to uncertainties reported in their papers. Relations at Delta=500 exist for all, but
    relations do not exist for all at Delta=2500.
    ym_rel must be either:
       (1) 'A10' (Arnaud 2010)
       (2) 'A11' (Anderson 2011)
       (3) 'M12' (Marrone 2012)
       (4) 'P14' (Planck 2014), or
       (5) 'P17' (Planelles 2017)

    Parameters
    ----------
    logy : np.floating
        The log_10 value of Y_spherical. (For relations with a pivot, this is important).
    ym_rel : str
        A string from of letter (initial) and last 2 digits of the year, which serves as
        a shorthand for the paper from which the relation originates. (See function description).
    delta : np.floating
        At which overdensity factor do you desire the relation. Only available for 500 or 2500.
    h70 : np.floating
        The Hubble parameter at z=0, scaled to 70 km/s/Mpc. Default is 1.
    
    Returns
    -------
    xer : np.floating
        The fractional error on the mass due to unertainty in the scaling relation itself.
    """

    if delta == 500:
        if ym_rel == 'A10':
            pivot = 3e14
            iofx  = 0.6145
            norm = 2.925e-5 * iofx * h70**(-1)
            aaa = 1.78
            t1 = 0.024 / aaa
            t2 = ((np.log10(norm) - logy)/aaa**2)*0.08
            xer = np.sqrt(t1**2 + t2**2) * np.log(10)
        elif ym_rel == 'A11':
            t1   = 0.29 # Fixed slope
            t2   = 0.1
            xer  = np.sqrt(t1**2 + t2**2) * np.log(10)
            print(xer)
        elif ym_rel == 'M12':
            t1   = np.array([1.0,logy+5])
            tcov = np.array([[0.098**2,-0.012],[-0.012,0.12**2]])
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
            norm = -4.305; aaa = 1.685
            t1   = 0.009 / aaa
            t2   = ((norm - logy)/aaa**2)*0.013
            xer  = np.sqrt(t1**2 + t2**2) * np.log(10)
        elif ym_rel == 'H20':
            pivot = 3e14; 
            norm  = 10**(-4.739); aaa = 1.79
            t1   = 0.003 / aaa
            t2   = ((np.log10(norm) - logy)/aaa**2)*0.015
            xer  = np.sqrt(t1**2 + t2**2) * np.log(10)
        else:
            raise ValueError("Specified Y-M relation (ym_rel) is not in this repository.")

    elif delta == 2500:
        if ym_rel == 'A10':
            xer  = np.log(1 + 0.23)
        elif ym_rel == 'A11':
            raise ValueError("A11 did not supply a relation at Delta=2500")
        elif ym_rel == 'M12':
            t1 = np.array([1.0,logy+5])
            tcov = np.array([[0.063**2,-(0.008**2)],[-(0.008**2),0.14**2]])
            t2 = np.abs(np.matmul(t1,np.matmul(tcov,t1)))
            xer = np.sqrt(t2) * np.log(10)
        elif ym_rel == 'M12-SS':
            t1 = 0.0 # Fixed slope
            t2 = 0.033
            xer = np.sqrt(t1**2 + t2**2) * np.log(10)
        elif ym_rel == 'P14':
            raise ValueError("P14 did not supply a relation at Delta=2500")
        elif ym_rel == 'P17':
            norm = -4.5855
            aaa = 1.755
            t1   = 0.014 / aaa
            t2   = ((norm - logy)/aaa**2)*0.020
            xer  = np.sqrt(t1**2 + t2**2) * np.log(10)
        elif ym_rel == 'H20':
            raise ValueError("H20 did not supply a relation at Delta=2500")
        else:
            raise ValueError("Specified Y-M relation (ym_rel) is not in this repository.")

    else:
        raise ValueError("Currently supported Delta values: 500, 2500.")

    return xer


