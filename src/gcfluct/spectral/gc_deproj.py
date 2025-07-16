import numpy as np
from scipy.special import gamma
from numpy.typing import NDArray,ArrayLike
from typing import TYPE_CHECKING, Union, List, Optional
if typing.TYPE_CHECKING:
    from gcfluct.gc.selfsimilar_gc import SS_Model
    from gcfluct.spectral.spectra2d import ImageFromPS

class GC_Deproj:
    """
    Class which has a self-similar galaxy cluster class as a member. This class allows the user to
    deproject 2d spectra to 3d spectra (for galaxy clusters). There is not a single "deprojection" method.
    Rather, windows, their Fourier transforms, and the integration of the square of those transforms along
    k_z (wavenumbers along the z-axis), denoted N in the literature, are calculated for both SZ and X-ray
    models. From these (arrays of) quantities, the appropriate deprojection can be calculated. See, e.g.
    Romero (2024) for notes on how to use N across a region to obtain the deprojected spectrum (for said
    region).

    Attributes
    ----------
    dkz : np.floating
        Spacing (step size) of kz; units are inverse kpc.
    kz : NDArray[np.floating]
        Array of wavenumbers along the z-axis; calculated internally to have uniform (linear) spacing.
    n : np.floating
        "N", the scalar (de)projection factor to relate the 2D and 3D power spectra along a given LOS.
        See, e.g. Khatri & Gaspari (2016). (Selected at nearest radius; either X-ray or SZ)
    ns_sz : NDArray[np.floating]
        Array of "N", for SZ, calculated for each sky radius (from ss_model).
    ns_xr : NDArray[np.floating]
        Array of "N", for X-ray, calculated for each sky radius (from ss_model).
    plist_sz : list[NDArray[np.floating]]
        A list element is an array of the square of the Fourier transform of the SZ window. That is,
        integrating this array over k_z (the abstract sense of wavenumbers along the z-direction)
        produces n. Each list element corresponds to a single line of sight.
    plist_xr : list[NDArray[np.floating]]
        A list element is an array of the square of the Fourier transform of the X-ray windows. That is,
        integrating this array over k_z (the abstract sense of wavenumbers along the z-direction)
        produces n. Each list element corresponds to a single line of sight.
    ss_model : SS_Model
        An object of the SS_Model (self-similar galaxy cluster model) class.
    taper : NDArray[np.floating]
        A taper upon the window has been applied; initializes to ones (not actually a taper).
        taper and its associated arrays (tapered_window and tapered_w_tilde) are considered advanced features.
        That is, use of these attributes requires a fairly advanced user and potential choices best suited to
        building wrappers around current methods and/or attributes.
    tapered_window : NDArray[np.floating]
        Derived when a taper is imposed upon the window has been applied; initializes to the inital window.
    tapered_w_tilde : NDArray[np.floating]
        Derived when a taper is imposed upon the window has been applied; integrating this quantity along k_z
        yields N (for the tapered window). Initializes to the inital w_tilde.
    w_radius : np.floating
        The radius corresponding to n, window, and w_tilde. 
    w_tilde : NDArray[np.floating]
        The square of the Fourier transform of the (singular) window.    
    window : NDArray[np.floating]
        A singular window for a singular line of sight (via select_window method).
    wlist_sz: list[NDArray[np.floating]]
        A list element is an array the SZ window for the corresponding line of sight. See Romero et al.
        (2023), equation 7.
    wlist_xr: list[NDArray[np.floating]]
        A list element is an array the X-ray window for the corresponding line of sight. See Romero et al.
        (2023), equation 8.
    z_kpc : NDArray[np.floating]
        Array of z-coordinates (in kpc)
    z_scale : np.floating
        Elongation factor, if playing with elongating pressure or emissivity along the line of sight.
        (Defaults to unity - so assumes spherical case.)
    z_step : np.floating
        Distance step along the z-axis (kpc)
    
    Methods
    -------
    update_zs(zpad=10,stepfrac=1e-2,zscale=1.0)
        Updates attributes z_step, z_kpc, kz, dkz, and zscale.
    update_sz_windows()
        Updates attributes ns_sz, plist_sz, and wlist_sz.
    update_xr_windows()
        Updates attributes ns_xr, plist_xr, and wlist_xr.
    select_window(radius,sz=True)
        Selects a given window to inspect; sets attributes window, w_tilde, and n to the corresponding radius.
        Choice of SZ window (sz=True); otherwise X-ray window, w_tild, and n are taken.
    apply_taper()
        Applies the current (attribute) taper to the current (attribute) window.
    set_taper(zcut1,zcut2,tukeyTaper=False,lstep=2)
        Defines a taper (as would be applied to the window, either SZ or X-ray)

    """
        
    def __init__(self,ssgc_ss_model: 'SS_Model',
                 zpad : Union[float, int] = 10,
                 stepfrac : Union[float, int] = 1e-2
                 zscale : Union[float, int] = 1.0
                 w_radius : float = 0.0
                 ):
        """
        Initialize GC_Deproj by defining all attributes.

        Parameters:
        -----------
        ssgc_ss_model: SS_Model
            An object of class SS_Model; it will contain parameters for a self-similar model of the
            galaxy cluster in question.
        zpad : Union[float, int]
            What factor of R500 to extend to in the z-direction (both + and -). Defaults to 10, e.g.
            default z-range is -10*R500 to +10*R500.
        stepfrac : Union[float, int]
            What factor of R500 is the step size in the z-direction. Default is 1e-2. Thus, to go from
            -10*R500 to +10*R500, we'll expect 2000 steps.
        zscale : Union[float, int]
            If you want to play with ellipticity along the line-of-sight, this will account for it.
            NB that it works on pressure or emissivity, so the user actually needs to account for
            emissivity being density squared.
        w_radius : np.floating
            Radius (in the plane of the sky) at which a window and its w_tilde counterpart are
            extracted.
        """
        self.ss_model = ssgc_ss_model
        self.update_los(zpad=zpad,stepfrac=stepfrac,zscale=zscale)
        self.update_sz_windows()
        self.update_xr_windows()
        self.select_window(w_radius) # As a default value.
        self.taper = np.ones(self.window.shape) 
        self.tapered_window = self.window.copy()
        self.tapered_w_tilde = self.w_tilde.copy()

    def update_los(self,zpad=10,stepfrac=1e-2,zscale=1.0):
        """
        Updates attributes concerned with the range along the line of sight (taken as the z-axis).

        Parameters:
        -----------
        zpad : Union[float, int]
            What factor of R500 to extend to in the z-direction (both + and -). Defaults to 10, e.g.
            default z-range is -10*R500 to +10*R500.
        stepfrac : Union[float, int]
            What factor of R500 is the step size in the z-direction. Default is 1e-2. Thus, to go from
            -10*R500 to +10*R500, we'll expect 2000 steps.
        zscale : Union[float, int]
            If you want to play with ellipticity along the line-of-sight, this will account for it.
            NB that it works on pressure or emissivity, so the user actually needs to account for
            emissivity being density squared.
        """
        R500kpc = self.ss_model.R500.to("kpc").value
        self.z_step = R500kpc * stepfrac
        self.z_kpc = np.arange(R500kpc*zpad,R500kpc*zpad,self.z_step) # in kpc
        self.kz = np.fft.fftfreq(len(self.z_kpc),zscale*self.z_step)  # inverse kpc 
        self.dkz = self.kz[1]-self.kz[0]                              #
        self.zscale = zscale

    def update_sz_windows(self):
        """
        Takes the attribute ss_model and computes windows (wlist_sz), the square of their Fourier
        Transform (plist_sz), and the integration of the latter along the line of sight (ns_sz)
        for the SZ case (i.e. pressure profile).
        """
        ns = np.zeros(radians.shape) # N is used in the literature for value along a single LOS.
        wlist = [] # List of window arrays, to be filled. (A LOT OF DATA!)
        plist = []
        sky_kpc = self.rads.to("kpc").value # Array of projected radii (on the sky), in kpc
        for i,(los,y_at_los) in enumerate(zip(sky_kpc,self.yProf)):
            radii = np.sqrt(self.z_kpc**2 + sky_kpc**2) * u.kpc
            
            uprof = self.ss_model(radii)
            pprof = uprof.to("keV cm**-3") / self.zscale
            
            window = (pprof * self.P2invkpc).decompose().value / (y_at_los)  # Now in kpc**-1
            winfft = np.fft.fft(window)*self.z_step*self.z_scale
            w_tilde = np.abs(winfft**2)
            wlist.append(window)
            plist.append(w_tilde)
            ns[i] = np.sum(w_tilde)*self.dkz

        self.ns_sz = ns
        self.wlist_sz = wlist
        self.plist_sz = plist
        
    def update_xr_windows(self):
        """
        Takes the attribute ss_model and computes windows (wlist_xr), the square of their Fourier
        Transform (plist_xr), and the integration of the latter along the line of sight (ns_xr)
        for the X-ray case (i.e. SB, emissivity profile).
        """
        
        ns = np.zeros(self.radians.shape) # N is used in the literature for value along a single LOS.
        wlist = [] # List of window arrays, to be filled. (A LOT OF DATA!)
        plist = []
        sky_kpc = self.rads.to("kpc").value # Array of projected radii (on the sky), in kpc
        betanorm = gamma(3*self.xr_beta-0.5)*gamma(0.5) / gamma(3*self.xr_beta) 
        for i,los in enumerate(sky_kpc):
            radii = np.sqrt(self.z_kpc**2 + sky_kpc**2) * u.kpc
            r3d_scaled = (radii/self.xr_r_c).decompose().value
            r2d_scaled = (los/self.xr_r_c).decompose().value
            emmisivity = (1.0 + (r3d_scaled)**2)**(-3.0*self.xr_beta)    # 
            surface_b = (1.0 + (r2d_scaled)**2)**(0.5 -3.0*self.xr_beta) # Recalculating...kind of
            window = emmisivity / (betanorm*self.xr_beta*self.xr_r_c.value*surface_b)   # inverse kpc
            winfft = np.fft.fft(window)*self.z_step*self.z_scale
            w_tilde = np.abs(winfft**2)
            wlist.append(window)
            plist.append(w_tilde)
            ns[i] = np.sum(w_tilde)*self.dkz

        self.ns_xr = ns
        self.wlist_xr = wlist
        self.plist_xr = plist

    def select_window(self,radius: np.floating, sz: bool =True):
        """
        For an input radius (in units of kpc), finds associated window, w_tilde, and n at the nearest
        radial value in (attribute) rads. For record-keeping, w_radius = radius is set.
        
        Parameters
        ----------
        radius : np.floating
            The radius (in kpc -- but not a quantity!) at which you want to find the (nearest) window.
        sz : bool
            Do you want the SZ window? Default is True; if False, the X-ray window and companion quantities
            are set.
        """
        
        mydiff = np.abs(self.ss_model.rads.to("kpc").value - radius)
        mynpind = np.where(mydiff == np.min(mydiff))            # Find the closest point to 500 kpc
        myind = mynpind[0][0]
        self.w_radius = radius
        self.window = self.wlist_sz[myind] if sz else self.wlist_xr[myind]
        self.w_tilde = self.plist_sz[myind] if sz else self.plist_xr[myind]
        self.n = self.ns_sz[myind] if sz else self.ns_xr[myind]
        
    ###################################################################################################
    ### One more method that will be useful, but requires some attention to its use.                ###
    ###################################################################################################

    def create_nmap(self,ps_from_img: 'PSfromImages', sz: bool =True) -> NDArray[np.floating]:
        """
        With a perceived intention of calculating an image that is rescaled so that one can directly
        calculate the deprojected power spectra, this method requires input an object PSfromImages.
        
        Parameters
        ----------
        ps_from_img : PSfromImages
            Object with which the observer works to deduce power spectra (within regions, i.e. bins)
        sz : bool
            If being applied to an SZ image, select true; otherwise an X-ray image is inferred.

        Returns:
        --------
        n_map : NDArray[np.floating]
            A map of the N values.
        """
        
        rout_flat = ps_from_img.rmat.flatten() # In units of pixunits
        if ps_from_img.pixunits.unit.is_equivalent(u.arcsec):
            conversion = pixunits.to("arcsec").value * self.ss_model.scale # To match N units to pixunits
            r_radians = rout_flat * pixunits.to("radian").value
        else:
            conversion = pixunits.to("kpc").value # To match N units to pixunits
            r_kpc = rout_flat * conversion
            r_arcsec = r_kpc *u.arcsec / self.ss_model.scale
            r_radians = r_arcsec.to("radian")

        n_input = self.ns_sz if sz else self.ns_xr
        n_input *= conversion # N is natively in kpc**-1 
        
        n_flat = np.interp(r_radians, self.radians, n_input) # Interpolate N values onto map locations
        n_map = n_flat.reshape(ps_from_img.imsz)

        return n_map
    ###################################################################################################
    ### Below are advanced features. I leave them as they are, as I would want a user to access them...
    ### in the event that the user has a good understanding of the waters into which they wade.
    ###################################################################################################

    def apply_taper(self):
        """
        Applies the (attribute) taper to the (attribute) window and sets the attributes
        tapered_window and tapered_w_tilde.
        """

        self.tapered_window = self.taper*self.window
        tapered_winfft = np.fft.fft(self.tapered_window)*self.z_step
        self.tapered_w_tilde = np.abs(tapered_winfft**2)
        
    def set_taper(self,
                  zcut1 : np.floating,
                  zcut2 : np.floating,
                  tukey : bool =False,
                  lstep : Union[float,int] = 2):
        """
        Sets a taper (values 0 <= taper <= 1) which defines depths (along the line of sight) at which a
        power spectrum is "valid". Two "cuts" are allowed, in which case one can have a "notch" which defines
        a range of depths where a power spectrum is valid.
        Note that if 0 < zcut1 < zcut2, then for zcut1 < |z| < zcut2, the taper will be 0 for that defined range of z,
        and 1 outside that range (with a smooth transition close to zcut2 and zcut1).
        If zcut1 > zcut2, then for zcut2 < |z| < zcut1, the taper will be 1 in that range of z and 0 outside of it.
        And if zcut < 0, then only the taper at zcut2 will be applied.
        
        Parameters
        ----------
        zcut1 : np.floating
            The first location along the line-of-sight for a taper.
        zcut2 : np.floating
            The second location along the line-of-sight for a taper (opposite direction).
        tukey : bool
            Signals the use of the Tukey taper, rather than a Gaussian taper.
        lstep : float | int
            scaling to determine the length of transition (larger value = longer transition)
        """

        padding = lstep*self.z_step * np.sqrt(np.log(2))
        if zcut1 > 0:
            zdiff1 = zcut1 - np.abs(self.z_kpc) + padding # 0 at zcut1 + padding
            phase = np.pi*zdiff1/(2*lstep*self.z_step)
            tukey1 = 0.5* (1 + np.sin(phase))
            gauss1 = np.exp(-zdiff1**2 / (2*(lstep*self.z_step)**2))
            bi = (np.abs(self.z_kpc) > zcut1+padding)
            gauss1[bi] = 1.0
            tzero = (phase < -np.pi/2.0) # True when |z_kpc| > zcut1 + buffer
            tone = (phase > np.pi/2.0)   # True when |z_kpc| < zcut1 - buffer
            tukey1[tzero] = 0.0          # zero when |z_kpc| > zcut1 + buffer
            tukey1[tone] = 1.0           # one  when |z_kpc| < zcut1 - buffer
        else:
            gauss1 = np.ones(self.z_kpc.shape)
            tukey1 = np.ones(self.z_kpc.shape)

        zdiff2 = np.abs(self.z_kpc) - zcut2 - padding
        phase2 = np.pi*zdiff2/(2*lstep*self.z_step)
        tukey2 = 0.5* (1 - np.sin(phase2))
        tzero2 = (phase2 > -np.pi/2.0)
        tone2 = (phase2 < np.pi/2.0)
        tukey2[tzero2] = 0.0
        tukey2[tone2] = 1.0
        gauss2 = np.exp(-zdiff2**2 / (2*lstep*self.z_step))
        gi = (np.abs(self.z_kpc) < zcut2 - padding)
        gauss2[gi] = 1.0
        
        gauss = gauss1*gauss2   # I guess it's a double layer itself
        tukey = tukey1*tukey2      # Would sqrt() be better??

        self.taper = tukey if tukeyTaper else gauss
        
    def theory_projection(self,
                          img_ps : "ImagesFromPS",
                          k_theta : NDArray[np.floating],
                          w_tilde: NDArray[np.floating]
                          ) -> NDArray[np.floating]:
        """
        Not particularly useful for observers. This computes equation 6 in Romero et al. (2023).
        That is, it computes the un-approximated projection from 3D to 2D.
        
        Parameters
        ----------
        img_ps : ImagesFromPS
            An object of class ImagesFromPS; it will have parameters that would define the power
            spectrum for some turbulent gas which are used to generate the theoretical 3D power
            spectrum.
        k_theta : NDArray[np.floating]
            The array of wavenumbers which the observer is constraining. (At which wavenumbers
            do you have a 2D spectrum to which you want to compare the theoretical projection?)
        w_tilde : NDArray[np.floating]
            The square of the Fourier transform of the window function. That is, the weighting
            function for the 3D spectrum (within the integral). Note that, for w_tilde being 1D,
            this only applies to a single LOS.

        Returns
        -------
        p2d : NDArray[np.floating]
            The (theoretically) projected 2D power spectrum.        
        """
        nkz = len(self.kz)
        nkt = len(k_theta)
        kt2d = np.outer(k_theta,np.ones(nkz))
        kz2d = np.outer(np.ones(nkt),self.kz)
        k3d = np.sqrt(kt2d**2 + kz2d**2)
        p3d = img_ps.get_parameterized_PS(k3d) 
        p3wt = np.outer(np.ones(nkt),w_tilde) # w_tilde, or the weighting function for p3d
    
        p2d = np.sum(p3wt*p3d,axis=1)*self.dkz

        ### NB, if one should want to combine multiple spectra (along a line of sight), one could envision:
        ### (adopting some of the same variables, or similarly named variables):
        #
        # cummul_int = np.zeros(power_spectrum)
        # for (power_spectrum,tapered_w_tilde) in zip(power_spectra,tapered_w_tilde_list):
        #     integrand = np.sqrt(power_spectrum * tapered_w_tilde)
        #     cummul_int += integrand
        # p2d = np.sum(cummul_int**2,axis=1)*self.dkz
        #
        ### The point being that one adds the Fourier transforms and ONLY AFTER all coadditions does
        ### one take the squared quantity to integrate (sum) over.
        ###
        ### In any case, it's not clear how users would like that implemented -- they may want to
        ### implement their own function.

        return p2d


