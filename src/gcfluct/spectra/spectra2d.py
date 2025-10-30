### Inter-package dependencies
import numpy as np
import scipy.special as sps
import astropy.units as u
import warnings

### Intra-package dependencies
import gcfluct.utils.utility_functions as uf

### For typing:
from numpy.typing import NDArray, ArrayLike
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union, TypeAlias
#from astropy.units import Quantity
from astropy.units import UnitBase

Floating: TypeAlias = Union[float, np.float32, np.float64]

class PSfromImages:
    """
    For an input image, and some input parameters, this allows one to compute the power spectrum
    via the Mexican hat method presented in Arevalo et al (2012). To aid in some sanity checks,
    a standard FFT method is also provided, though it will not respect any mask. 

    Attributes
    ----------
    center : ArrayLike
        A two-element iterable indicating the central pixel along respective axes.
    img : NDArray[Floating]
        The image upon which power spectra will be calculated.
    img2 : Optional[NDArray[Floating]]
        None if not set; otherwise a user-supplied image of the same astrometry to be used to
        calculate cross-spectra.
    intrinsic_mask : NDArray[Floating]
        Mask, independent of bin (region), which mask any "contaminating" pixels.
    mask : NDArray[Floating]
        Total mask, for a given bin (region)
    mask_by_bin : NDArray[Union[Floating,int]]
        Mask which defines bins by the integer values of the pixels.
    pixsize : Floating
        The value of the pixel size in the appropriate units.
    pixunits : UnitBase
        Units of the pixel size, as an astropy unit. Recommended to be set by user. Default is u.arcsec.

    
    
    Methods
    -------
    update_img():
    set_center():
    set_intrinsic_mask()
    set_wave_numbers()
    set_manual_mask_by_bin()
    set_annular_mask_by_bin()
    ps_via_fft_2d()
    ps_via_a12_2d()
    """
    
    def __init__(self,
                 img: NDArray[Floating],
                 pixsize: Floating = 1.0,
                 pixunits: Optional[UnitBase] = None,
                 center: Optional[ArrayLike] = None,
                 intrinsic_mask: Optional[NDArray[Floating]] = None,
                 img2: Optional[NDArray[Floating]] = None):
        """


        Parameters
        ----------
        img : NDArray[Floating]
           The image upon which power spectra will be calculated.
        pixsize : Floating
           The value of the pixel size in the appropriate units.
        pixunits : Optional[UnitBase]
           It is highly recommend to explicitly set this. Default None type becomes u.arcsec;
           user is warned.
        center : Optional[ArrayLike]
           A two-element array-like object with center in pixel units, along axis 0 and axis 1,
           respectively.
        intrinsic_mask : Optional[NDArray[Floating]]
           A mask that covers pixels that should be omitted (regardless of binning up pixels,
           i.e. calculating spectra in different bins/portions of the map). Should be binary:
           0 means pixels to be omitted; must match shape of img.
        img2 : Optional[NDArray[Floating]]
           A counterpart image, may be used for calculating cross-spectra.
        """

        self._imsz = img.shape
        self.pixsize = pixsize
        if isinstance(pixunits,UnitBase):
            is_angle = pixunits.is_equivalent(u.deg)
            is_length = pixunits.is_equivalent(u.kpc)
            if is_angle or is_length:
                self.pixunits = pixunits
        else:
            if pixunits is None:
                # Maybe the user is just playing around and doesn't care about units.
                warnings.warn("No pixel units were input! Using u.arcsec; proceed with caution.")
                self.pixunits = u.arcsec
            else:
                print(pixunits)
                import pdb;pdb.set_trace()
                raise AttributeError("Pixel units must either be a length or angle.")
        self.mask_by_bin = None
        self.update_img(img,img2=img2)
        self.set_intrinsic_mask(intrinsic_mask)
        self.set_center(center=center)
        self._set_xyrmat()
        self.set_wave_numbers()


    def update_img(self,
                   img: NDArray[Floating],
                   img2: Optional[NDArray[Floating]] = None):
        """
        If astrometry remains the same, user is permitted to update images (perhaps they are running
        over several realizations).
        
        Parameters
        ----------
        img : NDArray[Floating]
           The image upon which power spectra will be calculated.
        img2 : Optional[NDArray[Floating]]
           A counterpart image, may be used for calculating cross-spectra.
        """

        self.img = img.astype(np.float64)
        self.img2 = None if img2 is None else img2.astype(np.float64)
        
    def set_center(self,center: Optional[ArrayLike] = None):
        """
        Parameters
        ----------
        center : Optional[ArrayLike]
           A two-element array-like object with center in pixel units, along axis 0 and axis 1,
           respectively. If none set, takes the center of the image.
        """
        
        if center is None:
            self.center  = [npix/2 for npix in self._imsz]
        else:
            self.center = center
            
    def _set_xyrmat(self):
        """
        Sets matrices useful for gridding. 
        """
        
        xvec=np.arange(self._imsz[0]) - self.center[0]
        yvec=np.arange(self._imsz[1]) - self.center[1]
        self._xmat=np.repeat([xvec],self._imsz[1],axis=0).transpose()
        self._ymat=np.repeat([yvec],self._imsz[0],axis=0)
        self.rmat=np.sqrt( (self._xmat-self.center[0])**2+(self._ymat-self.center[1])**2)*self.pixsize

    def set_intrinsic_mask(self,
                           intrinsic_mask: Optional[NDArray[Floating]]):
        """
        Parameters
        ----------
        intrinsic_mask : Optional[NDArray[Floating]]
           A mask that covers pixels that should be omitted (regardless of binning up pixels,
           i.e. calculating spectra in different bins/portions of the map). Should be binary:
           0 means pixels to be omitted; must match shape of img.
        """

        self.intrinsic_mask = np.ones(self._imsz) if intrinsic_mask is None else intrinsic_mask

    def set_manual_mask_by_bin(self,mask_by_bin):
        """
        Parameters
        ----------
        mask_by_bin : NDArray[Floating]
           A mask that labels pixels with integers corresponding to their bin number.
           (Bins should start at 1; 0 values are omitted)
        """
        self.mask_by_bin = mask_by_bin

    def set_annular_mask_by_bin(self,annular_edges):
        """
        Parameters
        ----------
        annular_edges NDArray[Floating] | Sequence[float]
           An array-like set of annular edges, in increasing order, defined by the radius.
        """

        rcout = 0
        mask_by_bin = np.zeros(self._imsz)
        for rin,rout in zip(annular_edges[:-1],annular_edges[1:]):
            rcout += 1                    # Keep track of ring count
            imcopy = np.zeros(self._imsz)  # Make a copy of image
            imcopy[(self.rmat >= rin)] = 1  # 
            imcopy[(self.rmat >= rout)] = 0  #
            mask_by_bin += imcopy * rcout
        self.mask_by_bin = mask_by_bin

    def set_wave_numbers(self,
                        kmin: Optional[Floating] = None,
                        kmax: Optional[Floating] = None,
                        nk_node: Optional[int] = None,
                        k_node: Optional[NDArray[Floating]] = None,
                        pad_max: Floating=3.0,
                        pad_min: Floating=2.0):
        """
        Parameters
        ----------
        kmin : Optional[Floating]
            The minimum wavenumber (in units of inverse pixunits) at which to calculate the
            power spectrum via the Mexican hat filter in Arevalo et al. (2012).
        kmax : Optional[Floating]
            The maximum wavenumber (in units of inverse pixunits) at which to calculate the
            power spectrum via the Mexican hat filter in Arevalo et al. (2012).
        nk_node : Optional[int]
            The number of nodes to use when calculating the power spectrum via the Mexican hat
            filter in Arevalo et al. (2012). Defaults to the times 2 divides into (kmax/kmin) --
            rounded up.
        k_node : Optional[NDArray[Floating]]
            A user-specified array of wavenumbers (nodes) at which to calculate the power
            spectrum via the Mexican hat filter in Arevalo et al. (2012).
        pad_min : Floating
            As compared to 1/(npix*pixsize), the minimum wavenumber would be pad_min/(npix*pixsize).
        pad_max : Floating
            As compared to 1/(pixsize), the maximum wavenumber would be 1/(pad_max*pixsize).
        """
        k, dkx, dky = uf.get_freqarr_2d(self._imsz[0],self._imsz[1],
                                     self.pixsize,self.pixsize)
        self._k_mat = k.astype(np.float64) # 2D matrix of k_r values corresponding to the image.

        if k_node is None:
            if kmin is None or kmax is None:
                self._set_default_krange()
            else:
                self.kmin = kmin
                self.kmax = kmax
            np2 = int(np.ceil(np.log(self.kmax/self.kmin)/np.log(2)))
            self.nk_node = np2 if nk_node is None else nk_node
            self.a12_kn = np.logspace(np.log10(self.kmin),np.log10(self.kmax),self.nk_node)
        else:
            self.a12_kn = k_node # Nodes at which power spectrum is calculated
        self.r2a   = 3600.0 * 180.0 / np.pi # Number of arcseconds in a radian
        self.a12_pk = None
        self.fft_kb = None # Bins within which power spectrum is averaged
        self.fft_pk = None

    def _set_default_krange(self,pad_max=3.0,pad_min=2.0):
        """
        Sets the range of wavenumbers (k) based on the image size and pixel size.
        
        Parameters
        ----------
        pad_min : Floating
            As compared to 1/(npix*pixsize), the minimum wavenumber would be pad_min/(npix*pixsize).
        pad_max : Floating
            As compared to 1/(pixsize), the maximum wavenumber would be 1/(pad_max*pixsize).
        """
        kmax = 1.0/(pad_max*self.pixsize)
        gmean = np.sqrt(np.prod(self._imsz)) # Geometric mean of x- and y-axes (number of pixels)
        kmin = pad_min/(gmean*self.pixsize)  #
        self.kmin = kmin
        self.kmax = kmax

    def ps_via_fft_2d(self,
                      cross: bool=False,
                      corr_n: bool=True):
        """
        Computes the power spectrum of the image via numpy FFT. It does not respect any mask.
        Sets attributes fft_kb (for the binned wavenumbers) and corresponding fft_pk, the binned
        power spectra.
        
        Parameters
        ----------
        cross : bool
            Compute the cross-spectrum? Requires that img2 is set.
        corr_n : bool
            Correct for the image size (factor of N that numpy FFT retains in its computation).
            Default is True (this will give the correct amplitude for plotting back to the user).
        """

        imgfft = np.fft.fft2(self.img)*self.pixsize
        if cross and self.img2 is not None:
            imgfft2 = np.fft.fft2(self.img2)*self.pixsize
        else:
            imgfft2 = imgfft

        fftMag = np.abs(imgfft*imgfft2)
        imgps = fftMag / self.img.size if corr_n else fftMag
        np2 = int(np.round(np.log(self._imsz[0])/np.log(2))*2.0) # No padding like for A12
        kb,pb,pe,pcnt = uf.bin_log2Ds(self._k_mat,imgps,nbins=np2*2,witherr=True,withcnt=True)
        self.fft_kb = kb
        self.fft_pk = pb
        
    def ps_via_a12_2d(self,
                      cross=False,
                      cummul_bins=False):
        """
        Computes the power spectrum of the image with regions (bins) as defined by masks,
        both the mask_by_bin and intrinsic_mask. Computation is performed by the Mexican hat
        filter presented in Arevalo et al (2012). The resultant power spectra (bin index is
        in axis 1) is set to the attribute a12_pk.
        
        Parameters
        ----------
        cross : bool
            Compute the cross-spectrum? Requires that img2 is set.
        cummul_bins : bool
            Allows the user to treat bins width greater indices as the union of it and their
            counterparts with lesser indices. For example, if mask_by_bin has only two bins
            (indicated by pixels of values 1 and 2, respectively), then here, bin 2 will be
            treated as the union of bins 2 and 1. (For 3 bins, bin 3 would be the union of bins
            3, 2, and 1.)
        """
        
        n_bins = 1 if self.mask_by_bin is None else int(self.mask_by_bin.max())
        self.a12_pk = np.zeros(self.a12_kn.shape+(n_bins,))
        #mask =  np.zeros(self._imsz)

        for i in range(n_bins):

            ### Construct mask for the bin
            if self.mask_by_bin is None:
                this_mask = np.ones(self._imsz)
            else:
                this_mask = np.zeros(self._imsz)
                if cummul_bins:
                    this_bin = (0 < self.mask_by_bin <= i+1) # Boolean indexing
                else:
                    this_bin = (self.mask_by_bin == i+1) # Boolean indexing
                this_mask[this_bin] = 1
            mask = this_mask * self.intrinsic_mask
            self.mask = mask.astype(np.float64)
            self._mybin = i # Bookkeeping, JIC; FFTs of masked images have associated bin number...
            self._a12_2D_at_bin(cross=cross)
            ### Now compute the Fourier transforms of the image(s) and mask.

    def _a12_2D_at_bin(self,cross=False):
        """
        Computes the power at the wavenumbers (nodes) specified by attribute a12_kn. The resultant power spectrum
        is set to the attribute a12_pk.

        Parameters
        ----------
        cross : bool
            Compute the cross-spectrum? Default is False. Requires img2 to compute.
        """
        
        ### Compute the Fourier transforms of the image(s) and mask.
        ### In principle, we are saving doing this Fourier transform at each k_r, for a given bin.
        if self.img2 is None or (cross == False):
            self._f_i2 = None
        else:
            self._f_i2   = np.fft.fft2(self.img2*self.mask)
        self._f_i = np.fft.fft2(self.img*self.mask) # FFT of the (masked) image
        self._f_m = np.fft.fft2(self.mask)  # FFT of the mask       

        self._n = self.mask.size
        self._m = self.mask.sum()
        
        for ii, k_r in enumerate(self.a12_kn):
            if k_r == 0:
                continue
            else:
                self._k_r = k_r
                self.a12_pk[ii,self._mybin] = self._a12_2D_at_k_r()
            
    def _a12_2D_at_k_r(self,eps=1.0e-3):
        """
        Computes the power at k_r (taken from the attribute k_r)

        Parameters
        ----------
        eps : Floating
             epsilon; relates to scaling/separation of the two sigmas. Arevalo+ (2012) liked the value 
             of 1e-3, though one can change this. We're happy to keep with 1e-3.

        Returns
        -------
        p_kr : Floating
            The power at the specified k_r
        """

        # Compute sigmas (in the same units as pixsize is defined, e.g. arcseconds)
        sig    = 1./(np.sqrt(2.*np.pi**2)*self._k_r)
        sig1   = sig/np.sqrt(1. + eps)
        sig2   = sig*np.sqrt(1. + eps)
        smoothk1 = np.exp(-2*self._k_mat**2*np.pi**2 * (sig1)**2)
        smoothk2 = np.exp(-2*self._k_mat**2*np.pi**2 * (sig2)**2)
        g1_i = np.real(np.fft.ifft2(self._f_i*smoothk1))
        g2_i = np.real(np.fft.ifft2(self._f_i*smoothk2))
        g1_m = np.real(np.fft.ifft2(self._f_m*smoothk1))
        g2_m = np.real(np.fft.ifft2(self._f_m*smoothk2))
                        
        mone = (self.mask > 0.0) # Allows for Boolean indexing; avoids division by zero
        if self._f_i2 is None:
            ###################  Calculate the autocovariance (auto power spectrum) ######################
            delt = np.zeros(g1_i.shape)                            
            delt[mone] = (g1_i[mone]/g1_m[mone] - g2_i[mone]/g2_m[mone]) # Avoid divide by zero
            df = delt*delt*self.mask                                     # Avoid divide by zero
        else:
            g1_i2 = np.real(np.fft.ifft2(self._f_i2*smoothk1))
            g2_i2 = np.real(np.fft.ifft2(self._f_i2*smoothk2))
            ###################  Calculate the cross-covariance (cross power spectrum) ######################
            delt1 = np.nan_to_num(np.divide(g1_i,g1_m))  - np.nan_to_num(np.divide(g2_i,g2_m))
            delt2 = np.nan_to_num(np.divide(g1_i2,g1_m)) - np.nan_to_num(np.divide(g2_i2,g2_m)) 
            df = delt1*delt2*self.mask # Arguably self.mask**2, but 1**2 = 1 and 0**2 = 0. So. Save some computation.
        s2_kr = (self._n/self._m) * np.mean(df) # A12 paper would use mean. 

        gamf   = np.pi # Can change for different dimensionality. But this function is specifically for 2D
        p_kr   = s2_kr /(eps**2 * gamf * self._k_r**2) # this is intrinsic PS(k)

        return p_kr

class ImagesFromPS:
    """
    Attributes
    ----------
    n_pix : int
        The number of pixels on an image side. Image will be square.
    pixsize : Floating
        Value of the pixel size. Units are at the user discretion, but must match the (inverse) units of
        the wavenumber values (kc, kdist, and k_arr).
    pixunits : UnitBase
        Units of the pixel size, as an astropy unit. Recommended to be set by user. Default is u.arcsec.
    slope : Floating
        The slope of the underlying power-law in the spectrum. Respects the convention P(k) = P_0 k**slope.
    kc : Floating
        The wavenumber for the spectral cutoff (towards low wavenumbers). Value should correspond to inverse
        of units corresponding to pixel size. If pixsize is taken as a length (e.g. kpc), then kc should be
        reported in kpc**(-1).
    p0 : Floating
        The spectral normalization.
    kdis : Floating
        The wavenumber corresponding the dissipation scale. (Again, respective to units of pixsize).
    eta_c : Floating
        The rate of cutoff. Default is 4.0, as adopted in Khatri & Gaspri (2016).
    eta_d : Floating
        The rate of dissipation. Default is 1.5, as adopted in Khatri & Gaspri (2016).
    seed : Union[Floating,int]
        A seed for the random number generator, if desired.
    k_arr : Floating
        Positive wavenumbers at which the power spectrum is modeled.

    Methods
    -------
    set_ps_parameters():
        Sets the parameters for a cannonical turbulent power spectrum.
    set_image_size():
        Sets attributes n_pix, pixsize, pixunits, and if desired center.
    set_minfactor():
        Sets a factor which governs the lowest k-value in
    get_parameterized_ps():
        Get an array of the power spectrum at input wavenumbers.
    get_logspaced_k():
        Get an array of logarithmically spaced wavenumbers.
    set_seed():
        Set the seed for random number generator
    generate_realization():
        Generates a realization from the input power spectrum.
    set_center():
        Sets the center of an image. (Not necessary for many methods).
    """

    def __init__(self,
                 n_pix: int = 1024,
                 pixsize: Floating = 1.0,
                 pixunits : Optional[UnitBase] = None,
                 slope: Floating = 0.0,
                 kc: Floating = 1e-3,
                 p0: Floating = 1e0,
                 kdis: Floating = 1e3,
                 eta_c: Floating = 4.0,
                 eta_d: Floating = 1.5,
                 minfactor: Floating = 1e2,
                 seed: Optional[Union[Floating,int]] = None,
                 no_warn: bool = False):
        """
        All wavenumber values (kc, kdis, k) should match those for the inverse of the units of pixsize.
        For example, if pixsize=1.0 means that each pixel is 1 kpc on a side, then the corresponding k-values
        should denote inverse kpc units.

        Parameters
        ----------
        n_pix : int
            The number of pixels on an image side. Image will be square.
        pixsize : Floating
            Value of the pixel size.
        pixunits : Optional[UnitBase]
           It is highly recommend to explicitly set this. Default None type becomes u.arcsec;
           user is warned.
        slope : Floating
            The slope of the underlying power-law in the spectrum. Respects the convention P(k) = P_0 k**slope.
        kc : Floating
            The wavenumber for the spectral cutoff (towards low wavenumbers). Value should correspond to inverse
            of units corresponding to pixel size. If pixsize is taken as a length (e.g. kpc), then kc should be
            reported in kpc**(-1).
        p0 : Floating
            The spectral normalization.
        kdis : Floating
            The wavenumber corresponding the dissipation scale. (Again, respective to units of pixsize).
        eta_c : Floating
            The rate of cutoff. Default is 4.0, as adopted in Khatri & Gaspri (2016).
        eta_d : Floating
            The rate of dissipation. Default is 1.5, as adopted in Khatri & Gaspri (2016).
        minfactor : Floating
            If wavenumber array includes zero values, they will be reset to the minimum non-zero wavenumber
            divided by this value.
        seed : Optional[Union[Floating,int]]
            A seed for the random number generator, if desired.
        no_warn : bool
            Ignore warnings. Default is False.
        """

        ### Corresponding to a paremeterized power spectrum
        self.set_ps_parameters(slope=slope,kc=kc,p0=p0,kdis=kdis,eta_c=eta_c,eta_d=eta_d)

        ### Corresponding to the map (image).
        self.set_image_size(n_pix,pixsize,no_warn=no_warn)

        ### To control for numerical accuracy
        self.set_minfactor(minfactor)

        self._rng = np.random.default_rng(seed=seed)
        
    def set_ps_parameters(self,
                 slope: Floating = 0.0,
                 kc: Floating = 1e-3,
                 p0: Floating = 1e0,
                 kdis: Floating = 1e3,
                 eta_c: Floating = 4.0,
                 eta_d: Floating = 1.5):
        """
        All wavenumber values (kc, kdis, k) should match those for the inverse of the units of pixsize.
        For example, if pixsize=1.0 means that each pixel is 1 kpc on a side, then the corresponding k-values
        should denote inverse kpc units.

        Parameters
        ----------
        slope : Floating
            The slope of the underlying power-law in the spectrum. Respects the convention P(k) = P_0 k**slope.
        kc : Floating
            The wavenumber for the spectral cutoff (towards low wavenumbers). Value should correspond to inverse
            of units corresponding to pixel size. If pixsize is taken as a length (e.g. kpc), then kc should be
            reported in kpc**(-1).
        p0 : Floating
            The spectral normalization.
        kdis : Floating
            The wavenumber corresponding the dissipation scale. (Again, respective to units of pixsize).
        eta_c : Floating
            The rate of cutoff. Default is 4.0, as adopted in Khatri & Gaspri (2016).
        eta_d : Floating
            The rate of dissipation. Default is 1.5, as adopted in Khatri & Gaspri (2016).      
        """
        self.slope=slope   # Power-law slope (k**slope)
        self.kc = kc       # Cutoff scale
        self.p0 = p0       # Normalization
        self.kdis = kdis   # Disipation scale
        self.eta_c = eta_c # Cutoff exponent
        self.eta_d = eta_d # Dissipation exponent
        
    def set_minfactor(self, minfactor: Floating = 1e2):
        """

        Parameters
        ----------
        minfactor : Floating
            If wavenumber array includes zero values, they will be reset to the minimum non-zero wavenumber
            divided by this value.     
        """

        self._minfactor = minfactor # If min(k)==0, replace with smallest non-zero k divided by this factor        
        
    def set_image_size(self,
                       n_pix: int = 1024,
                       pixsize: Floating = 1.0,
                       pixunits: Optional[UnitBase] = None,
                       center: Optional[ArrayLike] = None,
                       no_warn: bool = False):
        """

        Parameters
        ----------
        n_pix : int
            The number of pixels on an image side. Image will be square.
        pixsize : Floating
            Value of the pixel size.      
        pixunits : Optional[UnitBase]
           It is highly recommend to explicitly set this. Default None type becomes u.arcsec;
           user is warned.
        center : Optional[ArrayLike]
           A two-element array-like object with center in pixel units, along axis 0 and axis 1,
           respectively. If none set, takes the center of the image.
        no_warn : bool
           If True, then no warnings about pixunits. Default is False.
        """

        self.n_pix = n_pix
        self.pixsize = pixsize        
        k,dkx,dky = uf.get_freqarr_2d(n_pix, n_pix, pixsize, pixsize)
        self._imsz = k.shape
        kflat = k.flatten()
        gki = (kflat > 0)
        gk = kflat[gki]
        self._kflat = kflat
        self.k_arr = gk
        self._gki = gki
        
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

        self.set_center(center)
        self._set_xyrmat()
            
    def set_center(self,center: Optional[ArrayLike] = None):
        """
        Parameters
        ----------
        center : Optional[ArrayLike]
           A two-element array-like object with center in pixel units, along axis 0 and axis 1,
           respectively. If none set, takes the center of the image.
        """
        
        if center is None:
            self.center  = [npix/2 for npix in self._imsz]
        else:
            self.center = center
            
    def _set_xyrmat(self):
        """
        Sets matrices useful for gridding. 
        """
        
        xvec=np.arange(self._imsz[0]) - self.center[0]
        yvec=np.arange(self._imsz[1]) - self.center[1]
        self._xmat=np.repeat([xvec],self._imsz[1],axis=0).transpose()
        self._ymat=np.repeat([yvec],self._imsz[0],axis=0)
        self.rmat=np.sqrt( (self._xmat-self.center[0])**2+(self._ymat-self.center[1])**2)*self.pixsize
        
    def get_parameterized_ps(self,k_in: Optional[NDArray[Floating]] = None):
        """

        Parameters
        ----------
        k_in : Optional[NDArray[Floating]]
            If provided, an array of wavenumbers at which to calculate the power spectral values. Otherwise adopts
            the attribute k_arr.

        Returns
        -------
        ps : NDArray[Floating]
        """

        k_arr = self.k_arr if k_in is None else k_in
        keqz = (k_arr == 0)
        kgtz = (k_arr > 0)
        kmin = np.min(k_arr[kgtz])
        if np.sum(keqz) > 0:
            k_arr[keqz] = kmin/self._minfactor # Avoid k=0, but get a really small value.
        ps  = self.p0*k_arr**(self.slope) * np.exp(-(self.kc/k_arr)**self.eta_c) * np.exp(-(k_arr/self.kdis)**self.eta_d)

        return np.nan_to_num(ps)

    def get_logspaced_k(self,
                        kmin: Optional[Floating] = None,
                        kmax: Optional[Floating] = None,
                        n_pts: int = 500):
        """
        Computers and returns an array of wavenumbers based on the inputs.
        
        Parameters
        ----------
        kmin : Optional[Floating]
            The minimum wavenumber (in units of inverse pixunits). Defaults to 1/(n_pix * pixsize)
        kmax : Optional[Floating]
            The maximum wavenumber (in units of inverse pixunits). Defaults to 1/(pixsize)
        n_pts: int
            The number of elements when making the array of wavenumbers.
        """
                        
        mykmin = 1.0/(self.n_pix * self.pixsize) if kmin is None else kmin
        mykmax = 1.0/(self.pixsize) if kmax is None else kmax
        k = np.logspace(np.log10(mykmin),np.log10(mykmax),n_pts)

        return k

    def set_seed(self,seed):
        """
        Sets the seed for the random number generator.

        Parameters
        ----------
        seed : Union[Floating,int]
            A seed for a random number generator. Any value will do.
        """
        
        self._rng = np.random.default_rng(seed=seed)
    
    def generate_realization(self,
                             k_in: Optional[NDArray[Floating]] = None,
                             ps_in: Optional[NDArray[Floating]] = None,
                             seed: Optional[Union[Floating,int]] = None):
        """
        Generate an image realization based on either the internally set power spectrum, or if user wants to
        supply a completely independently calculated one, this can be done too.

        Parameters
        ----------
        k_in : Optional[NDArray[Floating]]
            An array of wavenumbers. This should extend beyond the range of the (non-zero) wavenumbers as
            found in the FFT of the image. (Performs interpolation, not extrapolation).
        ps_in : Optional[NDArray[Floating]]
            A corresponding array of the power spectral values.
        seed : Optional[Union[Floating,int]]
            If wanting to set the seed, the user can.

        Returns
        -------
        img : NDArray[Floating]
            A realization of the input power spectrum.
        """

        psarr = self._kflat*0
        if k_in is None or ps_in is None:
            psout = self.get_parameterized_ps(self.k_arr)
        else:
            psout = np.exp(np.interp(np.log(self.k_arr),np.log(k_in),np.log(ps_in)))
        psarr[self._gki] = psout
        ps2d = psarr.reshape(self.n_pix,self.n_pix) * self.n_pix * self.n_pix

        if seed is not None:
            self.set_seed(seed)
        phase = self._rng.uniform(size=(self.n_pix,self.n_pix))*2*np.pi
        newfft = np.sqrt(ps2d) * np.exp(1j*phase)
        img = np.real(np.fft.ifft2(newfft/self.pixsize))
        img *= np.sqrt(2.0)

        return img

#################################################################################

class MultiGaussBeam:
    """
    For a beam that can be characterized as the sum of any number of 2D Gaussians,
    this provides a method to compute the imparted bias in the recovered power
    spectrum from the Areval et al (2012) Mexican hat methodology of power estimation.

    Attributes
    ----------
    norms : NDArray[Floating]
        Normalization (amplitude) of the compoment Gaussians
    widths : NDArray[Floating]
        Widths (Gaussian sigmas) of the components (with corresponding normalizations)
    

    Methods
    -------
    calc_ft_at_k():
        Computes the Fourier transform of a multi-Gaussian beam.
    calc_ps_at_k():
        Computes the power spectrum of a multi-Gaussian beam.
    get_multi_gauss_bias():
        Computes the bias due to a multi-Gaussian beam (PSF)
    """
    
    def __init__(self,
                 norms: NDArray[Floating] | Sequence[float],
                 widths: NDArray[Floating] | Sequence[float]):
        """
        Define a beam (or point-spread function, PSF) as multiple Gaussians via a list of
        normalizations (height) and widths (Gaussian sigmas).

        Parameters
        ----------
        norms : NDArray[Floating] | Sequence[float]
            array-like collection of Gaussian normalizations.
            Internally, the sum of norms will be normalized to equal unity
        widths : NDArray[Floating] | Sequence[float]
            array-like collection of Gaussian standard deviations.
        """
        self.norms = np.array(norms,dtype=float)
        self.norms /= np.sum(self.norms) # Impose unitary normalization
        self.widths = np.array(widths,dtype=float)

    def calc_ft_at_k(self,k: NDArray[Floating]):
        """
        Computes the Fourier transform of a multi-Gaussian Point Spread Function

        Parameters
        ----------
        k : NDArray[Floating]
            Array of wavenumber points at which to compute the power spectrum of the point spread function (PSF).

        Returns
        -------
        ft : NDArray[Floating]
            The Fourier transform of the point spread function (PSF) at the specified wavenumbers (k).
        """
        
        ft = np.zeros(k.shape)    
        for n,s in zip(self.norms,self.widths):
            t = 2 * np.pi**2 * s**2
            ft += n * np.exp(-k**2 * t)
        
        return ft

    def calc_ps_at_k(self,k: NDArray[Floating]):
        """
        Computes the power spectrum for a multi-Gaussian Point Spread Function

        Parameters
        ----------
        k : NDArray[Floating]
            Array of wavenumber points at which to compute the power spectrum of the point spread function (PSF).

        Returns
        -------
        psf_ps : NDArray[Floating]
            The power spectrum of the point spread function (PSF) at the specified wavenumbers (k).
        """
        psf_pt = self.calc_ft_at_k(k)

        return psf_ft**2
    
    def _get_multi_gauss_terms(self,karr: NDArray[Floating],alpha: Floating):
        """

        This corrects for the bias noted in Romero+ 2023, Romero 2024.
        Code assumes n_gauss are few (such that for-loops are cheap); could be optimized if n_gauss is
        huge (but you've probably done something wrong to have n_gauss that large).

        Parameters
        ----------
        karr : NDArray[Floating]
            A one-dimensional array of wavenumbers (e.g. inverse arcseconds). The units of karr must be
            the inverse of the units used for the sigma(s) describing the N-Gaussian (below).
        alpha : Floating 
             The spectral index assumed. [Convention: P(k) = P0 k**(-alpha)]
        """
    
        ndim = 2
        expo = (ndim/2 + 2 - alpha/2)

        kshape = karr.shape
        if len(kshape) > 1:
            raise IndexError("karr Must be one-dimensional")

        corrs = np.zeros(kshape[0])

        for norm1,sig1 in zip(self.norms,self.widths):
            k1 = 1.0/(np.sqrt(2)*np.pi*sig1)
            x1 = k1/karr
            n1 = norm1 * np.ones(kshape[0])
            for norm2,sig2 in zip(self.norms,self.widths):
                k2 = 1.0/(np.sqrt(2)*np.pi*sig2)
                x2 = k2/karr
                n2 = norm2 * np.ones(kshape[0])

                coef = n1*n2
                nume = (2*x1**2 * x2**2 + x1**2 + x2**2)
                deno = 2.0*x1**2 * x2**2

                term = coef * (nume/deno)**(-expo)
                corrs += term

        return corrs

    def get_multi_gauss_bias(self,
                             karr: NDArray[Floating],
                             alpha: Floating,
                             ign_psf:bool = False,
                             pb_only: bool = False):

        """
        Corrects for:
        (1) The scalar bias induced by Arevalo+ 2012
        (2) The non-scalar bias induced by a beam *and* use of Arevalo+ 2012
        --- In order for this to be done, some spectral index, alpha MUST be assumed! ---
        --- where P(k) = P0 * k**-alpha                                              ---
        Note that there is a correction for the power spectrum of the beam (PSF) within THIS term.
        If you want to correct for the PSF and the bias then these terms cancel! In light of this,
        I've added a keyword to allow you to ignore the PS term.

        Parameters
        ----------
        karr : NDArray[Floating]
            The array of k (wavenumber) values at which to calculate the total bias
        alpha : Floating 
            The assumed spectral index (convention given above)
        ign_psf : bool
            As mentioned above, allows you to ignore the PSF power spectrum term with respect
            to this bias. Given that you want to correct for the PSF power spectrum, they will
            cancel each other out when correcting your measured power spectrum. You can bypass
            the additional calculations by ignoring the PSF term here.
        pb_only : bool
            PSF BIAS only. That is, ignore the scalar bias due to an underlying power-law.

        Returns
        -------
        bias : NDArray[Floating]
            The bias due to the PSF; the precise bias being quantified depends on boolean option
            inputs.
        """
        ndim = 2
        lilg = (ndim/2 + 2 - alpha/2)

        # Split the power-law bias into 3 parts. There were derived in Arevalo et al (2012)
        p1 = 2**(alpha/2)
        p2 = self._get_multi_gauss_terms(karr,alpha)
        p3 = sps.gamma(lilg) / sps.gamma(ndim/2 + 2)

        # This correction is derived in Romero et al (2023)
        psf_ps = 1.0 if ign_psf else self.calc_ps_at_k(karr)

        bias  = p2/psf_ps if pb_only else p1*p2*p3 / psf_ps

        return bias

