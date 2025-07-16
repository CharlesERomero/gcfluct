import numpy as np
import UtilityFunctions as UF
import scipy.special as sps
import astropy.units as u
from astropy.units import Quantity
from typing import Union, List, Optional
from numpy.typing import NDArray, ArrayLike
import warnings

class PSfromImages:

    """

    """
    def __init__(self,
                 img: NDArray[np.floating],
                 pixsize: np.floating,
                 pixunits: Optional[Quantity],
                 center:  = Optional[ArrayLike[np.floating]] = None,
                 intrinsic_mask: Optional[NDArray[np.floating]] = None,
                 img2: Optional[NDArray[np.floating]] = None,
                 kpcperarcsec: np.floating = 0.0):
        """
        Initialization of PSfromImages
        
        Parameters
        ----------
        img : NDArray[np.floating]
            The image upon which you want to calculate power spectra.
        pixsize : np.floating
            The numerical value for the pixel size (e.g. if 1 arcsecond, then pixsize = 1 and
            pixunits = u.arcsec)
        pixunits : Optional[Quantity]
            An astropy quantity to which pixsize corresponds. This should be either an angle or a length,
            e.g. u.arcsec or u.kpc.
            WARNING! If this is not set (OK for exploration), it will assume arcseconds. 
        center : Optional[ArrayLike[np.floating]]]
            The center of the target in pixel-space.
        intrinsic_mask : Optional[NDArray[np.floating]]
            Mask valid regardless of dividing target up into regions (bins). Mask would omit bad pixels
            (e.g. chip gaps, hot pixels, or pixels subtended by contaminating signal).
        img2 : Optional[NDArray[np.floating]]
            Used only if computing a cross-spectrum
        kpcperarcsec : np.floating
            Scale, i.e. physical distance (in kpc) per angle (arcseconds). Defaults to 0 (unphysical),
            so the value is defined, but it will not try to convert units.
        """

        self.imsz = img.shape
        self.pixsize = pixsize
        
        ### NB, yes astropy has the @u.quantity_input decorator, but I would have to define
        ### an equivalence between angle and length, which requires additional information
        ### that defeats the simplest use-case for this. So, I approach it this way.
        if isinstance(pixunits,Quantity): # Astropy quantity.
            # unitless angle conversion is risky; I will not accommodate it
            is_angle = pixunits.unit.is_equivalent(u.arcsec) # Explicit angle
            is_length = pixunits.unit.is_equivalent(u.kpc) # Explicit length
            if is_angle or is_length:
                self.pixunits = pixunits
            else:
                raise u.UnitTypeError("Pixel units must be (explicitly) an angle or length.")
        else:
            if pixunits is None:
                pixunits = u.arcsec
                warning.warn("You did not specify pixel units; assuming arcseconds!")
            else:
                raise ValueError("Pixel units must be an angle or length.")
        
        self.mask_by_bin = np.ones(self.imsz)
        self.update_img(img,img2=img2)
        self.set_intrinsic_mask(intrinsic_mask)
        self.set_center(center=center)
        self.set_waveNumbers()
    
    def update_img(self,img,img2=None):
        """ 
        Sets the image (and potentially second image, used for cross-spectra).
        
        Parameters
        ----------
        img : NDArray[np.floating]
            Image upon which spectra will be calculated.
        img2 : Optional[NDArray[np.floating]]
            Optional image with which cross-spectra will be calculated.
        
        """
        self.img = img.astype(np.float64)
        self.img2 = None if img2 is None else img2.astype(np.float64)
        
    def set_center(self,center=None):
        """ 
        Sets the center of the target. If no center is provide, it assumes the center of the image is the
        center of the (astronomical) target in question.
        
        Parameters
        ----------
        center : Optional[ArrayLike[np.floating]]
            A tuple, list or array with 2 elements (pixel centers in x- (0-axis) and y- (1-axis) coordinates,
            respectively)
        """

        if center is None:
            self.center  = [npix/2 for npix in self.imsz]
        else:
            self.center = center

        self._set_xyrmat()
            
    def _set_xyrmat(self):
        """
        From attributes image and pixsize, I set up some internal attributes which are used later
        and may be of use to the user.
        """
        xvec=np.arange(self.imsz[0])
        yvec=np.arange(self.imsz[1])
        self._xmat_pix=np.repeat([xvec],self.imsz[1],axis=0).transpose() # In units of pixels
        self._ymat_pix=np.repeat([yvec],self.imsz[0],axis=0) # In units of pixels
        # rmat is in the same units as pixsize.
        self.rmat=np.sqrt( (self._xmat_pix - self.center[0])**2 +
                           (self._ymat_pix - self.center[1])**2) * self.pixsize

    def set_intrinsic_mask(self,intrinsic_mask: NDArray[np.floating]):
        """ 
        Sets an intrinsic mask (to the entire image). Examples of pixels to be masked for this would be:
        (1) those origininating due to detector chip gaps, (2) hot pixels, (3) those subtended by
        compact sources, or (4) those subtended by any other contaminating source.
        
        Parameters
        ----------
        intrinsic_mask : Optional[NDArray[np.floating]]
            User supplied mask; must be the same shape as the attribute img.
        """

        self.intrinsic_mask = np.ones(self.imsz) if intrinsic_mask is None else intrinsic_mask

    def set_manual_mask_by_bin(self,mask_by_bin: NDArray[np.floating]):
        """ 
        Sets the mask_by_bin to a user-defined bin where elements (pixels) are assigned the value
        of their bin. This means bins, *by default* cannot overlap, with the exception of cummulative
        binning (see method ps_via_a12_2D).
        
        Parameters
        ----------
        mask_by_bin : NDArray[np.floating]
            User supplied mask (by bin); must be the same shape as the attribute img.
        """
        self.mask_by_bin = mask_by_bin

    def set_annular_mask_by_bin(self,annular_edges: ArrayLike[np.floating]):
        """ 
        Defines a (set of masks) by bin. In particular, elements (pixels) are assigned the value
        of their bin. This means bins, *by default* cannot overlap with the exception of cummulative
        binning (see method ps_via_a12_2D). In particular, this method takes a user-supplied array-like
        series of edges which define the edges of the rings. No gaps are assumed such that
        n_annuli = len(annular_edges) - 1.
        
        Parameters
        ----------
        annular_edges : ArrayLike[np.floating]
            The edges at which 
        """
        rcout = 0
        mask_by_bin = np.zeros(self.imsz)
        for rin,rout in zip(annular_edges[:-1],annular_edges[1:]):
            rcout += 1                    # Keep track of ring count
            imcopy = np.zeros(self.imsz)  # Make a copy of image
            imcopy[(rmat >= rin)]  = 1  # 
            imcopy[(rmat >= rout)] = 0  #
            mask_by_bin += imcopy * rcout
        self.mask_by_bin = mask_by_bin

    def set_wave_numbers(self,
                        kmin: Optional[np.floating] = None,
                        kmax: Optional[np.floating] = None,
                        nk_node: Optional[np.floating] =None,
                        pad_max: np.floating = 3.0,
                        pad_min: np.floating = 2.0,
                        k_node: Optional[np.floating] = None):
        """ 
        Sets arrays of wavenumbers useful for many subsequent methods.
        
        Parameters
        ----------
        kmin : Optional[np.floating]
            Minimum wavenumber at which the value of the power spectrum will be calculated via
            the Arevalo et al (2012) method. If None, the a default value is adopted from the size
            of the image (with dependence on attribute pixsize) and pad_min.
        kmax : Optional[np.floating]
            Maximum wavenumber at which the value of the power spectrum will be calculated via
            the Arevalo et al (2012) method. If None, the a default value is adopted from the
            attribute pixsize and inpute parameter pad_max.
        nk_node : Optional[np.floating]
            Number of wavenumber nodes at which to calculate the power spectrum. By default, this
            is the number of times that 2 factors (divides) into kmax/kmin, rounded up.        
        pad_max : np.floating
            Used in defining kmax if a user supplied value of kmax is not provided
        pad_min : np.floating
            Used in defining kmin if a user supplied value of kmax is not provided
        k_node : Optional[NDArray[np.floating]]
            If an array of wavenumbers is supplied, then the above 
        """
        
        k, dkx, dky = get_freqarr_2d(self.imsz[0],self.imsz[1],
                                     self.pixsize,self.pixsize)
        self.k_mat = k.astype(np.float64) # 2D matrix of k_r values corresponding to the image.

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
        self.a12_pk = None
        self.fft_kb = None # Bins within which power spectrum is averaged
        self.fft_pk = None

    def _set_default_krange(self,
                            pad_max: np.floating = 3.0,
                            pad_min: np.floating = 2.0):
        """ 
        Calculates the power spectra via numpy's FFT for the defined bins. As a simple FFT won't
        (properly) handle a mask, the user should understand this. (No apodizations are currently
        implemented) Sets attributes kmin and kmax accordingly.
        
        Parameters
        ----------
        pad_max : np.floating
            Where the default maximum wavenumber is 1/pixsize, this reduces it by the factor
            pad_max. For pad_max < 2.0, one will start to see numerical errors due to precision
            error with the Fourier transforms and small smoothing kernels.
        pad_min : np.floating
            Where the default minimum wavenumber would be 1/(npix * pixsize), this increase
            the minimum by a factor pad_min.
        """

        kmax = 1.0/(pad_max*self.pixsize)
        gmean = np.sqrt(np.prod(self.imsz)) # Geometric mean of x- and y-axes (number of pixels)
        kmin = pad_min/(gmean*self.pixsize)  #
        self.kmin = kmin
        self.kmax = kmax

    def ps_via_fft_2D_nomask(self,cross: bool=False,corr_n: bool=True):
        """ 
        Calculates the power spectra via numpy's FFT for the defined bins. As a simple FFT won't
        (properly) handle a mask, the user should understand this. (No apodizations are currently
        implemented)
        
        Parameters
        ----------
        cross : bool
            Compute the cross-spectrum? Defaults to False. Also requires that the attribute img2
            is set.
        corr_n : bool
            Correct for the number of elements (size) of the array. By default, numpy's FFT will
            include a factor of this size, which is handled correctly with numpy's inverse FFT.
            However, if you want the actual values for power spectra, then you want to use corr_n.
        """
             
        imgfft = np.fft.fft2(self.img)*self.pixsize
        if cross and self.img2 is not None:
            imgfft2 = np.fft.fft2(self.img2)*self.pixsize
        else:
            imgfft2 = imgfft

        fft_mag = np.abs(imgfft*imgfft2)
        imgps = fft_mag / self.img.size if corr_n else fft_mag
        np2 = int(np.round(np.log(self.imsz[0])/np.log(2))*2.0) # No padding like for A12
        kb,pb,pe,pcnt = UF.bin_log2Ds(self.k_mat,imgps,nbins=np2,witherr=True,withcnt=True)
        self.fft_kb = kb
        self.fft_pk = pb
        
    def ps_via_a12_2D(self,cross: bool=False,cumm_bins: bool=False):
        """ 
        Calculates the power spectra for the defined bins via the Mexican hat
        (a type of delta variance) method proposed in Arevalo et al. (2012).
        This implementation is only valid for 2D data sets.
        
        Parameters
        ----------
        cross : bool
            Compute the cross-spectrum? Defaults to False. Also requires that the attribute img2
            is set.
        cumm_bins : bool
            Use cummulative bins. If this is set and, for example, two bins exist (bin 1 and bin 2),
            then bin 2 will (additionally) include all the pixels in bin 1.
        """
                
        nBins = self.mask_by_bin.max() 
        self.a12_pk = np.zeros(self.a12_kn.shape+(nBins,))
        #mask =  np.zeros(self.imsz)

        for i in range(nBins):

            ### Construct mask for the bin. A bit excessive if mask_by_bin is all ones, but
            ### that case is not gauranteed! This generalizes the handling of the mask.
            this_mask = np.zeros(self.imsz)
            if cumm_bins:
                this_bin = (self.mask_by_bin <= i+1) # Boolean indexing
            else:
                this_bin = (self.mask_by_bin == i+1) # Boolean indexing
            this_mask[this_bin] = 1
            mask = this_mask * self.intrinsic_mask
            self.mask = mask.astype(np.float64)
            self.mybin = i # Bookkeeping, JIC; FFTs of masked images have associated bin number...
            self._a12_2D_at_bin(cross=cross)
            ### Now compute the Fourier transforms of the image(s) and mask.

    def _a12_2D_at_bin(self,cross: bool = False):
        """ 
        For a given bin (an image and a specific mask), this computes the power spectrum via the
        Mexican hat (a type of delta variance) method proposed in Arevalo et al. (2012). That is,
        it wraps over the computation at each wavenumber (k_r) for the set (array) of wavenumbers
        of interest.

        Parameters
        ----------
        cross: bool
            Compute the cross-spectrum? Defaults to False. Also requires that the attribute img2
            is set.
        """
        
        ### Compute the Fourier transforms of the image(s) and mask.
        ### Motivation: we are saving doing this Fourier transform at each k_r, for a given bin.
        ### Conversely, the inverse FFT of each (convolved-equivalent image) is still done at each
        ### k_r. (Potential performance gains could be sought here)
        if self.img2 is None or (cross == False):
            self._f_i2 = None
        else:
            self._f_i2   = np.fft.fft2(self.img2*self.mask)
        self._f_i = np.fft.fft2(self.img*self.mask) # FFT of the (masked) image
        self._f_m = np.fft.fft2(self.mask)  # FFT of the mask       

        self._n = self.mask.size
        self._m = self.mask.sum()

        ### NB, this for-loop could be multi-threaded/MPI'd
        for ii, k_r in enumerate(self.a12_kn):
            if k_r == 0:
                continue
            else:
                self._k_r = k_r
                self.a12_pk[ii,self.mybin] = self._a12_2D_at_k_r()
            
    def _a12_2D_at_k_r(self,eps: np.floating = 1.0e-3) -> np.floating:
        """ 
        The foundational computation of the power spectrum as per the delta variance method
        presented in Arevalo et al. (2012). They use a Mexican hat filter that approximates a
        delta function for their delta variance method.

        Parameters
        ----------
        eps : np.floating
            In Arevalo et al. (2012), the epsilon parameter which designates the scaling to the
            two Gaussians to use in making a Mexican hat which acts as a delta function in this
            flavor of "delta variance" methods. Arevalo et al. (2012) adopt epsilon = 1e-3; we
            do the same (as a default).
        
        Returns
        -------
        p_kr : np.floating
            The power spectrum value at k_r
        """

        # Compute sigmas (in the same units as pixsize is defined, e.g. arcseconds)
        sig    = 1./(np.sqrt(2.*np.pi**2)*self._k_r)
        sig1   = sig/np.sqrt(1. + eps)
        sig2   = sig*np.sqrt(1. + eps)
        smoothk1 = np.exp(-2*self.k_mat**2*np.pi**2 * (sig1)**2)
        smoothk2 = np.exp(-2*self.k_mat**2*np.pi**2 * (sig2)**2)
        g1_i = np.real(np.fft.ifft2(self._f_i*smoothk1))
        g2_i = np.real(np.fft.ifft2(self._f_i*smoothk2))
        g1_m = np.real(np.fft.ifft2(self._f_m*smoothk1))
        g2_m = np.real(np.fft.ifft2(self._f_m*smoothk2))
                        
        mone = (self.mask > 0.0) # Allows for Boolean indexing; avoids division by zero
        if self._f_i2 is None:
            ###################  Calculate the autocovariance (auto power spectrum) ######################
            delt = np.zeros(g1_i.shape)                            
            delt[mone] = (g1_i[mone]/g1_m[mone] - g2_i[mone]/g2_m[mone]) # Avoid divide by zero
            delt_full = delt*delt*self.mask                                     # Avoid divide by zero
        else:
            g1_i2 = np.real(np.fft.ifft2(self._f_i2*smoothk1))
            g2_i2 = np.real(np.fft.ifft2(self._f_i2*smoothk2))
            ###################  Calculate the cross-covariance (cross power spectrum) ######################
            delt1 = np.nan_to_num(np.divide(g1_i,g1_m))  - np.nan_to_num(np.divide(g2_i,g2_m))
            delt2 = np.nan_to_num(np.divide(g1_i2,g1_m)) - np.nan_to_num(np.divide(g2_i2,g2_m)) 
            delt_full = delt1*delt2*self.mask # Arguably self.mask**2, but 1**2 = 1 and 0**2 = 0. So. Save some computation.
        s2_kr = (self._n/self._m) * np.mean(delt_full) # mean is correct in this case.
        
        gamf   = np.pi # Can change for different dimensionality. But this function is specifically for 2D
        p_kr   = s2_kr /(eps**2 * gamf * self._k_r**2) # this is intrinsic PS(k)

        return p_kr

class ImagesFromPS:

    def __init__(self,n_pix=1024,pixsize=1.0,slope=0.0,kc=1e-3,p0=1e0,kdis=1e3,eta_c=4.0,eta_d=1.5,minfactor=1e2,
                 seed=None):
        """
        Constructor for ImagesFromPS.
        All wavenumber values (kc, kdis, k) should match those for the inverse of the units of pixsize.
        For example, if pixsize=1.0 means that each pixel is 1 kpc on a side, then the corresponding k-values
        should denote inverse kpc units.

        Parameters
        ----------
        n_pix : int
            The number of pixels on a side. Images are restricted to squares (same pixels on each side).
        pixsize: Union[ np.floating, int ]
            Pixels are also taken to be squares (same length on each side). The value should correspond
            to the length in the units that match the (inverted) units of parameters associated with
            wavenumbers (kc, k_dis).
        slope : np.floating
            The slope of the power spectrum. [Sign convention: P = P0 * k**slope].
            The default value is 0.0
        kc : np.floating
            The "cutoff" wavenumber. Default value is 1e-3.
        p0 : np.floating
            The normalization of the power spectrum [Sign convention: P = P0 * k**slope]
        kdis : np.floating
            The dissipation wavenumber.
        eta_c : np.floating
            Cutoff rate. Defaults to 4.0, used in Khatri & Gaspari (2016)
        eta_d : np.floating
            Dissipation rate. Defaults to 1.5, used in Khatri & Gaspari (2016)
        minfactor : np.floating
            What factor (1/minfactor) of the smallest non-zero k will we adopt as a new smallest k?        
        """

        ### Corresponding to a paremeterized power spectrum
        self.set_ps_parameters(slope=slope,kc=kc,p0=p0,kdis=kdis,eta_c=eta_c,eta_d=eta_d)

        ### Corresponding to the map (image).
        self.set_image_size(n_pix,pixsize)

        ### To control for numerical accuracy
        self.set_minfactor(minfactor)

        self.rng = np.random.default_rng(seed=seed)
        
    def set_ps_parameters(self,
                          slope: np.floating = 0.0,
                          kc: np.floating = 1e-3,
                          p0: np.floating = 1e0,
                          kdis: np.floating = 1e3,
                          eta_c: np.floating = 4.0,
                          eta_d: np.floating = 1.5):
        """
        Sets the power spectrum parameters for:
        ps = p0 * k**slope * np.exp(-(kc/k)**eta_c) * np.exp(-(k/kdis)**eta_d)

        Parameters
        ----------
        slope : np.floating
            The slope of the power spectrum. [Sign convention: P = P0 * k**slope].
            The default value is 0.0
        kc : np.floating
            The "cutoff" wavenumber. Default value is 1e-3.
        p0 : np.floating
            The normalization of the power spectrum [Sign convention: P = P0 * k**slope]
        kdis : np.floating
            The dissipation wavenumber.
        eta_c : np.floating
            Cutoff rate. Defaults to 4.0, used in Khatri & Gaspari (2016)
        eta_d : np.floating
            Dissipation rate. Defaults to 1.5, used in Khatri & Gaspari (2016)
        """
        self.slope=slope   # Power-law slope (k**slope)
        self.kc = kc       # Cutoff scale
        self.p0 = p0       # Normalization
        self.kdis = kdis   # Disipation scale
        self.eta_c = eta_c # Cutoff exponent
        self.eta_d = eta_d # Dissipation exponent
        
    def set_minfactor(self,minfactor: np.floating):
        """
        Sets a factor that will define the smallest k-value in an array. 

        Parameters
        ----------
        minfactor : np.floating
            What factor (1/minfactor) of the smallest non-zero k will we adopt as a new smallest k?
        """

        ### This is not actually used. I will slate this for deprecation.
        

        self._minfactor = minfactor # If min(k)==0, replace with smallest non-zero k divided by this factor
        
    def set_image_size(self,n_pix,pixsize):
        """
        Sets class attributes related to image and pixel size.

        Parameters
        ----------
        n_pix : int
            The number of pixels on a side. Images are restricted to squares (same pixels on each side).
        pixsize: Union[ np.floating, int ]
            Pixels are also taken to be squares (same length on each side). The value should correspond
            to the length in the units that match the (inverted) units of parameters associated with
            wavenumbers (kc, k_dis).
        """

        self.n_pix = n_pix
        self.pixsize = pixsize        
        k,dkx,dky = get_freqarr_2d(n_pix, n_pix, pixsize, pixsize)
        kflat = k.flatten()
        gki = (kflat > 0)
        gk = kflat[gki]
        self._kflat = kflat
        self.k = gk
        self._gki = gki

    def get_parameterized_ps(self,k:NDArray[np.floating]):
        """
        Computes an array of power spectrum values at wavenumbers k assuming the parameterization
        stored within this class.

        Parameters
        ----------
        k : NDArray[np.floating]
            The output array of (logarithmically spaced) wavenumbers

        Returns
        -------
        ps : NDArray[np.floating]
            The corresponding power spectrum.
        """

        keqz = (k == 0)
        kgtz = (k > 0)
        kmin = np.min(k[kgtz])
        if np.sum(keqz) > 0:
            k[keqz] = kmin/self.kfactor # Avoid k=0, but get a really small value.
        ps = self.p0*k**(self.slope) * np.exp(-(self.kc/k)**self.eta_c) * np.exp(-(k/self.kdis)**self.eta_d)

        return np.nan_to_num(ps)

    def get_logspaced_k(self,
                        kmin: Optional[np.floating]=None,
                        kmax: Optional[np.floating]=None,
                        nPts: int = 500
                        ) -> NDArray[np.floating]:
        """
        Generate an array of logarithmically spaced wavenumbers, k

        Parameters
        ----------
        kmin : Optional[np.floating]
            Minimum wavenumber
        kmax : Optional[np.floating]
            Maximum wavenumber
        nPts: int
            Number of points (elements) in the wavenumber array to generate

        Returns
        -------
        k : NDArray[np.floating]
            The output array of (logarithmically spaced) wavenumbers
        """

        mykmin = 1.0/(self.n_pix * self.pixsize) if kmin is None else kmin
        mykmax = 1.0/(self.pixsize) if kmax is None else kmax
        k = np.logspace(np.log10(mykmin),np.log10(mykmax),nPts)

        return k

    def set_seed(self,seed):
        """
        Set the seed for the random number generator.

        Parameters
        ----------
        seed : Optional[int]
            A seed for the random number generator.
        """

        self.rng = np.random.default_rng(seed=seed)
    
    def generate_realization(self,
                             k_in: Optional[NDArray[np.floating]] = None,
                             ps_in: Optional[NDArray[np.floating]] = None,
                             seed: Optional[int] = None
                             ) -> NDArray[np.floating]:
        """
        Generate an image (realization) based on either:
        1) The current power spectrum parameters
                     OR
        2) The input power spectrum, in which case both k_in and ps_in must be specified
           (and match in dimensionality).

        The generated image will adopt the properties (n_pix and pixsize) as input for class
        in which this resides.

        Parameters
        ----------
        k_in : Optional[NDArray[np.floating]]
            1D numpy array of input wavenumbers (should match units of inverse pixsize)
        ps_in : Optional[NDArray[np.floating]]
            The corresponding (input) power spectrum.
        seed : Optional[int]
            A seed for the random number generator.

        Returns
        -------
        img : NDArray[np.floating]
            The output image, matched to n_pix and pixsize.
        """
        psarr = self._kflat*0
        if k_in is None or ps_in is None:
            psout = self.get_parameterized_ps(self.k)
        else:
            psout = np.exp(np.interp(np.log(self.gk),np.log(k_in),np.log(ps_in)))
        psarr[self.gki] = psout
        ps2d = psarr.reshape(self.n_pix,self.n_pix) * self.n_pix * self.n_pix

        if seed is not None:
            self.set_seed(seed)
        phase = self.rng.uniform(size=(self.n_pix,self.n_pix))*2*np.pi
        newfft = np.sqrt(ps2d) * np.exp(1j*phase)
        img = np.real(np.fft.ifft2(newfft/self.pixsize))
        img *= np.sqrt(2.0)

        return img

#################################################################################
    
def get_freqarr_2d(nx: int, ny: int,
                   psx: np.floating, psy: np.floating
                   ) -> Tuple[NDArray[np.floating], np.floating, np.floating]:
    """
    Compute frequency array for 2D FFT transform

    Parameters
    ----------
    nx : int
        number of samples in the x direction
    ny : int
        number of samples in the y direction
    psx : np.floating
        map pixel size in the x direction
    psy : np.floating
        map pixel size in the y direction

    Returns
    -------
    k : float 2D numpy array
        frequency vector
    """
    kx =  np.outer(np.fft.fftfreq(nx),np.zeros(ny).T+1.0)/psx
    ky =  np.outer(np.zeros(nx).T+1.0,np.fft.fftfreq(ny))/psy
    dkx = kx[1:][0]-kx[0:-1][0]
    dky = ky[0][1:]-ky[0][0:-1]
    k  =  np.sqrt(kx*kx + ky*ky)
    return k, dkx[0], dky[0]

class MultiGaussBeam:

    def __init__(self,norms,widths):
        """
        Define a beam (or point-spread function, PSF) as multiple Gaussians via a list of
        normalizations (height) and widths (Gaussian sigmas).

        Parameters
        ----------
        norms : ArrayLike[np.floating]
            array-like collection of Gaussian normalizations.
            Internally, the sum of norms will be normalized to equal unity
        widths : ArrayLike[np.floating]
            array-like collection of Gaussian standard deviations.
        """
        self.norms = np.array(norms,dtype=float)
        self.norms /= np.sum(self.norms) # Impose unitary normalization
        self.widths = np.array(widths,dtype=float)
        self.nGauss = len(widths)

    def calc_ft_at_k(self,k : NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute the Fourier transform for a multi-Gaussian (2D) Point Spread Function
        
        Parameters
        ----------
        k : NDArray[np.floating]
            An array of wavenumbers (e.g. inverse arcseconds). The units of karr must be
            the inverse of the units used for the sigma(s) describing the N-Gaussian (below).
        """
        
        g = np.zeros(k.shape)    
        for n,s in zip(self.norms,self.widths):
            t = 2 * np.pi**2 * s**2
            g += n * np.exp(-k**2 * t)
        
        return g

    def calc_ps_at_k(self,k : NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Compute the power spectrum for a multi-Gaussian Point Spread Function
        
        Parameters
        ----------
        k : NDArray[np.floating]
            An array of wavenumbers (e.g. inverse arcseconds). The units of karr must be
            the inverse of the units used for the sigma(s) describing the N-Gaussian (below).
        """
        psf_ft = self.calc_ft_at_k(k)

        return psf_ft**2
    
    def get_multiGauss_terms(self,
                             karr: NDArray[np.floating],
                             alpha: float ) -> NDArray[np.floating]:
        """

        This corrects for the bias noted in Romero+ 2023, Romero 2024.
        Code assumes nGauss are few (such that for-loops are cheap); could be optimized if nGauss is
        huge (but you've probably done something wrong to have nGauss that large).

        Parameters
        ----------
        karr : NDArray[np.floating]
            A one-dimensional array of wavenumbers (e.g. inverse arcseconds). The units of karr must be
            the inverse of the units used for the sigma(s) describing the N-Gaussian (below).
        alpha : float 
             The spectral index assumed. [Sign convention: P(k) = P0 * k **(-alpha)]

        Returns:
        --------
        corrs : NDArray[np.floating]
             An array of correction factors corresponding to input karr wavenumbers.
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

    def get_multiGauss_bias(self,
                            karr: NDArray[np.floating],
                            alpha: float ,
                            ignpsf: bool = False,
                            pbonly: bool = False):
        """
        Corrects for:
        (1) The scalar bias induced by Arevalo+ 2012
        (2) The non-scalar bias induced by a beam *and* use of Arevalo+ 2012
        --- In order for this to be done, some spectral index, alpha MUST be assumed! ---
        --- where P(k) = P0 * k**-alpha                                              ---
        Note that there is a correction for the power spectrum of the beam (PSF) within THIS term.
        If you want to correct for the PSF and the bias then these terms cancel! In light of this,
        I've added a keyword to allow you to ignore the PS term.
        ---------------------------------------------------------------------------------
        karr : NDArray[np.floating]
            The array of k (wavenumber) values at which to calculate the total bias
        alpha : float 
            The assumed spectral index (convention given above)
        ignpsf : bool
            As mentioned above, allows you to ignore the PSF power spectrum term with respect
            to this bias. Given that you want to correct for the PSF power spectrum, they will
            cancel each other out when correcting your measured power spectrum. You can bypass
            the additional calculations by ignoring the PSF term here.
        pbonly : bool
            PSF BIAS only. That is, ignore the scalar bias due to an underlying power-law.
        """
        ndim = 2
        lilg = (ndim/2 + 2 - alpha/2)
    
        p1 = 2**(alpha/2)
        p2 = self.get_multiGauss_terms(karr,alpha)
        p3 = sps.gamma(lilg) / sps.gamma(ndim/2 + 2)
        psf_ps = 1.0 if ignpsf else self.calc_ps_at_k(karr)

        bias  = p2/psf_ps if pbonly else p1*p2*p3 / psf_ps

        return bias

