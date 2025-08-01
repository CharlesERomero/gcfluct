import numpy as np
from astropy.io import fits
from scipy.special import gamma
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union
from astropy.units import Quantity
import warnings

import gcfluct.gc.selfsimilar_gc as ssgc
from gcfluct.gc.selfsimilar_gc import SS_Model
from gcfluct.spectra.spectra2d import ImagesFromPS, PSfromImages

class SpecDeproj:
    """

    Attributes
    ----------
    gc_model : SS_Model
        A class which models a galaxy cluster, especially under an assumption of self-similarity.
    z_step : np.floating
        Step size long the line of sight, in kpc.
    z_extent : np.floating
        Maximum depth along the line of sight: |z| < z_extent with onject center located at z=0, in kpc.
    z_array : NDArray[np.floating]
        Array of z-values (along the line of sight), in kpc.
    kz : NDArray[np.floating]
        Array of wavenumber values along the line of sight (z-axis). Units are inverse kpc.
    pos_kz : NDArray[np.floating]
        Array of positive wavenumber values along the line of sight (z-axis). Units are inverse kpc.
    dkz : np.floating
        Step size of wavenumbers long the line of sight. Units are inverse kpc.
    sz_windows : NDArray[np.floating]
        2D array of the window function pertaining to the SZ model with the gc_model attribute.
        The shape is (N_sky, N_los), where N_sky is the number of radial points in the plane of the
        sky and N_los is the number of points along the line of sight. Units are inverse kpc.
    sz_tilde_pow : NDArray[np.floating]
        2D array of the square of the Fourier transform of the window function pertaining to the
        SZ model with the gc_model attribute.
        The shape is (N_sky, N_los), where N_sky is the number of radial points in the plane of the
        sky and N_los is the number of points along the line of sight. Unitless.
    sz_integ_ns : NDArray[np.floating]
        1D array of sz_tilde_pow integrated along k_z. For a given line of sight P_2D = N * P_3D, where the
        equivalence is an approximation that holds best at high k (smaller scales.) Units are inverse kpc.
    xr_windows : NDArray[np.floating]
        2D array of the window function pertaining to the X-ray model with the gc_model attribute.
        The shape is (N_sky, N_los), where N_sky is the number of radial points in the plane of the
        sky and N_los is the number of points along the line of sight. Units are inverse kpc.
    xr_tilde_pow : NDArray[np.floating]
        2D array of the square of the Fourier transform of the window function pertaining to the
        X-ray model with the gc_model attribute.
        The shape is (N_sky, N_los), where N_sky is the number of radial points in the plane of the
        sky and N_los is the number of points along the line of sight. Unitless.
    xr_integ_ns : NDArray[np.floating]
        1D array of xr_tilde_pow integrated along k_z. For a given line of sight P_2D = N * P_3D, where the
        equivalence is an approximation that holds best at high k (smaller scales.) Units are inverse kpc.
    spec_model: Optional[PSfromImages]
        A class relevant to modeling power spectra and generating (image) realizations.
    spec_observed: Optional[ImagesFromPS]
        A class relevant to an image and measuring power spectra on the image.

    window : NDArray[np.floating]
        A single window (along a single LOS, either SZ or X-ray) as selected by the user.
    tilde_pow : NDArray[np.floating]
        The square of the Fourier transform of the selected window.
    integ_n : np.floating
        The integrated value of tilde_pow over k_z.


    Methods
    -------
    set_spec_model()
        Sets the attribute spec_model
    set_spec_observed()
        Sets the attribute spec_observed and should indicate whether its contains SZ or X-ray data.
    set_sz_windows()
        Sets the SZ windows based on gc_model.
    set_xr_windows()
        Sets the X-ray windows based on gc_model.
    set_windows_manual()
        Sets either SZ or X-ray windows based on user input thermodynamic and srface brightness profiles.
    select_window()
        Selects either SZ or X-ray windows based on user input.
    return_integ_nmap()
        Returns a map of integ_n values based on the windows and spec_observed attribute. (Recall the method to set the
        spec_observed attribute also wants an indication whether the corresponding data refers to SZ or X-ray data.)
    theory_projection()
        If a P3D model exists (as it would with the spec_model attribute), then the full integration:
        P2D = int (P3D * |tilde(W)|^2) dkz can be computed, and this method does this.
    """

    def __init__(self,
                 gc_model,
                 extent500 = 10.0,
                 step500 = 1e-2,
                 spec_model = None,
                 spec_observed = None):
        """
        Initiates the class. If attributes should have units, they will be in kpc (or inverse kpc).

        Parameters
        ----------
        gc_model : SS_Model
            A class which models a galaxy cluster, especially under an assumption of self-similarity.
        extent500 : np.floating
            Along the line of sight, make calculations with |z| < extent500 * R500
        step500 : np.floating
            Along the line of sight, the step size relative to R500.
        spec_model : Optional[ImagesFromPS]
            A class which models a parameterized power spectrum.
        spec_observed : Optional[PSfromImages]
            A class with an image and inferred (calculated/measured) spectra.
        """
        
        # I will work in kpc (rather than angular units). We'll transform from that if need be.
        if gc_model is SS_Model:
            self.gc_model = gc_model
        else:
            raise TypeError("gc_model must be of the class SS_Model.")
        
        r500_kpc = gc_model.r500.to("kpc").value
        self.z_step = step500 * r500_kpc
        self.z_extent = r500_kpc * extent500

        self.z_array = np.arange(-self.z_extent, self.z_extent, self.z_step)  # in kpc
        self.kz = np.fft.fftfreq(len(z_array),self.z_step) 
        self._pos_inds = self.kz > 0         
        self.pos_kz = self.kz[self._pos_inds] #
        self.dkz = mykz[1]-mykz[0] #

        self.set_sz_windows()
        self.set_xr_windows()
        
        self.set_spec_model(spec_model)
        self.set_spec_observed(spec_observed)
        self._deselect_window()

    def set_spec_model(self,spec_model):
        """
        Sets the attribute spec_model, which must be of the class ImagesFromPS. 

        Parameters
        ----------
        spec_model : ImagesFromPS
            A class relevant to modeling power spectra and generating (image) realizations.
        """
        
        if spec_model is None or spec_model is ImagesFromPS:
            self.spec_model = spec_model
        else:
            raise TypeError("spec_model must either be None or of the class ImagesFromPS.")

    def set_spec_observed(self,
                          spec_observed: PSfromImages,
                          sz: Optional[bool] = None):
        """
        Sets the attribute spec_observed, which should be of the class PSfromImages.

        Parameters
        ----------
        spec_observed : PSfromImages
            A class relevant to an image and measuring power spectra on the image.
        sz : Optional[bool]
            Preferably this is set; this provides for better bookkeeping. It is required for other methods
            such as return_integ_nmap().
        """
        
        if spec_observed is None or spec_observed is PSfromImages:
            self.spec_observed = spec_observed
        else:
            raise TypeError("spec_observed must either be None or of the class PSfromImages.")

        if sz is None:
            warnings.warn("Please indicate whether this attribute corresponds to SZ (True) or X-ray (False)")
            self.spec_obs_sz = None
        else:
            self.spec_obs_sz = sz

    
    def _get_mats(self) -> Tuple[NDArray[np.floating],NDArray[np.floating],NDArray[np.floating]] :
        """
        A little helper method to get 2D arrays of radii for array-based computation.

        Returns
        -------
        expr : Tuple[NDArray[np.floating],NDArray[np.floating],NDArray[np.floating]]
           rad_mat is the 2-dimensional array of the 3D radii
           sky_mat is the 2-dimensional array of the plane-of-sky distance (radii).
           los_mat is the 2-dimensional array of the distance along the line of sight.
        """

        theta_sky = self.gc_model.rads.to("kpc") # Keep as quantity
        z_los = self.z_array * u.kpc
        
        sky_mat = np.repeat([theta_sky], z_los.size, axis=0)
        los_mat = np.repeat([z_los], theta_sky.size, axis=0).transpose()

        rad_mat = np.sqrt(sky_mat**2 + los_mat**2) # 3D radii, as a 2D matrix

        return rad_mat,sky_mat,los_mat
        
    def set_sz_windows(self):
        """
        Based on the attribute gc_model (and its model of the pressure and Compton y),
        computes the SZ windows as a 2D array with first axis corresponding to radii in
        the plane of the sky and the second axis corresponding to depth along the line
        of sight (attribute z_array).

        Sets attributes sz_windows, sz_tilde_pow, and sz_integ_ns. sz_windows and sz_integ_ns
        have units of inverse kpc. sz_tilde_pow is unitless.
        """
        
        rad_mat, sky_mat, los_mat = self._get_rad_mat()        
        y_mat = np.repeat([self.gc_model.y_prof], self.z_array.size, axis=0)
        pres_mat = self.gc_model.gnfw(rad_mat) # Pressure matrix

        self.sz_windows = (pres_mat*self.gc_model._p2invkpc).decompose().value / y_mat

        # axis=-1, is numpy's default behavior. But for explicity:
        tilde_mat = np.fft.fft(self.sz_windows,axis=1)*self.z_step
        self.sz_tilde_pow = np.abs(tilde_mat**2)
        self.sz_integ_ns = np.sum(self.tilde_pow,axis=1) * self.dkz
        
    def set_xr_windows(self):
        """
        Based on the attribute gc_model (and its model of the emissivity and surface brightness),
        computes the X-ray windows as a 2D array with first axis corresponding to radii in
        the plane of the sky and the second axis corresponding to depth along the line
        of sight (attribute z_array).

        Sets attributes xr_windows, xr_tilde_pow, and xr_integ_ns. xr_windows and xr_integ_ns
        have units of inverse kpc. xr_tilde_pow is unitless.
        """

        rad_mat, sky_mat, los_mat = self._get_rad_mat()
        theta_kpc = self.gc_model.xr_theta_c.to("kpc")
        r3d_scaled = (rad_mat/theta_kpc).decompose().value
        r2d_scaled = (sky_mat/theta_kpc).decompose().value
        emmisivity = (1.0 + (r3d_scaled)**2)**(-3.0*self.gc_model.xr_beta)          # 
        surface_b = (1.0 + (r2d_scaled)**2)**(0.5 -3.0*self.gc_model.xr_beta)      # Recalculating...kind of
        betanorm = gamma(3*beta-0.5)*gamma(0.5) / gamma(3*beta) 

        self.xr_windows = emmisivity / (betanorm*beta*theta_kpc.value*surface_b)   # inverse kpc
        
        # axis=-1, is numpy's default behavior. But for explicity:
        tilde_mat = np.fft.fft(self.xr_windows,axis=1)*self.z_step
        self.xr_tilde_pow = np.abs(tilde_mat**2)
        self.xr_integ_ns = np.sum(self.tilde_pow,axis=1) * self.dkz

    def set_windows_manual(self,rad_in,thermo_prof,sb_prof,sz=True):
        """
        Supposing the user has a thermodynamic (3D quantity) profile, a corresponding surface brightness profle,
        and an array of corresponding radii (that matches both the 2D and 3D profiles), we can calculate the windows.
        (NB, 1 radial profile is used; it is suggested that this profile goes far out, e.g. 10*R500; lack of values)
        """
        
        rad_mat, sky_mat, los_mat = self._get_rad_mat()
        
        rad1d = rad_mat.flatten()
        
        # Logarithmic interpolation (linear interpolation in log-space).
        # This has potential for adverse results at 3D radii beyond the input profile radii. With the defaults, this
        # will be negligible for foreseeable values (< 10 R_500). In principle the user could change this and cause problems.
        #####################################################################################################################
        thermo_mat = np.interp(np.log(rad1d),np.log(rad_in.ravel()),np.log(thermo_prof.ravel()),right=-np.inf)
        thermo_mat = np.exp(thermo_mat.reshape(rad_mat.shape)) * u.keV / u.cm**3 # Add units in.
        del rad1d # cleanup

        sb_mat = np.repeat([sb_prof], self.z_array.size, axis=0)

        windows = thermo_mat / sb_mat
        # axis=-1, is numpy's default behavior. But for explicity:
        tilde_mat = np.fft.fft(windows,axis=1)*self.z_step

        if sz:
            self.sz_windows = windows
            self.sz_tilde_pow = np.abs(tilde_mat**2)
            self.sz_integ_ns = np.sum(self.tilde_pow,axis=1) * self.dkz
        else:
            self.xr_windows = windows
            self.xr_tilde_pow = np.abs(tilde_mat**2)
            self.xr_integ_ns = np.sum(self.tilde_pow,axis=1) * self.dkz
            
    def _deselect_window(self):
        """
        Mostly used as an initiation method such that attributes exist, even if None.
        """
        
        self.window = None
        self.tilde_pow = None
        self.integ_n = None
        self._radius = None
        self._sz = None
        
    def select_window(self, radius, sz: bool = True):
        """
        Selects a window, W, corresponding square of its Fourier transform (\tilde{W}^2), and corresponding
        integrated value, N based on the radius (in the plane of the sky) at which one is interested. Does this
        for either SZ or X-ray, as indicated by input arguments. Sets the attributes window, tilde_pow, and integ_n.

        Parameters
        ----------
        radius : np.floating
            A scalar (not an iterable); the radius (in kpc, as a value) at which the window and corresponding quantities
            are desired.
        sz : bool
            If true, sets attributes for SZ (relating P_pressure to P_Compton-y)? If false, then attributes will
            correspond to the scaling between P_emissivity and P_XRSB.
        """
        
        mydiff    = np.abs(rad_kpc - myradius)
        mynpind   = np.where(mydiff == np.min(mydiff))            # Find the closest point to 500 kpc
        myind     = mynpind[0][0]
        self.window = self.sz_windows[:,myind] if sz else self.xr_windows[:,myind]
        self.tilde_pow = self.sz_tilde_pow[:,myind] if sz else self.xr_tilde_pow[:,myind]
        self.integ_n = self.sz_integ_ns[myind] if sz else self.xr_integ_ns[myind]
        
        # Some bookkeeping. In case the user forgot what was set here.
        self._radius = radius
        self._sz = sz 

    def return_integ_nmap(self) -> NDArray[np.floating]:
        """
        Calculates a grid (map) of N = int( |W^2| dkz ) values that can be used to rescale an image such
        that one directly computes a deprojected power spectrum. This requires that the attribute
        spec_observed is set (i.e. it uses a reference image).

        Returns
        -------
        nmap : NDArray[np.floating]
            A map of N values (based on
        """
        
        rmap1d = self.spec_observed.rmat.flatten() # Radii in map
        rns = self.gc_model.rads.to("kpc") # Radii associated with integ_n's (not Registered Nurses)
        is_length = self.spec_observed.pixunits.is_equivalent(u.kpc)
        if self.spec_obs_sz is None:
            raise TypeError("Well, you were warned to set this. Now it's a problem.")
        else:
            integ_ns = self.sz_integ_ns if self.spec_obs_sz else self.xr_integ_ns # Has units of inverse kpc
        
        if is_length:
            rn_vals = rns.to(self.gc_model.pixunits).value # A value, in the units of the map.
            n1d = np.interp(r1d, rn_vals, integ_ns,right=0)
        else:
            rn_arcsec = rns.value * self.gc_model.kpcperas * u.arcsec # In arcseconds
            rn_vals = rns.to(self.gc_model.pixunits).value # A value, in the units of the map.
            n1d = np.interp(r1d, rn_vals, integ_ns*self.gc_model.kpcperas,right=0)
        
        nmap = n1d.reshape(self.spec_observed._imsz)        
        return nmap

    ##########################################################################################
        
    def theory_projection(self,
                          k_out: Optional[NDArray[np.floating]] = None
                          ) -> NDArray[np.floating]:
        """
        Performs the full power spectrum projection (integration) from the 3D power spectrum to its
        2D counterpart. This method requires that the attribute spec_model is set.

        Parameters
        ----------
        k_out : Optional[NDArray[np.floating]]
            An 1D array of output wavenumbers at which the projected spectrum is computed. If None, uses
            the wavenumbers at which an observed spectrum is computed via the Arevalo (2012) method for
            an input image. The latter case requires that the attribute spec_observed is set.

        Returns
        -------
        p2d : NDArray[np.floating]
            The projected (P2D) power spectrum at wavenumbers k_out if set, otherwise at wavenumbers at
            spec_model.a12_kn.
        """

        if k_out is None:
            
            if self.spec_observed is PSfromImages:
                k_theta = self.spec_observed.a12_kn
            else:
                raise TypeError("Incorrect or no k_out supplied")
        else:
            k_theta = k_out

        nkz = len(self.kz)
        nkt = len(k_theta)
        kt2d = np.outer(k_theta,np.ones(nkz))
        kz2d = np.outer(np.ones(nkt),self.kz)
        k3d = np.sqrt(kt2d**2 + kz2d**2)
        p3d = self.spec_model.get_parameterized_ps(k3d) # 3D power spectrum
        w3d = np.outer(np.ones(nkt),self.tilde_pow) # |\tilde{W}^2| (square of Fourier transform of window)
    
        p2d = np.sum(p3d * w3d, axis=1) * self.dkz # Projected, "exact", power spectrum

        return p2d
    
    def _taper_window_z(self,
                        zoff: np.floating,
                        zon: np.floating = 0,
                        width: np.floating = 2,
                        gentle: bool = True,
                        use_tukey: bool = True):
        """
        Sets attributes _tapered_window and _tapered_tilde_pow based on the input taper depth(s) and the
        current selected window. This method is considered "private" insofar as its usage is considered
        advanced. One has to be very thoughtful and careful about what one is intending to do here as there
        is no accepted practice in the literature. Direct contact with the author(s) of this code may be
        the most productive path.

        Parameters
        ----------
        zoff : np.floating
           The depth (along the z-axis) at which the gas is no longer characterized by a given
           power spectrum. That is, when to taper off the window.
        zon : np.floating
           If zon > 0, then for |z| < zon, it is taken that the gas is not characterized by a
           given power spectrum and the window will taper on at zon. Default is 0, i.e. no "turn-on" depth.
        width : np.floating
           Roughly half the number of array elements (indices traversed) to complete a taper (from 1 to 0 or
           vice versa). Default is 2.
        gentle : bool
           Make use of the width argument. Otherwise a sharp transition from 1 to 0. (If False, this produces
           ringing in the Fourier transform, as expected, and thus leaving as True is recommended.)
        use_tukey : bool
           Use the Tukey taper, rather than a Gaussian taper. Default is True
        """
        
        
        padding = width*self.z_step * np.sqrt(np.log(2)) if gentle else 0
        if zon > 0:
            zdiff1 = zon - np.abs(self.z_array) + padding
            phase = np.pi*zdiff1/(2*width*self.z_step)
            tukey1 = 0.5* (1 + np.sin(phase))
            gauss_t1 = np.exp(-zdiff1**2 / (2*(width*self.z_step)**2))
            bi = (np.abs(self.z_array) > zon+padding)
            gauss_t1[bi] = 1.0
            tzero = (phase < -np.pi/2.0)
            tone = (phase > np.pi/2.0)
            tukey1[tzero] = 0.0
            tukey1[tone] = 1.0
        else:
            gauss_t1 = np.ones(self.z_array.shape)
            tukey1 = np.ones(self.z_array.shape)

        zdiff2 = np.abs(self.z_array) - zoff - padding
        phase2 = np.pi*zdiff2/(2*width*self.z_step)
        tukey2 = 0.5* (1 - np.sin(phase2))
        tzero2 = (phase2 > -np.pi/2.0)
        tone2 = (phase2 < np.pi/2.0)
        tukey2[tzero2] = 0.0
        tukey2[tone2] = 1.0
        gauss_t2 = np.exp(-zdiff2**2 / (2*width*self.z_step))
        gi = (np.abs(self.z_array) < zoff - padding)
        gauss_t2[gi] = 1.0
        
        gauss_t = gauss_t1*gauss_t2   
        tukey = tukey1*tukey2    

        self._tapered_window = tukey * self.window if use_tukey else gauss_t * self.window
        tapered_tilde = np.fft.fft(self.tapered_window)*self.z_step
        self._tapered_tilde_pow = np.abs(tapered_tilde**2)
