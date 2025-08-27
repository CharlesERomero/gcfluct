import pytest
import astropy.units as u
import numpy as np
import matplotlib.pyplot as plt

import gcfluct.spectra.spectra2d as spectra
import gcfluct.utils.image_filtering as imf
import gcfluct.utils.mock_obs as mo
import gcfluct.gc.selfsimilar_gc as ssgc
import gcfluct.spectra.gc_spec_deproj as deproj


def test_SSGC():

    z = 1.0
    M500 = 1e14 * u.M_sun
    myssgc = ssgc.Cluster(z,M500)

def test_nb_coverage():

    nPix = 1024                  # Number of pixels on a side (square image)
    pixsize = 1.0                # Assume this is in kpc
    units = "kpc"                # Let's have a string that goes with this.
    kc = 3e-5                    # Assume this is in [pixsize]^(-1) -- so kpc^-1 in our case.
    slope = -3.667               # Power goes as k**slope
    p0 = 9e-4                    # Arbitrary units at the moment
    fwhm = 5.0                   # kpc
    s2f = 2*np.sqrt(2*np.log(2)) # Gaussian sigma to FWHM (if needed)
    pixfwhm = fwhm/pixsize       # FWHM in pixels

    PSobj = spectra.ImagesFromPS(nPix,pixsize=pixsize,slope=slope,kc=kc,p0=p0)
    ks = PSobj.get_logspaced_k()
    ps = PSobj.get_parameterized_ps(ks)

    raw_image = PSobj.generate_realization()
    image = imf.fourier_filtering_2d(raw_image,"gauss",pixfwhm)     # in Compton y
    
    intrinsic_mask = np.ones(image.shape) # Suppose there are bad pixels; those could be omitted with zeros in this array.
    img2PS = spectra.PSfromImages(image,pixsize=pixsize,intrinsic_mask=intrinsic_mask)

    img2PS.ps_via_a12_2d()
    
    sigma   = fwhm/(2.0*np.sqrt(2.0*np.log(2)))

    normalizations = [1] # The iterable normalizations (height) of the Gaussian
    sigmas = [sigma] # The iterable Gaussian sigmas (width)
    PSF = spectra.MultiGaussBeam(normalizations,sigmas)
    A12biasPSF = PSF.get_multi_gauss_bias(img2PS.a12_kn,-PSobj.slope,ign_psf=True)

def test_mock_nb():

    z = 0.4
    m500 = 5e14 * u.M_sun
    gc_model = ssgc.SS_Model(z,m500)
    kpc_per_arcsecond = gc_model.d_ang.to("kpc").value / (3600 * 180 / np.pi)
    gc_model.set_xr_USBP()

    kc = 3e-3         # Assume this is in [pixsize]^(-1) -- so kpc^-1 in our case.
    slope = -3        # Power goes as k**slope
    p0 = 3e-2         # Arbitrary units at the moment
    no_warn = True    # Ignore some warnings (in regards to pixels for image creation) for now.
    ps2img = spectra.ImagesFromPS(slope=slope,kc=kc,p0=p0,no_warn=no_warn)

    pix_arcsec = 2.5 # 
    pixsize = pix_arcsec*kpc_per_arcsecond    # in kpc
    # We set no_warn earlier because the class strongly prefers units to be set.
    # It's still "optional" insofar as there are use cases where you don't need to set
    # the units
    pixunits = u.kpc 
    n_r500 = 2
    npix = int(np.round(gc_model.arcminutes500*2*n_r500*60/pix_arcsec))   
    ps2img.set_image_size(npix,pixsize,pixunits=pixunits)
    # Same number as in x -- keep things simple.
    image = ps2img.generate_realization()
    gc_model.set_ss_maps(imagefromps=ps2img) # Let gc_model know the grid used for producing maps.

    image_is_sz = False # We are modeling (simulating) an X-ray image, not SZ here.
    gc_deproj = deproj.SpecDeproj(gc_model,spec_model=ps2img,sz = image_is_sz) 
    n_map = gc_deproj.return_integ_nmap(use_obs_spec=False) # Gives factors to translate between P_2D and P_3D

    ds_s = np.sqrt(n_map) * image     # Normalized residual.
    ds = ds_s * gc_model.xrsb_map      # Residual image in cnts/s/arcmin**2
    s_cluster = ds + gc_model.xrsb_map # Image in cnts/s/arcmin**2

    ksec = 50 # kiloseconds
    exposure_map = mo.mock_XMM_exposure(ps2img,ksec,rot=30,incChipGaps=True) # in seconds

    s_particle_bkg = 1e-3                  # This is a typical uniform, quiescent particle X-ray background. Soft protons and SWCX usually are sub-dominant.
    s_cosmic_bkg = 1e-3                    # The cosmic X-ray background
    s_bkg = s_particle_bkg + s_cosmic_bkg  # Total background rate

    s_tot = s_cluster + s_bkg  # Assume that we can account for all photons in this manner
    Counts_Total = s_tot * exposure_map * (pix_arcsec/60)**2

    ### Don't need to generate Poisson realization; we've tested the lines of code we wanted to.
