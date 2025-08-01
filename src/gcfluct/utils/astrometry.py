import numpy as np
from astropy.io import fits
import astropy.units as u


def astro_from_hdr(hdr):
    
    xsz = hdr['naxis1']
    ysz = hdr['naxis2']
    xar = np.outer(np.arange(xsz),np.zeros(ysz)+1.0)
    yar = np.outer(np.zeros(xsz)+1.0,np.arange(ysz))
    ####################

    w = wcs.WCS(hdr)
    #import pdb;pdb.set_trace()
    
    xcen = hdr['CRPIX1']
    ycen = hdr['CRPIX2']
    dxa = xar - xcen
    dya = yar - ycen
    ### RA and DEC in degrees:
    if 'CD1_1' in hdr.keys():
        ras = dxa*hdr['CD1_1'] + dya*hdr['CD2_1'] + hdr['CRVAL1']
        decs= dxa*hdr['CD1_2'] + dya*hdr['CD2_2'] + hdr['CRVAL2']
        pixs= abs(hdr['CD1_1'] * hdr['CD2_2'])**0.5 * 3600.0
    if 'PC1_1' in hdr.keys():
        pcmat = w.wcs.get_pc()
        ras = dxa*pcmat[0,0]*hdr['CDELT1'] + \
              dya*pcmat[1,0]*hdr['CDELT2'] + hdr['CRVAL1']
        decs= dxa*pcmat[0,1]*hdr['CDELT1'] + \
              dya*pcmat[1,1]*hdr['CDELT2'] + hdr['CRVAL2']
        pixs= abs(pcmat[0,0]*hdr['CDELT1'] * \
                  pcmat[1,1]*hdr['CDELT2'])**0.5 * 3600.0

    pixs = pixs*u.arcsec
    ### Do I want to make ras and decs Angle objects??
    ras  = ras*u.deg; decs = decs*u.deg 
    
    return ras, decs, pixs

def get_astro(file):

    hdu = fits.open(file)
    hdr = hdu[0].header
    image_data = hdu[0].data

    ras, decs, pixs = astro_from_hdr(hdr)

    return image_data, ras, decs, hdr, pixs

class astrometry:

    def __init__(self,hdr):

        ras,decs,pixs = astro_from_hdr(hdr)
        self.ras = ras
        self.decs= decs
        self.pixs= pixs
        
def make_template_hdul(nx,ny,cntr,pixsize,cx=None,cy=None):
    """
    Return a map of RMS sensitivites based on input set of scans.

    Parameters
    ----------
    nx : int
       Number of pixels along axis 0
    ny : int
       Number of pixels along axis 1
    cntr : array_like
       Two-element object specifying the RA and Dec of the center.
    pixsize : float
       Pixel size, in arcseconds
    cx : float
       The pixel center along axis 0
    cy : float
       The pixel center along axis 1

    Returns
    -------
    TempHDU : class:`astropy.io.fits.HDUList`
       A Header-Data-Unit list (only one HDU)

    """

    if cx is None:
        cx = nx/2.0
    if cy is None:
        cy = ny/2.0
    ### Let's make some WCS information as if we made 1 arcminute pixels about Coma's center:
    w = WCS(naxis=2)
    w.wcs.crpix = [cx,cy]
    w.wcs.cdelt = np.array([-pixsize/3600.0,pixsize/3600.0])
    w.wcs.crval = [cntr[0], cntr[1]]
    #w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    hdr = w.to_header()

    zero_img    = np.zeros((nx,ny))
    Phdu        = fits.PrimaryHDU(zero_img,header=hdr)
    TempHdu     = fits.HDUList([Phdu])

    return TempHdu

def get_xymap(map,pixsize,xcentre=[],ycentre=[],oned=True,cpix=0):
    """
    Returns a map of X and Y offsets (from the center) in arcseconds.

    INPUTS:
    -------
    map      - a 2D array for which you want to construct the xymap
    pixsize  - a quantity (with units of an angle)
    xcentre  - The number of the pixel that marks the X-centre of the map
    ycentre  - The number of the pixel that marks the Y-centre of the map

    """

    #cpix=0
    ny,nx=map.shape
    ypix = pixsize.to("arcsec").value # Generally pixel sizes are the same...
    xpix = pixsize.to("arcsec").value # ""
    if xcentre == []:
        xcentre = nx/2.0
    if ycentre == []:
        ycentre = ny/2.0

    #############################################################################
    ### Label w/ the transpose that Python imposes?
    #y = np.outer(np.zeros(ny)+1.0,np.arange(0,xpix*(nx), xpix)- xpix* xcentre)   
    #x = np.outer(np.arange(0,ypix*(ny),ypix)- ypix * ycentre, np.zeros(nx) + 1.0)
    #############################################################################
    ### Intuitive labelling:
    x = np.outer(np.zeros(ny)+1.0,np.arange(nx)*xpix- xpix* xcentre)   
    y = np.outer(np.arange(ny)*ypix- ypix * ycentre, np.zeros(nx) + 1.0)

    #import pdb;pdb.set_trace()
    if oned == True:
        x = x.reshape((nx*ny)) #How important is the tuple vs. integer?
        y = y.reshape((nx*ny)) #How important is the tuple vs. integer?

    
    return x,y
