import numpy as np
from astropy.io import fits
import astropy.units as u
from astropy import wcs
from astropy.io.fits import HDUList

from numpy.typing import NDArray
from typing import Union, TypeAlias, Tuple, Sequence, Optional
Floating: TypeAlias = Union[float, np.float32, np.float64]


def astro_from_hdr(hdr: dict) -> Tuple[NDArray[Floating], NDArray[Floating], Floating]:
    """
    Returns variables with the easy access to the astrometry. That is, it returns grids
    of the right ascension and declination, as well as the pixel size, in arcseconds.

    Parameters
    ----------
    hdr : dict
        The header, from a fits file.

    Returns
    -------
    ras : NDArray[Floating]
        A 2D array (same size as the relevant image) with RA coordinates.
    decs : NDArray[Floating]
        A 2D array (same size as the relevant image) with declination coordinates.
    pixs : Floating
        The pixel size (geometric mean of the size in x and y), in arcseconds.
    """

    xsz = hdr['naxis1']
    ysz = hdr['naxis2']
    xar = np.outer(np.arange(xsz), np.zeros(ysz)+1.0)
    yar = np.outer(np.zeros(xsz)+1.0, np.arange(ysz))
    ####################

    w = wcs.WCS(hdr)

    xcen = hdr['CRPIX1']
    ycen = hdr['CRPIX2']
    dxa = xar - xcen
    dya = yar - ycen
    # RA and DEC in degrees:
    if 'CD1_1' in hdr.keys():
        ras = dxa*hdr['CD1_1'] + dya*hdr['CD2_1'] + hdr['CRVAL1']
        decs = dxa*hdr['CD1_2'] + dya*hdr['CD2_2'] + hdr['CRVAL2']
        pixs = abs(hdr['CD1_1'] * hdr['CD2_2'])**0.5 * 3600.0
    if 'PC1_1' in hdr.keys():
        pcmat = w.wcs.get_pc()
        ras = dxa*pcmat[0, 0]*hdr['CDELT1'] + \
            dya*pcmat[1, 0]*hdr['CDELT2'] + hdr['CRVAL1']
        decs = dxa*pcmat[0, 1]*hdr['CDELT1'] + \
            dya*pcmat[1, 1]*hdr['CDELT2'] + hdr['CRVAL2']
        pixs = abs(pcmat[0, 0]*hdr['CDELT1'] *
                   pcmat[1, 1]*hdr['CDELT2'])**0.5 * 3600.0

    pixs = pixs*u.arcsec
    ras = ras*u.deg
    decs = decs*u.deg

    return ras, decs, pixs


def get_astro(fitsfile, ext=0):
    """
    Returns astrometric variables as well as the image itself.

    Parameters
    ----------
    fitsfile : str
        The path to the (fits) file to be opened.
    ext : int
        The extension of the fits file to be read / used. Default is zero.

    Returns
    -------
    image_data : NDArray[Floating]
        A 2D array -- the image of interest.
    ras : NDArray[Floating]
        A 2D array (same size as the relevant image) with RA coordinates.
    decs : NDArray[Floating]
        A 2D array (same size as the relevant image) with declination coordinates.
    hdr : dict
        The header to the relevant extension of the fits file.
    pixs : Floating
        The pixel size (geometric mean of the size in x and y), in arcseconds.
    """

    hdu = fits.open(fitsfile)
    hdr = hdu[ext].header
    image_data = hdu[ext].data

    ras, decs, pixs = astro_from_hdr(hdr)

    return image_data, ras, decs, hdr, pixs

# I have in mind to create a class. But I'm still thinking about its use (and organization).
# class astrometry:
#
#    def __init__(self, hdr):
#
#        ras,decs,pixs = astro_from_hdr(hdr)
#        self.ras = ras
#        self.decs= decs
#        self.pixs= pixs


def make_template_hdul(nx: int,
                       ny: int,
                       cntr: Sequence,
                       pixsize: Floating,
                       cx: Optional[Floating] = None,
                       cy: Optional[Floating] = None
                       ) -> HDUList:
    """
    Return an HDU object (see astropy).

    Parameters
    ----------
    nx : int
       Number of pixels along axis 0
    ny : int
       Number of pixels along axis 1
    cntr : Sequence
       Two-element object specifying the RA and Dec of the center.
    pixsize : Floating
       Pixel size, in arcseconds
    cx : Optional[Floating]
       The pixel center along axis 0
    cy : Optional[Floating]
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
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [cx, cy]
    w.wcs.cdelt = np.array([-pixsize/3600.0, pixsize/3600.0])
    w.wcs.crval = [cntr[0], cntr[1]]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    hdr = w.to_header()

    zero_img = np.zeros((nx, ny))
    phdu = fits.PrimaryHDU(zero_img, header=hdr)
    temp_hdu = fits.HDUList([phdu])

    return temp_hdu


def get_xymap(map: NDArray[Floating],
              pixsize: Floating,
              xcentre: Optional[Floating] = None,
              ycentre: Optional[Floating] = None,
              oned: bool = True
              ) -> Tuple[NDArray[Floating], NDArray[Floating]]:
    """
    Returns maps of X and Y coordinates (from the center) in arcseconds.

    INPUTS:
    -------
    map : NDArray[Floating]
        a 2D array for which you want to construct the xymap
    pixsize : Quantity
        a quantity (with units of an angle)
    xcentre : Optional[Floating]
        The number of the pixel that marks the X-centre of the map
    ycentre : Optional[Floating]
        The number of the pixel that marks the Y-centre of the map
    oned : bool
        Return X- and Y-arrays as 1D arrays (and not 2D, as the image is).
        Default is True

    Returns
    -------
    x : NDArray[Floating]
        An array (1D or 2D, per user input) of the x-coordinates.
    y : NDArray[Floating]
        An array (1D or 2D, per user input) of the y-coordinates.
    """

    ny, nx = map.shape
    ypix = pixsize.to("arcsec").value  # Generally pixel sizes are the same...
    xpix = pixsize.to("arcsec").value  # ""
    if xcentre is None:
        xcentre = nx/2.0
    if ycentre is None:
        ycentre = ny/2.0

    x = np.outer(np.zeros(ny) + 1.0, np.arange(nx)*xpix - xpix*xcentre)
    y = np.outer(np.arange(ny)*ypix - ypix*ycentre, np.zeros(nx) + 1.0)

    if oned:
        x = x.reshape((nx*ny))  # How important is the tuple vs. integer?
        y = y.reshape((nx*ny))  # How important is the tuple vs. integer?

    return x, y
