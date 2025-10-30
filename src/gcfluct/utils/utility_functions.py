import numpy as np

from numpy.typing import NDArray
from typing import Tuple, Union, TypeAlias
Floating: TypeAlias = Union[float, np.float32, np.float64]


def bin_two2Ds(independent: NDArray[Floating],
               dependent: NDArray[Floating],
               binsize: Union[Floating, int] = 1,
               witherr: bool = False,
               withcnt: bool = False
               ) -> Tuple[NDArray[Floating], NDArray[Floating], NDArray[Floating], NDArray[Floating]]:

    """
    A function to bin X (independent) and Y (dependent) variables
    based on X (and binsize).

    Parameters
    ----------
    independent: NDArray[Floating]
        The (2D) array of the independent variable
    dependent: NDArray[Floating]
        The (2D) array of the dependent variable
    binsize: Union[Floating, int]
        The binsize of the independent variable. Default is 1.
    witherr: bool
        Return the error on each bin? This is taken as the RMS within each bin. Default is False.
    withcnt: bool
        Return the count per bin? (How many elements are found within each bin). Default is False.

    Returns
    -------
    abin: NDArray[Floating]
        The binned abscissa (independent) variable.
    obin: NDArray[Floating]
        The binned ordinate (dependent) variable.
    oerr: NDArray[Floating]
        The binned error on the ordinate (dependent) variable. Zeros if witherr==False.
    cnts: NDArray[Floating]
        The binned counts on the ordinate (dependent) variable. Zeros if withcnts==False.
    """

    flatin = independent.flatten()
    flatnt = dependent.flatten()
    inds = flatin.argsort()

    abscissa = flatin[inds]
    ordinate = flatnt[inds]

    nbins = int(np.ceil((np.max(abscissa) - np.min(abscissa))/binsize))
    abin = np.zeros(nbins)
    obin = np.zeros(nbins)
    oerr = np.zeros(nbins)
    cnts = np.zeros(nbins)
    for i in range(nbins):
        blow = i*binsize
        gi = (abscissa >= blow)*(abscissa < blow+binsize)
        abin[i] = np.mean(abscissa[gi])
        obin[i] = np.mean(ordinate[gi])
        if witherr:
            oerr[i] = np.std(ordinate[gi]) / np.sqrt(np.sum(gi))
        if withcnt:
            cnts[i] = np.sum(gi)

    return abin, obin, oerr, cnts


def two2Ds_binned(independent: NDArray[Floating],
                  dependent: NDArray[Floating],
                  bins: NDArray[Floating],
                  witherr: bool = False,
                  withcnt: bool = False
                  ) -> Tuple[NDArray[Floating], NDArray[Floating], NDArray[Floating], NDArray[Floating]]:

    """
    A function to bin X (independent) and Y (dependent) variables
    based on X and specified bins (edges).

    Parameters
    ----------
    independent: NDArray[Floating]
        The (2D) array of the independent variable
    dependent: NDArray[Floating]
        The (2D) array of the dependent variable
    binsize: Union[Floating, int]
        The binsize of the independent variable. Default is 1.
    witherr: bool
        Return the error on each bin? This is taken as the RMS within each bin. Default is False.
    withcnt: bool
        Return the count per bin? (How many elements are found within each bin). Default is False.

    Returns
    -------
    abin: NDArray[Floating]
        The binned abscissa (independent) variable.
    obin: NDArray[Floating]
        The binned ordinate (dependent) variable.
    oerr: NDArray[Floating]
        The binned error on the ordinate (dependent) variable. Zeros if witherr==False.
    cnts: NDArray[Floating]
        The binned counts on the ordinate (dependent) variable. Zeros if withcnts==False.
    """

    flatin = independent.flatten()
    flatnt = dependent.flatten()
    inds = flatin.argsort()

    abscissa = flatin[inds]
    ordinate = flatnt[inds]

    nbins = len(bins)-1
    abin = np.zeros(nbins)
    obin = np.zeros(nbins)
    oerr = np.zeros(nbins)
    cnts = np.zeros(nbins)
    for i in range(nbins):
        gi = (abscissa >= bins[i])*(abscissa < bins[i+1])
        if np.sum(gi) == 0:
            abin[i] = (bins[i] + bins[i+1])/2.0
            obin[i] = 0
        else:
            abin[i] = np.mean(abscissa[gi])
            obin[i] = np.mean(ordinate[gi])
            if witherr:
                oerr[i] = np.std(ordinate[gi]) / np.sqrt(np.sum(gi))
            if withcnt:
                cnts[i] = np.sum(gi)

    return abin, obin, oerr, cnts


def bin_log2Ds(independent: NDArray[Floating],
               dependent: NDArray[Floating],
               nbins: int = 10,
               witherr: bool = False,
               withcnt: bool = False
               ) -> Tuple[NDArray[Floating], NDArray[Floating], NDArray[Floating], NDArray[Floating]]:

    """
    A function to bin X (independent) and Y (dependent) variables
    based on X and number of bins. Spacing is logarthmic and automatically inferred.

    Parameters
    ----------
    independent: NDArray[Floating]
        The (2D) array of the independent variable
    dependent: NDArray[Floating]
        The (2D) array of the dependent variable
    binsize: Union[Floating, int]
        The binsize of the independent variable. Default is 1.
    witherr: bool
        Return the error on each bin? This is taken as the RMS within each bin. Default is False.
    withcnt: bool
        Return the count per bin? (How many elements are found within each bin). Default is False.

    Returns
    -------
    abin: NDArray[Floating]
        The binned abscissa (independent) variable.
    obin: NDArray[Floating]
        The binned ordinate (dependent) variable.
    oerr: NDArray[Floating]
        The binned error on the ordinate (dependent) variable. Zeros if witherr==False.
    cnts: NDArray[Floating]
        The binned counts on the ordinate (dependent) variable. Zeros if withcnts==False.
    """

    flatin = independent.flatten()
    flatnt = dependent.flatten()
    inds = flatin.argsort()

    abscissa = flatin[inds]
    ordinate = flatnt[inds]

    agtz = (abscissa > 0)
    lgkmin = np.log10(np.min(abscissa[agtz])*2.5)
    lgkmax = np.log10(np.max(abscissa))
    bins = np.logspace(lgkmin, lgkmax, nbins+1)
    abin = np.zeros(nbins)
    obin = np.zeros(nbins)
    oerr = np.zeros(nbins)
    cnts = np.zeros(nbins)
    for i, (blow, bhigh) in enumerate(zip(bins[:-1], bins[1:])):
        gi = (abscissa >= blow)*(abscissa < bhigh)
        abin[i] = np.mean(abscissa[gi])
        omean = np.mean(ordinate[gi])
        obin[i] = omean
        if witherr:
            oerr[i] = np.exp(np.std(np.log(ordinate[gi]))) / np.sqrt(np.sum(gi))
        if withcnt:
            cnts[i] = np.sum(gi)

    return abin, obin, oerr, cnts


def plfit(y: NDArray[Floating],
          sy: NDArray[Floating],
          x: NDArray[Floating],
          xp: Floating = 0.0
          ) -> Tuple[Floating, Floating, Floating, Floating]:

    lny = np.log(y)
    slny = sy/y
    lnx = np.log(x)

    # N = len(y)

    wts = 1.0/slny**2
    w = np.sum(wts)
    wxx = np.sum(wts*lnx**2)
    wy = np.sum(wts*lny)
    wx = np.sum(wts*lnx)
    wxy = np.sum(wts*lnx*lny)
    Del = w*wxx - wx**2
    A = (wxx*wy - wx*wxy)/Del
    B = (w*wxy - wx*wy)/Del
    sA = np.sqrt(wxx / Del)
    sB = np.sqrt(w / Del)

    if xp > 0.0:
        lnxp = np.log(xp)

        lnS = A + B*lnxp
        S = np.exp(lnS)
        sS = np.sqrt(sA**2 + (sB*lnxp)**2)*S

        return S, sS

    return A, B, sA, sB


def grid_profile(rads: NDArray[Floating],
                 profile: NDArray[Floating],
                 xymap: Tuple[NDArray[Floating], NDArray[Floating]],
                 geoparams: list = [0, 0, 0, 1, 1, 1, 0, 0],
                 myscale: Floating = 1.0,
                 axis: str = 'z',
                 xyinas: bool = True,
                 ) -> NDArray[Floating]:
    """
    Return a tuple of x- and y-coordinates.

    Parameters
    ----------
    rads : NDArray[Floating]
        An array of radii (same units as xymap)
    profile : NDArray[Floating]
        A radial profile of surface brightness.
    xymap : tuple(NDArray[Floating])
        A tuple of x- and y-coordinates
    geoparams : array-like
        [X_shift, Y_shift, Rotation, Ella*, Ellb*, Ellc*, Xi*, Opening Angle]
    myscale : float
        Generally best to leave as unity.
    axis : str
        Which axis are you projecting along.
    xyinas : bool
        Is the xymap in arcseconds. Default is True.

    Returns
    -------
    mymap : NDArray[Floating]
        An output map

    """

    # Get new grid:
    arc2rad = 4.84813681109536e-06  # arcseconds to radians
    conv = arc2rad if xyinas else 1.0  # Is xymap in arcseconds or radions?
    (x, y) = xymap
    x, y = rot_trans_grid(x, y, geoparams[0], geoparams[1], geoparams[2])  # 0.008 sec per call
    x, y = get_ell_rads(x, y, geoparams[3], geoparams[4])                # 0.001 sec per call
    theta = np.sqrt(x**2 + y**2)*arc2rad
    theta_min = rads[0]*2.0  # Maybe risky, but this is defined so that it is sorted.
    bi = (theta < theta_min)
    theta[bi] = theta_min
    mymap = np.interp(theta, rads, profile)

    if axis == 'x':
        xell = (x/(geoparams[3]*myscale))*conv  # x is initially presented in arcseconds
        modmap = geoparams[5]*(xell**2)**(geoparams[6])  # Consistent with model creation??? (26 July 2017)
    if axis == 'y':
        yell = (y/(geoparams[4]*myscale))*conv  # x is initially presented in arcseconds
        modmap = geoparams[5]*(yell**2)**(geoparams[6])  # Consistent with model creation??? (26 July 2017)
    if axis == 'z':
        modmap = geoparams[5]

    if modmap != 1:
        mymap *= modmap   # Very important to be precise here.
    if geoparams[7] > 0:
        angmap = np.arctan2(y, x)
        bi = (abs(angmap) > geoparams[7]/2.0)
        mymap[bi] = 0.0

    return mymap


def rot_trans_grid(x: NDArray[Floating],
                   y: NDArray[Floating],
                   xs: Floating,
                   ys: Floating,
                   rot_rad: Floating
                   ) -> Tuple[NDArray[Floating], NDArray[Floating]]:
    """
    Shift and rotate coordinates

    Parameters
    ----------
    x : NDArray[Floating]
        coordinate along default x-axis (major axis)
    y : NDArray[Floating]
        coordinate along default y-axis (minor axis)
    xs : Floating
       translation along x-axis
    ys : Floating
       translation along y-axis
    rot_rad : Floating
        rotation angle, in radians

    Returns
    -------
    xnew : NDArray[Floating]
        coordinate along major axis (a)
    ynew : NDArray[Floating]
        coordinate along minor axis (b)
    """

    xnew = (x - xs)*np.cos(rot_rad) + (y - ys)*np.sin(rot_rad)
    ynew = (y - ys)*np.cos(rot_rad) - (x - xs)*np.sin(rot_rad)

    return xnew, ynew


def get_ell_rads(x: NDArray[Floating],
                 y: NDArray[Floating],
                 ella: Floating,
                 ellb: Floating
                 ) -> Tuple[NDArray[Floating], NDArray[Floating]]:
    """
    Get ellipsoidal radii from x,y standard

    Parameters
    ----------
    x : NDArray[Floating]
        coordinates along major axis (a)
    y: NDArray[Floating]
        coordinates along minor axis (b)
    ella: Floating
        scaling along major axis (should stay 1)
    ellb: Floating
        scaling along minor axis

    Returns
    -------
    newx : NDArray[Floating]
        New coordinates along major axis (a)
    newy : NDArray[Floating]
        New coordinates along minor axis (b)
    """

    xnew = x/ella
    ynew = y/ellb

    return xnew, ynew


def get_freqarr_2d(nx: int,
                   ny: int,
                   psx: Floating,
                   psy: Floating
                   ) -> Tuple[NDArray[Floating], Floating, Floating]:
    """
       Compute frequency array for 2D FFT transform

       Parameters
       ----------
       nx : int
            number of samples in the x direction
       ny : int
            number of samples in the y direction
       psx: Floating
            map pixel size in the x direction
       psy: Floating
            map pixel size in the y direction

       Returns
       -------
       k : NDArray[Floating]
           Frequency vector: 2D array, of k_r = sqrt(k_x**2 + k_y**2)
       dkx : Floating
           Stepsize of k_x (k along the x-axis).
       dky : Floating
           Stepsize of k_y (k_along the y-axis).
    """

    kx = np.outer(np.fft.fftfreq(nx), np.zeros(ny).T+1.0)/psx
    ky = np.outer(np.zeros(nx).T+1.0, np.fft.fftfreq(ny))/psy
    dkx = kx[1:][0]-kx[0:-1][0]
    dky = ky[0][1:]-ky[0][0:-1]
    k = np.sqrt(kx*kx + ky*ky)

    return k, dkx[0], dky[0]
