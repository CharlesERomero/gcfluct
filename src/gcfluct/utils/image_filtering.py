import numpy as np

from numpy.typing import NDArray
from typing import Union, TypeAlias, Tuple, Sequence
Floating: TypeAlias = Union[float, np.float32, np.float64]


def get_freqarr_only_2d(nx: int,
                        ny: int,
                        psx: Union[Floating, int],
                        psy: Union[Floating, int]
                        ) -> NDArray[Floating]:
    """
    Compute frequency array for 2 D FFT transform

    Parameters
    ----------
    nx : int
        number of samples in the x direction
    ny : int
        number of samples in the y direction
    psx: int
        map pixel size in the x direction
    psy: int
        map pixel size in the y direction

    Returns
    -------
    k : float 2D numpy array
        frequency vector
    """
    kx = np.outer(np.fft.fftfreq(nx), np.zeros(ny).T+1.0)/psx
    ky = np.outer(np.zeros(nx).T+1.0, np.fft.fftfreq(ny))/psy
    k = np.sqrt(kx*kx + ky*ky)
    return k


def power_spectrum_2d(arr: NDArray[Floating],
                      nbins: int = 10,
                      psx: Union[Floating, int] = 1,
                      psy: Union[Floating, int] = 1,
                      logbins: bool = True
                      ) -> Tuple[NDArray[Floating], NDArray[Floating], NDArray[Floating]]:
    """
    Compute 2D power spectrum of arr, via numpy FFT.

    Parameters
    ----------
    arr: NDArray[Floating]
         2D array for which we compute the power spectrum
    nbins: int
         number of frequency k bins (10)
    psx: Union[Floating, int]
        Pixel size in the x-direction. Default is 1 (units as interpreted by user).
    psy: Union[Floating, int]
        Pixel size in the y-direction. Default is 1 (units as interpreted by user).
    logbins: bool
        Use logarithmic binning? Default is True.

    Returns
    -------
    kbin: NDArray[Floating]
         1D array of bins in k-space
    pkbin: NDArray[Floating]
         1D power spectrum for kbin
    pkbins: NDArray[Floating]
         1D power spectrum uncertainty, taken as the standard deviation of the mean (within each bin).
    """

    farr = np.fft.fft2(arr)/np.double(arr.size)
    nx, ny = arr.shape
    k = get_freqarr_only_2d(nx, ny, psx, psy)
    pk = np.double(farr * np.conj(farr))
    if logbins:
        kbinarr = np.logspace(0.0, np.log(k.max()), nbins+1)
    else:
        kbinarr = np.arange(nbins+1)/np.double(nbins)*(k.max()-k.min())
    kbin = np.zeros(nbins+1)
    pkbin = np.zeros(nbins+1)
    pkbins = np.zeros(nbins+1)

    for idx in range(0, nbins):
        indices = np.nonzero((k > kbinarr[idx]) * (k <= kbinarr[idx+1]))
        kbin[idx+1] = np.median(k[indices])
        pkbin[idx+1] = np.mean(pk[indices])
        pkbins[idx+1] = np.std(pk[indices])/np.sqrt(len(indices[0]))
    return kbin, pkbin, pkbins


# def simu_gaussian_noise(nx,ny,pk):
#    """
#     Simulate a 2D map assuming Gaussian noise as computed from 2D
#     power spectrum
#
#
#     To be DONE !!!!!
#    """
#    simumap = np.random.normal(0.0,1.0,[nx,ny])
#    karr = get_freqarr_only_2d(nx,ny,psx, psy)


def cross_power_spectrum_2d(arr1: NDArray[Floating],
                            arr2: NDArray[Floating],
                            nbins: int = 10,
                            psx: Union[Floating, int] = 1,
                            psy: Union[Floating, int] = 1,
                            logbins: bool = True
                            ) -> Tuple[NDArray[Floating], NDArray[Floating], NDArray[Floating]]:
    """
    Compute 2D cross-power spectrum of arr1 and arr2, via numpy FFT.

    Parameters
    ----------
    arr1: NDArray[Floating]
         2D array (image) to use in cross-spectrum calculation.
    arr2: NDArray[Floating]
         Complementary 2D array (image) to use in cross-spectrum calculation.
    nbins: int
         number of frequency k bins (10)
    psx: Union[Floating, int]
        Pixel size in the x-direction. Default is 1 (units as interpreted by user).
    psy: Union[Floating, int]
        Pixel size in the y-direction. Default is 1 (units as interpreted by user).
    logbins: bool
        Use logarithmic binning? Default is True.

    Returns
    -------
    kbin: NDArray[Floating]
         1D array of bins in k-space
    pkbin: NDArray[Floating]
         1D power spectrum for kbin
    pkbins: NDArray[Floating]
         1D power spectrum uncertainty, taken as the standard deviation of the mean (within each bin).
    """
    nx, ny = arr1.shape
    nx1, ny1 = arr2.shape
    if nx1 == nx:
        if ny1 == ny:
            farr1 = np.fft.fft2(arr1)/np.double(arr1.size)
            farr2 = np.fft.fft2(arr2)/np.double(arr2.size)
            k = get_freqarr_only_2d(nx, ny, psx, psy)
            pk = np.double(farr1 * np.conj(farr2))
            if logbins:
                kbinarr = np.logspace(0.0, np.log(k.max()), nbins+1)
            else:
                kbinarr = np.arange(nbins+1)/np.double(nbins)*(k.max()-k.min())
            kbin = np.zeros(nbins+1)
            pkbin = np.zeros(nbins+1)
            pkbins = np.zeros(nbins+1)
            for idx in range(0, nbins):
                indices = np.nonzero((k > kbinarr[idx]) * (k <= kbinarr[idx+1]))
                kbin[idx+1] = np.median(k[indices])
                pkbin[idx+1] = np.mean(pk[indices])
                pkbins[idx+1] = np.std(pk[indices])/np.sqrt(len(indices[0]))

    return kbin, pkbin, pkbins


def fourier_conv_2d(arr: NDArray[Floating],
                    kernel: NDArray[Floating],
                    doshift: bool = True
                    ) -> NDArray[Floating]:
    """
    Performs a convolution of an input image (arr) with a kernel (kernel).
    If kernel (e.g. a Gaussian) is peaked in the center, then you
    want to have doshift=True.

    Parameters
    ----------
    arr: NDArray[Floating]
        The input image - a 2D numpy array.
    kernel: NDArray[Floating]
        The convolution kernel, also a 2D numpy array.
    doshift: bool
        Shift output image via numpy.fft.fftshift. Default is True.
    """
    farr = np.fft.fft2(arr)
    fker = np.fft.fft2(kernel)
    farr = farr * fker
    arrc = np.real(np.fft.ifft2(farr))
    if doshift:
        arrc = np.fft.fftshift(arrc)

    return arrc


def fourier_filtering_2d(arr: NDArray[Floating],
                         filt_type: str,
                         par: Union[list, Floating, Tuple]
                         ) -> NDArray[Floating]:
    """
    Performs one of 5 preset types of Fourier filtering on an image.

    Parameters
    ----------
    arr: NDArray[Floating]
        The input image - a 2D numpy array.
    filt_type: str
        Can be one of: "gauss", "hpcos", "lpcos", "bpcos", or "tab". The respective options correspond to a Gaussian filter,
        cosine high-pass, cosine low-pass, cosine high- and low-pass, or a tabulated filter (transfer) function.
    par: Union[list, Floating, Tuple]
        The parameter type will depend on the filter type. If Gaussian, it expects a scalar corresponding to the FWHM.
        If hpcos or lpcos, it expects a two-element iterable. If bpcos, it expects a four-element iterable.
        If tab, then it expects a two-tuple of (numpy arrays): wavenumber and associated transfer value.

    Returns
    -------
    arrfilt: NDArray[Floating]
        The filtered image
    """

    farr = np.fft.fft2(arr)
    nx, ny = arr.shape
    kx = np.outer(np.fft.fftfreq(nx), np.zeros(ny).T+1.0)
    ky = np.outer(np.zeros(nx).T+1.0, np.fft.fftfreq(ny))
    k = np.sqrt(kx*kx + ky*ky)
    # Perhaps use "case"?
    if filt_type == 'gauss':
        filter = gauss_filter_2d(k, par)
    if filt_type == 'hpcos':
        filter = hpcos_filter_2d(k, par)
    if filt_type == 'lpcos':
        filter = lpcos_filter_2d(k, par)
    if filt_type == 'bpcos':
        filter = bpcos_filter_2d(k, par)
    if filt_type == 'tab':
        filter = table_filter_2d(k, par)
    farr = farr * filter
    arrfilt = np.real(np.fft.ifft2(farr))

    return arrfilt


def gauss_filter_2d(k: NDArray[Floating],
                    par: Floating
                    ) -> NDArray[Floating]:
    """
    Returns the Gaussian smoothing kernel, in Fourier space.

    Parameters
    ----------
    arr: NDArray[Floating]
        The input image - a 2D numpy array.
    par: Floating
        A scalar corresponding to the FWHM.

    Returns
    -------
    filter: NDArray[Floating]
        The filtering kernel, in Fourier space.
    """

    fwhm = par
    sigma = fwhm/(2.0*np.sqrt(2.0*np.log(2)))
    filter = np.exp(-2.0*k*k*sigma*sigma*np.pi*np.pi)

    return filter


def lpcos_filter_2d(k: NDArray[Floating],
                    par: Sequence
                    ) -> NDArray[Floating]:
    """
    Returns the low-pass cosine filter, in Fourier space.

    Parameters
    ----------
    arr: NDArray[Floating]
        The input image - a 2D numpy array.
    par: Sequence
        A two-element iterable; the first element denotes the starting wavenumber (transmission is unity) and the second
        element denotes the ending wavenumber (transmission is null).

    Returns
    -------
    filter: NDArray[Floating]
        The filtering kernel, in Fourier space.
    """
    k1 = par[0]
    k2 = par[1]
    filter = k*0.0
    filter[k < k1] = 1.0
    filter[k >= k1] = 0.5 * (1+np.cos(np.pi*(k[k >= k1]-k1)/(k2-k1)))
    filter[k > k2] = 0.0
    return filter


def hpcos_filter_2d(k: NDArray[Floating],
                    par: Sequence
                    ) -> NDArray[Floating]:
    """
    Returns the high-pass cosine filter, in Fourier space.

    Parameters
    ----------
    arr: NDArray[Floating]
        The input image - a 2D numpy array.
    par: Sequence
        A two-element iterable; the first element denotes the starting wavenumber (transmission is null) and the second
        element denotes the ending wavenumber (transmission is unity).

    Returns
    -------
    filter: NDArray[Floating]
        The filtering kernel, in Fourier space.
    """
    k1 = par[0]
    k2 = par[1]
    filter = k*0.0
    filter[k < k1] = 0.0
    filter[k >= k1] = 0.5 * (1-np.cos(np.pi*(k[k >= k1]-k1)/(k2-k1)))
    filter[k > k2] = 1.0
    return filter


def bpcos_filter_2d(k: NDArray[Floating],
                    par: Sequence
                    ) -> NDArray[Floating]:
    """
    Returns the high- and low-pass cosine filter, in Fourier space.

    Parameters
    ----------
    arr: NDArray[Floating]
        The input image - a 2D numpy array.
    par: Sequence
        A four-element iterable; the first two correspond to the high-pass filtering parameters and
        the last two elements correspond to the low-pass filtering parameters (frequencies).

    Returns
    -------
    filter: NDArray[Floating]
        The filtering kernel, in Fourier space.
    """
    filter = hpcos_filter_2d(k, par[0:2]) * lpcos_filter_2d(k, par[2:4])
    return filter


def gauss_2d(sigma: Floating,
             nx: int,
             ny: int) -> NDArray[Floating]:
    """
    Parameters
    ----------
    sigma: Floating
        Guassian sigma, in pixels.
    nx: int
        Number of pixels along the x-axis.
    ny: int
        Number of pixels along the x-axis.

    Returns
    -------
    fg: NDArray[Floating]
        2D Guassian, in linear (real) space.
    """

    ix = np.outer(np.arange(nx), np.zeros(ny).T+1)-nx/2
    iy = np.outer(np.zeros(nx)+1, np.arange(ny).T)-ny/2
    r = ix*ix+iy*iy
    fg = np.exp(-0.5*r/sigma/sigma)

    return fg


def table_filter_2d(k: NDArray[Floating],
                    par: Floating
                    ) -> NDArray[Floating]:
    """
    Returns the interpolation of a tabulated filter (transfer function).

    Parameters
    ----------
    arr: NDArray[Floating]
        The input image - a 2D numpy array.
    par: Tuple[NDArray[Floating], NDArray[Floating]]
        A two-element tuple, wherein the first element is a 1D array of wavenumber and the second element is a
        1D array of the transmission.

    Returns
    -------
    filter: NDArray[Floating]
        The filtering kernel, in Fourier space.
    """

    from scipy import interpolate
    kbin, filterbin = par
    f = interpolate.interp1d(kbin, filterbin)
    kbin_min = kbin.min()
    kbin_max = kbin.max()

    filter = k * 0.0
    filter[(k >= kbin_min) & (k <= kbin_max)] = f(k[(k >= kbin_min) & (k <= kbin_max)])   # use interpolation function returned by `interp1d`
    filter[(k < kbin_min)] = filterbin[kbin == kbin_min]
    filter[(k > kbin_max)] = filterbin[kbin == kbin_max]

    return filter


def fourier_filtering_1d(arr: NDArray[Floating],
                         filt_type: str,
                         par: Tuple[NDArray[Floating], NDArray[Floating]],
                         xarr=[]):
    """
    Returns the filtered 1D array

    Parameters
    ----------
    arr: NDArray[Floating]
        The input image - a 1D numpy array.
    filt_type: str
        Currently only "tab" is accepted. There is the intention to generalize this to include "hpcos", "lpcos",
        "bpcos" and "gauss" as in the 2D version.
    par: Tuple[NDArray[Floating], NDArray[Floating]]
        A two-element tuple, wherein the first element is a 1D array of wavenumber and the second element is a
        1D array of the transmission.

    Returns
    -------
    arrfilt: NDArray[Floating]
        The filtered array.
    """

    farr = np.fft.fft(arr)
    # Not the best construction. Will leave for now.
    if len(xarr) == 0:
        nx = len(arr)         # it's strictly one dimensional!
        k = np.fft.fftfreq(nx)
    else:
        nx = len(arr)         # it's strictly one dimensional!
        raise Exception

    if filt_type == 'tab':
        filter = table_filter_1d(k, par)
    farr = farr * filter
    arrfilt = np.real(np.fft.ifft(farr))
    return arrfilt


def table_filter_1d(k: NDArray[Floating],
                    par: Floating
                    ) -> NDArray[Floating]:
    """
    Returns the interpolation of a tabulated filter (transfer function).

    Parameters
    ----------
    arr: NDArray[Floating]
        The input 1D numpy array.
    par: Tuple[NDArray[Floating], NDArray[Floating]]
        A two-element tuple, wherein the first element is a 1D array of wavenumber and the second element is a
        1D array of the transmission.

    Returns
    -------
    filter: NDArray[Floating]
        The filtering kernel, in Fourier space.
    """

    from scipy import interpolate
    kbin, filterbin = par
    f = interpolate.interp1d(kbin, filterbin)
    kbin_min = kbin.min()
    kbin_max = kbin.max()

    filter = k * 0.0

    filter[(k >= kbin_min) & (k <= kbin_max)] = f(k[(k >= kbin_min) & (k <= kbin_max)])   # use interpolation function returned by `interp1d`
    filter[(k < kbin_min)] = filterbin[kbin == kbin_min]
    filter[(k > kbin_max)] = filterbin[kbin == kbin_max]

    return filter
