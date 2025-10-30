import numpy as np
from scipy.interpolate import interp1d

from numpy.typing import NDArray
from typing import Union, TypeAlias
Floating: TypeAlias = Union[float, np.float32, np.float64]


def int_profile(profrad: NDArray[Floating],
                profile: NDArray[Floating],
                rad_projected: NDArray[Floating],
                zmax: Union[Floating, int] = 0,
                log_interp: bool = True
                ) -> NDArray[Floating]:
    """
    Numerically integrates a spherical profile along the line of sight. Computation is inherently unitless;
    user must apply appropriate conversions to ensure the output is as desired.

    Parameters
    ---------
    profrad : NDArray[np.float64]
        Array of (3D) radii corresponding to the variable profile.
    profile : NDArray[np.float64]
        Values of the profile corresponding to profrad.
    rad_projected : NDArray[np.float64]
        Array of (2D) radii corresponding to the radii in the plane of the sky (projected radii).
    zmax : np.float64
        If zmax > 0, then sets any |z| > zmax profile value to zero (for integration).
    log_interp : bool
        Internally, another radial grid is created and profile values are interpolated onto it.
        If this is set, then a logarithmic interpolation scheme is set *AND* the profile is extrapolated.
        Otherwise a linear interpolation scheme without interpolation is used. Default is logarithmic.

    Returns
    -------
    int_profile : NDArray[np.float64]
        The projected (integrated) profile at the projected radii specified.
    """

    nrP = len(rad_projected)
    nPr = len(profrad)
    x = np.outer(profrad, np.zeros(nrP)+1.0)
    z = np.outer(np.zeros(nPr)+1.0, rad_projected)
    rad = np.sqrt(x**2 + z**2)
    if log_interp:
        fint = interp1d(np.log(profrad), np.log(profile), bounds_error=False, fill_value="extrapolate")
        log_profile = fint(np.log(rad.reshape(nrP*nPr)))
        rad_profile = np.exp(log_profile)
    else:
        fint = interp1d(profrad, profile, bounds_error=False, fill_value=0)
        rad_profile = fint(rad.reshape(nrP*nPr))
    if zmax > 0:
        zre = z.reshape(nrP*nPr)
        settozero = (zre > zmax)
        rad_profile[settozero] = 0.0
    foo = np.diff(z)
    bar = foo[:, -1]
    diffz = np.insert(foo, -1, bar, axis=1)
    int_profile = 2.0*np.sum(rad_profile.reshape(nPr, nrP)*diffz, axis=1)

    return int_profile
