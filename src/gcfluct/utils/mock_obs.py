import numpy as np
import gcfluct.utils.utility_functions as uf
from numpy.typing import NDArray
from gcfluct.gc.selfsimilar_gc import SS_Model
from typing import Union, TypeAlias
Floating: TypeAlias = Union[float, np.float32, np.float64]


def mock_XMM_exposure(obj_w_map: SS_Model,
                      ksec: Floating,
                      rot: Floating = 30,
                      incChipGaps: bool = True
                      ) -> NDArray[Floating]:
    """
    Creates a mock XMM exposure (very much a toy-model!).

    Parameters
    ----------
    obj_w_map : SS_Model
        Any object with map (grid) information.
    ksec : Floating
        Duration of the (mock) clean exposure in kiloseconds.
    rot : Floating
        Rotation of chip gaps, in degrees. In general this should be arbitrary. Default is 30.
    incChipGaps : bool
        Include chip gaps in the exposure map? Default is True.

    Returns
    -------
    exposure_map : NDArray[Floating]
        An exposure map consistent with the map(s) of the input object.
    """
    # How to parameterize the radial profile of the relative exposure.

    rads_arcsec = np.logspace(0, 3.5, 500)  # Out to 3000 arcseconds (~almost a a degree)
    beta_expo = 1.0/3.0
    r_c = 6*60  # 5 arcminutes = 300 arcseconds. Half-exposure at roughly 9 arcminutes

    relative_exposure = (1 + (rads_arcsec/r_c)**2)**(-1.5*beta_expo)
    exposure_profile = ksec*relative_exposure*1000
    xymap = (obj_w_map._xmat, obj_w_map._ymat)
    exposure_map = uf.grid_profile(rads_arcsec, exposure_profile, xymap)

    if incChipGaps:

        GapPars = [3.0, 5.0]  # Arcseconds
        xmap, ymap = xymap
        Line1 = xmap*np.sin(rot*np.pi/180) + 45
        Diff1 = np.abs(ymap - Line1)
        Gap1 = chip_gap_cosine(Diff1, GapPars)
        exposure_map *= Gap1

        Line2 = 150 - ymap*np.sin(rot*np.pi/180)
        Line3 = -90 - ymap*np.sin(rot*np.pi/180)
        Diff2 = np.abs(xmap - Line2)
        Gap2 = chip_gap_cosine(Diff2, GapPars)
        Diff3 = np.abs(xmap - Line3)
        Gap3 = chip_gap_cosine(Diff3, GapPars)
        exposure_map *= Gap2*Gap3

    return exposure_map


def chip_gap_cosine(r: NDArray[Floating],
                    par: list
                    ) -> NDArray[Floating]:
    """
    A helper function to create the chip gaps as a smooth transition (and not a step function).

    Parameters
    ----------
    r : NDArray[Floating]
        A relevant array of coordinates (not necessarily a radius).
    par : list
        A two-element list of where the transition begins and ends.

    Returns
    -------
    chip_gap : NDArray[Floating]
        Array of values between 1 and 0 (inclusive) to immidate a (single) chip gap.
    """

    r1 = par[0]
    r2 = par[1]
    chip_gap = r*0.0
    chip_gap[r < r1] = 0.0
    chip_gap[r >= r1] = 0.5 * (1-np.cos(np.pi*(r[r >= r1]-r1)/(r2-r1)))
    chip_gap[r > r2] = 1.0

    return chip_gap
