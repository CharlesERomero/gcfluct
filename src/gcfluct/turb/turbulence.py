import numpy as np 
import astropy.units as u

from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union
from astropy.units import Quantity
from numpy.typing import NDArray



def nelson_alpha(rads,r200m,A=0.452,B=0.841,gNT=1.628):
    
    """
    Returns a profile of non-thermal pressure support, quantified as alpha.
    See Nelson et al (2014).
 
    Values as in Nelson et al (2014):
    A    = 0.452 +/- 0.001
    B    = 0.841 +/- 0.0008
    gNT  = 1.628 +/- 0.019
    
    """
    
    myexp = (rads/(B*r200m) ).decompose().value
    g     = (1.0 + np.exp(-myexp**gNT))
    alpha = 1 - A*g
    
    return alpha

def battaglia_alpha(rads,M200,R500,z,nNT=0.8,nM=0.2,anot=0.18,beta=0.5):
    
    """
    Returns a profile of non-thermal pressure support, quantified as alpha.
    See Battaglia et al (2012).
 
    anot = 0.18 +/- 0.06
    nNT  = 0.8  +/- 0.25
    
    """
    
    alpha_z = anot*(1+z)**beta
    Rscale  = (rads / R500).decompose().value
    Mscale  = (M200 / (3e14*u.M_sun)).decompose().value 
    
    alpha   = alpha_z * (Rscale**nNT)*(Mscale**nM)
    
    return alpha

def pNT_from_P_alpha(rads,P,alpha):
    """
    Converts alpha into a more common non-thermal pressure fraction.
    
    """
    
    pNT = alpha*P / (1 - alpha)
    
    return pNT

def mach_from_alpha(alpha,Adiabatic=5.0/3.0):
    
    pNT_pTh = alpha / (1 - alpha)
    mach3d  = np.sqrt( pNT_pTh / (Adiabatic/3.0) )
    
    return mach3d
    
def lnPnt_lnPth(rads,machs,pTh):
    
    lnMlnr = np.diff(np.log(machs)) / np.diff(np.log(rads))
    lnPlnr = np.diff(np.log(pTh)) / np.diff(np.log(rads))
    
    dlnPnt_dlnPth = 1 + 2*lnMlnr/lnPlnr
    
    return dlnPnt_dlnPth

def HydroBias(machs,dlnPnt_dlnPth,adiabatic=5.0/3.0):
    
    t1 = (-adiabatic/3)*machs[1:]**2 * dlnPnt_dlnPth
    t2 = 1.0 + (adiabatic/3)*machs[1:]**2 * dlnPnt_dlnPth
    
    b_mach = t1/t2
    
    return b_mach
