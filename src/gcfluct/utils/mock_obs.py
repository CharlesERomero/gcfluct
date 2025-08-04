import numpy as np

import gcfluct.utils.utility_functions as uf

def mock_XMM_exposure(obj_w_map,ksec,rot=30,incChipGaps=True):

    ### How to parameterize the radial profile of the relative exposure.

    rads_arcsec = np.logspace(0,3.5,500) # Out to 3000 arcseconds (~almost a a degree)
    beta_expo = 1.0/3.0
    r_c = 6*60 # 5 arcminutes = 300 arcseconds. Half-exposure at roughly 9 arcminutes

    relative_exposure = (1 + (rads_arcsec/r_c)**2)**(-1.5*beta_expo)
    exposure_profile = ksec*relative_exposure*1000

    xymap = (obj_w_map._xmat,obj_w_map._ymat)
    
    exposure_map = uf.grid_profile(rads_arcsec, exposure_profile, xymap)

    if incChipGaps:

        GapPars = [3.0,5.0] # Arcseconds 
        xmap,ymap = xymap
        Line1 = xmap*np.sin(rot*np.pi/180) + 45
        Diff1 = np.abs(ymap - Line1)
        Gap1 = chip_gap_cosine(Diff1,GapPars)
        exposure_map *= Gap1

        Line2 = 150 - ymap*np.sin(rot*np.pi/180)
        Line3 = -90 - ymap*np.sin(rot*np.pi/180)
        Diff2 = np.abs(xmap - Line2)
        Gap2 = chip_gap_cosine(Diff2,GapPars)
        Diff3 = np.abs(xmap - Line3)
        Gap3 = chip_gap_cosine(Diff3,GapPars)        
        exposure_map *= Gap2*Gap3
        
    return exposure_map

def chip_gap_cosine(r,par):
    r1 = par[0]
    r2 = par[1]
    chip_gap = r*0.0
    chip_gap[r < r1]  = 0.0
    chip_gap[r >= r1] = 0.5 * (1-np.cos(np.pi*(r[r >= r1]-r1)/(r2-r1)))
    chip_gap[r > r2]  = 1.0
    
    return chip_gap
