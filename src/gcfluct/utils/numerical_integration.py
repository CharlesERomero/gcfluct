import numpy as np
import scipy as sp
from scipy.interpolate import interp1d
import astropy.units as u
import analytic_integrations as ai
import os
import InstrumentSpecific as IS

def interp_plaws(nodes,norms,alphas,radii):
    """
    Assumes you'll want values extrapolated to min of outradii and max of outradii
    """
    for node,norm,alpha in zip(nodes,norms,alphas):
        if node == np.min(nodes):
            vals = norm*(radii/node)**alpha
        else:
            gi = (radii > node)
            vals[gi] = norm*(radii[gi]/node)**alpha

    return vals

def int_profile(profrad, profile,radProjected,zmax=0):
    """
    This currently only integrates out to the max of profrad. If you want
    to give a *fixed z*, you should be sure it is LESS THAN the max of
    profrad, and then adjust the code below.

    You likely want profrad to be in kpc. In this way, you will integrate
    units of pressure over kpc, and the resultant units are comprehensible.

    radProjected is an array of z-coordinates along the line of sight.
    """
    
    nrP = len(radProjected); nPr=len(profrad)
    x = np.outer(profrad,np.zeros(nrP)+1.0)
    z = np.outer(np.zeros(nPr)+1.0,radProjected)
    rad = np.sqrt(x**2 + z**2)
    fint = interp1d(profrad, profile, bounds_error = False, fill_value = 0)
    radProfile = fint(rad.reshape(nrP*nPr))
    if zmax > 0:
        zre = z.reshape(nrP*nPr); settozero = (zre > zmax)
        radProfile[settozero] = 0.0
    foo =np.diff(z); bar =foo[:,-1];peloton=radProfile.reshape(nPr,nrP)
    diffz = np.insert(foo,-1,bar,axis=1)
    intProfile = 2.0*np.sum(radProfile.reshape(nPr,nrP)*diffz,axis=1)
    
    return intProfile

def int_tapered_profile_per_theta(profrad, profile,radProjected,mytheta,thetamin=0.4,thetamax=0.6,
                                  zmax=0):
    """
    This currently only integrates out to the max of profrad. If you want
    to give a *fixed z*, you should be sure it is LESS THAN the max of
    profrad, and then adjust the code below.

    NEW EDIT.
    profrad and radProjected should be in RADIANS!!!
    This will make integrating the taper function easier.

    YOU MUST THEREFORE HAVE profile CONVERTED TO THE APPROPRIATE UNITS TO 
    MAKE THIS INTEGRATION CORRECT (scaling/unit-wise).

    radProjected is an array of z-coordinates along the line of sight.
    """

    #test= 2.0*np.sum(z2,axis=1)
    #thetamin = 0.4
    #thetamax = 0.6
    #profrad  = np.arange(20)
    #profile  = 450.0 - profrad**2
    #radProjected = np.arange(10)

    deltat  = thetamax-thetamin
    
    nrP = len(radProjected); nPr=len(profrad)
    x = np.outer(radProjected,np.ones(nrP))
    y = x * np.tan(mytheta)
    z = np.outer(np.ones(nrP),profrad)
    #print(x.shape,y.shape,z.shape)
    rad     = np.sqrt(x**2 + y**2 + z**2)
    polar   = np.sqrt(z**2 + y**2)
    theta   = np.arctan(polar/x)

    tgttmin = (theta > thetamin)
    tlttmax = (theta < thetamax)
    tgttmax = (theta > thetamax)
    tcos    = tgttmin*tlttmax
    taper   = np.ones(theta.shape)
    taper[tcos] = 0.5 + 0.5*np.cos((theta[tcos]-thetamin)*np.pi/deltat)
    taper[tgttmax] = 0

    radflat = rad.reshape(x.size)
    radProfile = np.interp(radflat,profrad,profile,left=0,right=0)
    #fint = interp1d(profrad, profile, bounds_error = False, fill_value = 0)
    #radProfile = fint(rad.reshape(x.size))
    if zmax > 0:
        zre = z.reshape(nrP*nPr); settozero = (zre > zmax)
        radProfile[settozero] = 0.0
    foo =np.diff(z); bar =foo[:,-1]
    diffz = np.insert(foo,-1,bar,axis=1)
    Pressure3d = radProfile.reshape(taper.shape) * taper
    intProfile = 2.0*np.sum(Pressure3d*diffz,axis=1)
    #print("Why not")
    
    #import pdb; pdb.set_trace()
    
    return intProfile

def loop_int_tppt(thetas,profrad, profile,radProjected,thetamin=0.4,thetamax=0.6,zmax=0):

    deltat  = thetamax-thetamin
    
    nrP = len(radProjected); nPr=len(profrad)
    x = np.outer(radProjected,np.ones(nrP))
    z = np.outer(np.ones(nrP),profrad)

    for mytheta in thetas:
        y = x * np.tan(mytheta)
        rad     = np.sqrt(x**2 + y**2 + z**2)
        polar   = np.sqrt(z**2 + y**2)
        theta   = np.arctan(polar/x)
        tgttmin = (theta > thetamin)
        tlttmax = (theta < thetamax)
        tgttmax = np.where(theta > thetamax)
        tcos    = np.where(tgttmin*tlttmax)
        taper   = np.ones(theta.shape)
        taper[tcos] = 0.5 + 0.5*np.cos((theta[tcos]-thetamin)*np.pi/deltat)
        taper[tgttmax] = 0

        radflat = rad.reshape(x.size)
        radProfile = np.interp(radflat,profrad,profile,left=0,right=0)
        if zmax > 0:
            zre = z.reshape(nrP*nPr); settozero = (zre > zmax)
            radProfile[settozero] = 0.0
        foo =np.diff(z); bar =foo[:,-1]
        diffz = np.insert(foo,-1,bar,axis=1)
        Pressure3d = radProfile.reshape(taper.shape) * taper
        intProfile = 2.0*np.sum(Pressure3d*diffz,axis=1)

        if (mytheta == 0):
            Int_Prof = intProfile
        else:
            Int_Prof = np.vstack((Int_Prof,intProfile))

             
    return Int_Prof

def int_tapered_profile_old(profrad, profile,radProjected,thetamin=0.4,thetamax=0.6,
                            zmax=0,twod=False):
    """
    This currently only integrates out to the max of profrad. If you want
    to give a *fixed z*, you should be sure it is LESS THAN the max of
    profrad, and then adjust the code below.

    NEW EDIT.
    profrad and radProjected should be in RADIANS!!!
    This will make integrating the taper function easier.

    YOU MUST THEREFORE HAVE profile CONVERTED TO THE APPROPRIATE UNITS TO 
    MAKE THIS INTEGRATION CORRECT (scaling/unit-wise).

    radProjected is an array of z-coordinates along the line of sight.
    """

    #test= 2.0*np.sum(z2,axis=1)
    #thetamin = 0.4
    #thetamax = 0.6
    #profrad  = np.arange(20)
    #profile  = 450.0 - profrad**2
    #radProjected = np.arange(10)

    deltat  = thetamax-thetamin
    
    nrP = len(radProjected); nPr=len(profrad)
    x2 = np.outer(radProjected,np.ones(nrP))
    y2 = np.outer(np.ones(nrP),radProjected)
    x = np.repeat(x2[:, :,np.newaxis],nPr,axis=2)
    y = np.repeat(y2[:, :,np.newaxis],nPr,axis=2)
    z2 = np.outer(np.ones(nrP),profrad)
    z = np.repeat(z2[np.newaxis,:, :],nrP,axis=0)
    print(x.shape,y.shape,z.shape)
    rad     = np.sqrt(x**2 + y**2 + z**2)
    polar   = np.sqrt(z**2 + y**2)
    theta   = np.arctan(polar/x)

    tgttmin = (theta > thetamin)
    tlttmax = (theta < thetamax)
    tgttmax = (theta > thetamax)
    tcos    = tgttmin*tlttmax
    taper   = np.ones(theta.shape)
    taper[tcos] = 0.5 + 0.5*np.cos((theta[tcos]-thetamin)*np.pi/deltat)
    taper[tgttmax] = 0

    radflat = rad.reshape(x.size)
    radProfile = np.interp(radflat,profrad,profile,left=0,right=0)
    #fint = interp1d(profrad, profile, bounds_error = False, fill_value = 0)
    #radProfile = fint(rad.reshape(x.size))
    if zmax > 0:
        zre = z.reshape(nrP*nPr); settozero = (zre > zmax)
        radProfile[settozero] = 0.0
    foo =np.diff(z); bar =foo[:,:,-1]
    diffz = np.insert(foo,-1,bar,axis=2)
    Pressure3d = radProfile.reshape(taper.shape) * taper
    intProfile = 2.0*np.sum(Pressure3d*diffz,axis=2)

    #    if twod:
    return intProfile, (x2,y2)
    
def create_profile_alla_cer(hk,dv,efv,nw=True,plot=False,finite=False):

    prmunit = (efv.prmunit*u.keV).to("cm**3")
    uless_r, edensity, etemperature, geoparams, inalphas = ai.prep_a2146_binsky(hk.hk_ins,nw=nw)
    edensity = edensity*(prmunit.value) # Incorportate all the relevant factors. (~3.16)
    uless_rad = (uless_r/dv.cluster.d_a).value
    kpc_range = (dv.mapping.theta_range*dv.cluster.d_a).value;
    if finite == False:
        myrs=np.append(uless_r,500.0)
    else:
        myrs = uless_r
        
    if nw == True:
        ealp = hk.hk_ins.ealp_nw; talp = hk.hk_ins.talp_nw
    else:
        ealp = hk.hk_ins.ealp_se; talp = hk.hk_ins.talp_se

    myprof = kpc_range*0.0; sindex = ealp+talp
    for idx in range(len(myrs)-1):
        epsnot = edensity[idx]*etemperature[idx]; rmin = myrs[idx]; rmax = myrs[idx+1]
        tSZ,kSZ,int_factors = get_SZ_factors(etemperature[idx],dv,hk,efv,beta=0.0,betaz=0.0)
        if idx == 0:
            rmin = rmax/1000.0
        gi = (kpc_range > rmin) & (kpc_range < rmax)
        if idx == 0:
            myprof[gi] = epsnot*(kpc_range[gi]/rmax)**sindex[idx]
        else:
            myprof[gi] = (kpc_range[gi]/rmin)**sindex[idx]
        myprof[gi]*= int_factors*tSZ

    myind = np.where(myprof != 0.0)
    myprof = myprof[myind]; kpc_range = kpc_range[myind]
        
    if plot == True:
        import matplotlib.pyplot as plt
        plt.clf();  plt.figure(1); fig1,ax1 = plt.subplots()
        ax1.plot(kpc_range,-myprof);ax1.set_yscale("log")
        plt.axvline(uless_r[1],color="k", linestyle ="dashed")
        plt.axvline(uless_r[2],color="k", linestyle ="dashed")
        plt.xlabel("Radius (kpc)");  plt.title("Profile")
        filename = "Abell_2146_profile_checker.png"
        fullpath = os.path.join(hk.hk_outs.newpath,hk.hk_outs.prefilename+filename)
        plt.savefig(fullpath)

    return kpc_range,myprof

def get_SZ_factors(temp,dv,hk,efv,beta=0.0,betaz=0.0):

    int_factors = (efv.factors*u.cm)*u.kpc.to("cm"); int_factors = int_factors.value
    tSZ,kSZ = IS.get_sz_bp_conversions(temp,hk.hk_ins.instrument,
                                        array="2",inter=False,beta=beta,
                                        betaz=betaz,rel=True)
    return tSZ,kSZ,int_factors

def Ycyl_from_yProf(yprof,rads,r500,nrads=10000.0):

    radVals = np.arange(nrads)*r500/nrads
    drad    = r500/nrads
    yVals   = np.interp(radVals, rads,yprof)
    goodr   = (radVals < r500)
    
    Yint = np.sum(yVals[goodr] * 2.0 * np.pi * radVals[goodr] * drad)

    return Yint
