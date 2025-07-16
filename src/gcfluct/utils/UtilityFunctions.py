import numpy as np

def bin_two2Ds(independent,dependent,binsize=1,witherr=False,withcnt=False):

    flatin = independent.flatten()
    flatnt = dependent.flatten()
    inds = flatin.argsort()

    abscissa = flatin[inds]
    ordinate = flatnt[inds]

    nbins = int(np.ceil((np.max(abscissa) - np.min(abscissa))/binsize))
    abin  = np.zeros(nbins)
    obin  = np.zeros(nbins)
    oerr  = np.zeros(nbins)
    cnts  = np.zeros(nbins) 
    for i in range(nbins):
        blow = i*binsize
        gi = (abscissa >= blow)*(abscissa < blow+binsize)
        abin[i] = np.mean(abscissa[gi])
        obin[i] = np.mean(ordinate[gi])
        if witherr:
            oerr[i] = np.std(ordinate[gi]) / np.sqrt(np.sum(gi))
        if withcnt:
            cnts[i] = np.sum(gi)

    return abin,obin,oerr,cnts

def two2Ds_binned(independent,dependent,bins,witherr=False,withcnt=False):

    flatin = independent.flatten()
    flatnt = dependent.flatten()
    inds = flatin.argsort()

    abscissa = flatin[inds]
    ordinate = flatnt[inds]

    nbins = len(bins)-1
    abin  = np.zeros(nbins)
    obin  = np.zeros(nbins)
    oerr  = np.zeros(nbins)
    cnts  = np.zeros(nbins) 
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

    return abin,obin,oerr,cnts

def bin_log2Ds(independent,dependent,nbins=10,witherr=False,withcnt=False):

    flatin = independent.flatten()
    flatnt = dependent.flatten()
    inds   = flatin.argsort()

    abscissa = flatin[inds]
    ordinate = flatnt[inds]

    #nbins = int(np.ceil((np.max(abscissa) - np.min(abscissa))/binsize))
    agtz    = (abscissa > 0)
    lgkmin  = np.log10(np.min(abscissa[agtz])*2.5)
    lgkmax  = np.log10(np.max(abscissa))
    bins  = np.logspace(lgkmin,lgkmax,nbins+1)
    abin  = np.zeros(nbins)
    obin  = np.zeros(nbins)
    oerr  = np.zeros(nbins)
    cnts  = np.zeros(nbins) 
    for i,(blow,bhigh) in enumerate(zip(bins[:-1],bins[1:])):
        gi = (abscissa >= blow)*(abscissa < bhigh)
        abin[i]  = np.mean(abscissa[gi])
        omean    = np.mean(ordinate[gi])
        #abin[i]  = np.exp(np.mean(np.log(abscissa[gi])))
        #Ord,sOrd = plfit(ordinate[gi],ordinate[gi],abscissa[gi],abin[i])
        #Ord,sOrd = plfit(ordinate[gi],ordinate[gi]*0+omean,abscissa[gi],abin[i])
        #obin[i]  = Ord*1.0
        obin[i]   = omean
        #obin[i]  = np.exp(np.mean(np.log(ordinate[gi])))
        if witherr:
            oerr[i] = np.exp(np.std(np.log(ordinate[gi]))) / np.sqrt(np.sum(gi))
            #oerr[i]  = sOrd*1.0
        if withcnt:
            cnts[i] = np.sum(gi)

    #print(abin)
    #import pdb;pdb.set_trace()

    return abin,obin,oerr,cnts

def plfit(y,sy,x,xp=0.0):

    lny  = np.log(y)
    slny = sy/y
    lnx  = np.log(x)

    N    = len(y)
    #Del  = N * np.sum(lnx**2) - (np.sum(lnx)**2)
    #A    = (np.sum(lnx**2)*np.sum(lny) - np.sum(lnx)*np.sum(lnx*lny))/Del
    #B    = (N*np.sum(lnx*lny) - np.sum(lnx)*np.sum(lny))/Del
    #sA   = slny * np.sqrt(np.sum(lnx**2)/Del)
    #sB   = slny * np.sqrt(N/Del)

    wts = 1.0/slny**2
    w   = np.sum(wts)
    wxx = np.sum(wts*lnx**2)
    wy  = np.sum(wts*lny)
    wx  = np.sum(wts*lnx)
    wxy = np.sum(wts*lnx*lny)
    Del = w*wxx - wx**2
    A   = (wxx*wy - wx*wxy)/Del
    B   = (w*wxy - wx*wy)/Del
    sA  = np.sqrt( wxx / Del)
    sB  = np.sqrt( w   / Del)
    
    if xp > 0.0:
        lnxp = np.log(xp)

        lnS = A + B*lnxp
        S   = np.exp(lnS)
        #sS  = (sA + sB*lnxp)*S
        sS  = np.sqrt(sA**2 + (sB*lnxp)**2)*S
        #sS  = np.abs(sA - sB*lnxp)*S
        #print(sA,sB,lnxp)
        #import pdb;pdb.set_trace()
        return S,sS

    return A,B, sA, sB

def grid_profile(rads, profile, xymap, geoparams=[0,0,0,1,1,1,0,0],myscale=1.0,axis='z'):
    """
    Return a tuple of x- and y-coordinates.

    Parameters
    ----------
    rads : class:`numpy.ndarray`
       An array of radii (same units as xymap)
    profile : class:`numpy.ndarray`
       A radial profile of surface brightness.
    xymap : tuple(class:`numpy.ndarray`)
       A tuple of x- and y-coordinates
    geoparams : array-like
       [X_shift, Y_shift, Rotation, Ella*, Ellb*, Ellc*, Xi*, Opening Angle]
    myscale : float
       Generally best to leave as unity.
    axis : str
       Which axis are you projecting along.

    Returns
    -------
    mymap : class:`numpy.ndarray`
       An output map

    """

    ### Get new grid:
    arc2rad =  4.84813681109536e-06 # arcseconds to radians
    (x,y) = xymap
    x,y = rot_trans_grid(x,y,geoparams[0],geoparams[1],geoparams[2]) # 0.008 sec per call
    x,y = get_ell_rads(x,y,geoparams[3],geoparams[4])                # 0.001 sec per call
    theta = np.sqrt(x**2 + y**2)*arc2rad
    theta_min = rads[0]*2.0 # Maybe risky, but this is defined so that it is sorted.
    bi=(theta < theta_min);   theta[bi]=theta_min
    mymap  = np.interp(theta,rads,profile)
    
    if axis == 'x':
        xell = (x/(geoparams[3]*myscale))*arc2rad # x is initially presented in arcseconds
        modmap = geoparams[5]*(xell**2)**(geoparams[6]) # Consistent with model creation??? (26 July 2017)
    if axis == 'y':
        yell = (y/(geoparams[4]*myscale))*arc2rad # x is initially presented in arcseconds
        modmap = geoparams[5]*(yell**2)**(geoparams[6]) # Consistent with model creation??? (26 July 2017)
    if axis == 'z':
        modmap = geoparams[5]

    if modmap != 1:
        mymap *= modmap   # Very important to be precise here.
    if geoparams[7] > 0:
        angmap = np.arctan2(y,x)
        bi = (abs(angmap) > geoparams[7]/2.0)
        mymap[bi] = 0.0

    return mymap
