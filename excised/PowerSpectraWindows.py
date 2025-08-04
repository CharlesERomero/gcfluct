import numpy as np
from astropy.io import fits
import SelfSimilar_GalaxyClusters as SSGC
import MockMaps as MM
from scipy.special import gamma

def MakeWindowCut(rZkpc,myWindow,zcut1,zcut2,lStep=2,Optimize=True,useTukey=False):

    if useTukey:
        Optimize=False

    kpc_step         = np.median(np.diff(rZkpc))
    qz               = np.fft.fftfreq(len(rZkpc),kpc_step)             # 
    posfreqs         = np.where(qz > 0)                                #
    myqz             = qz[posfreqs] #/ (2.0*np.pi)                     #
    dqz              = myqz[1]-myqz[0]                                 #
    myBuff           = lStep*kpc_step * np.sqrt(np.log(2)) if Optimize else 0
    if zcut1 > 0:
        zdiff1        = zcut1 - np.abs(rZkpc) + myBuff
        phase         = np.pi*zdiff1/(2*lStep*kpc_step)
        tukey1        = 0.5* (1 + np.sin(phase))
        window1       = np.exp(-zdiff1**2 / (2*(lStep*kpc_step)**2))
        #gi            = (np.abs(rZkpc) <= zcut)
        bi            = (np.abs(rZkpc) > zcut1+myBuff)
        window1[bi]   = 1.0
        tzero         = (phase < -np.pi/2.0)
        tone          = (phase > np.pi/2.0)
        tukey1[tzero] = 0.0
        tukey1[tone]  = 1.0
    else:
        window1      = np.ones(rZkpc.shape)
        tukey1       = np.ones(rZkpc.shape)

    zdiff2           = np.abs(rZkpc) - zcut2 - myBuff
    phase2           = np.pi*zdiff2/(2*lStep*kpc_step)
    tukey2           = 0.5* (1 - np.sin(phase2))
    tzero2           = (phase2 > -np.pi/2.0)
    tone2            = (phase2 < np.pi/2.0)
    tukey2[tzero2]   = 0.0
    tukey2[tone2]    = 1.0
    window2          = np.exp(-zdiff2**2 / (2*lStep*kpc_step))
    gi               = (np.abs(rZkpc) < zcut2 - myBuff)
    window2[gi]      = 1.0
    CutWindow        = window1*window2   # I guess it's a double layer itself
    GoodTukey        = tukey1*tukey2      # Would sqrt() be better??
    DoublePane       = GoodTukey*myWindow if useTukey else CutWindow*myWindow 
    Wp               = np.fft.fft(DoublePane)*kpc_step
    CutTildeW        = np.abs(Wp**2)

    return CutTildeW

def SelectWindows(rad_kpc,Wlist,TildeWlist,N,myradius=500,zcut=None,rZkpc=None,lStep=2,Taper=None):

    mydiff    = np.abs(rad_kpc - myradius)
    mynpind   = np.where(mydiff == np.min(mydiff))            # Find the closest point to 500 kpc
    myind     = mynpind[0][0]
    #print(myind)
    #import pdb;pdb.set_trace()
    myWindow  = Wlist[myind]
    myTildeW  = TildeWlist[myind]
    myN       = N[myind]

    if zcut is None or rZkpc is None:
        CutTildeW        = myTildeW*1.0
    else:
        if len(zcut) == 1:
            zcut1 = 0
            zcut2 = zcut[0]
        else:
            zcut1 = zcut[0]
            zcut2 = zcut[1]
        if Taper is None:
            CutTildeW        = MakeWindowCut(rZkpc,myWindow,zcut1,zcut2,lStep=lStep)
        else:
            kpc_step         = np.median(np.diff(rZkpc))
            #qz               = np.fft.fftfreq(len(rZkpc),kpc_step)             # 
            #posfreqs         = np.where(qz > 0)                                #
            #myqz             = qz[posfreqs] #/ (2.0*np.pi)                     #
            #dqz              = myqz[1]-myqz[0]                                 #
            CutWindow        = myWindow*Taper
            Wp               = np.fft.fft(CutWindow)*kpc_step
            CutTildeW        = np.abs(Wp**2)

    return myradius,myWindow,myTildeW,myN,CutTildeW

def get_P3d(k,slope=0.0,kc=1e-3,p0=1e0,kdis=1e3,kfactor=1e2,eta_c=4.0,eta_d=1.5):

    keqz    = (k == 0)
    kgtz    = (k > 0)
    if np.sum(kgtz) == 0:
        print(k.shape)
        import pdb;pdb.set_trace()
    kmin    = np.min(k[kgtz])
    if np.sum(keqz) > 0:
        k[keqz] = kmin/kfactor
    ps0        = p0**2
    P3D        = ps0*k**(slope) * np.exp(-(kc/k)**eta_c) * np.exp(-(k/kdis)**eta_d)
    if np.sum(keqz) > 0:
        P3D[keqz]  = 0
    #print(ps0)
    #print(P3D)
    #import pdb;pdb.set_trace()

    return P3D

def theory_integrate(ktheta,kz,pWt,slope=0.0,kc=1e-3,p0=1e0,thresh=1e-3,eta_c=4.0,eta_d=1.5,kdis=1e3):

    #ps0    = p0**2
    nkz    = len(kz)
    nkt    = len(ktheta)
    kt2d   = np.outer(ktheta,np.ones(nkz))
    kz2d   = np.outer(np.ones(nkt),kz)
    k3d    = np.sqrt(kt2d**2 + kz2d**2)
    P3D    = get_P3d(k3d,slope=slope,kc=kc,p0=p0,kdis=kdis,eta_c=eta_c,eta_d=eta_d)
    #P3D    = ps0*k3d**(slope) * np.exp(-kc/k3d)
    dkzs   = np.diff(kz)
    if np.abs(np.log(np.median(dkzs)/dkzs[0])) < thresh:
        dkz = np.median(dkzs)
    else:
        print("It seems that kz is not linearly spaced!")
        import pdb;pdb.set_trace()
    #p2Wt   = np.interp(
    P3Wt   = np.outer(np.ones(nkt),pWt)
    
    Int3d2 = np.sum(P3Wt*P3D,axis=1)*dkz

    return ktheta,Int3d2

def theory_multipleP3D_integration(ktheta,kz,TWs_Sq,slopes,kcs,p0s,thresh=1e-3,corrF=1.0,eta_c=4.0,eta_d=1.5,kdis=None):

    #ps0    = p0**2
    nkz    = len(kz)
    nkt    = len(ktheta)
    kt2d   = np.outer(ktheta,np.ones(nkz))
    kz2d   = np.outer(np.ones(nkt),kz)
    k3d    = np.sqrt(kt2d**2 + kz2d**2)
    #P3D    = ps0*k3d**(slope) * np.exp(-kc/k3d)
    dkzs   = np.diff(kz)
    if np.abs(np.log(np.median(dkzs)/dkzs[0])) < thresh:
        dkz = np.median(dkzs)
    else:
        print("It seems that kz is not linearly spaced!")
        import pdb;pdb.set_trace()
    #p2Wt   = np.interp(
    if kdis is None:
        kdisarray = np.ones(len(kcs))*1e3
    else:
        kdisarray = kdis

    SqrtInteg  = 0
    for TW,slope,kc,p0,kd in zip(TWs_Sq,slopes,kcs,p0s,kdisarray):
        P3D    = get_P3d(k3d,slope=slope,kc=kc,p0=p0,eta_c=eta_c,eta_d=eta_d,kdis=kd)
        P3Wt   = np.outer(np.ones(nkt),TW)
        FT3D   = np.sqrt(P3D)
        FTWt   = np.sqrt(P3Wt)
        SqrtInteg += (FT3D*FTWt)
        #print(SqrtInteg.shape)
    #import pdb;pdb.set_trace()
    
    Integrand = SqrtInteg**2
    Int3d2 = np.sum(Integrand,axis=1)*dkz / corrF

    return ktheta,Int3d2

def get_FullWindow_SZ_kpc(M500,z,SmRads=None,Smy=None,rMax=5,c=1,npts=500,lgRmin=-0.5,arcminstep=0.01):

    radians,yProf,theta500 = SSGC.get_A10_yprof(z,M500,npts=npts,rMax=rMax,rBuff=rMax*2,lgRmin=lgRmin)
    if not (SmRads is None):
        unSmy   = yProf*1.0
        unSmRad = radians*1.0
        radians = SmRads*1.0
        yProf   = Smy*1.0
        
    radii      = np.arange(-theta500*10,theta500*10,arcminstep)  # in Arcminutes
    z_radians  = radii*np.pi/180/60                              # in radians
    D_a_kpc    = MM.get_d_ang(z).to("kpc")
    D_a_mpc    = D_a_kpc.to("Mpc").value
    kpc_step   = arcminstep*np.pi/180/60 *D_a_kpc.value
    qz         = np.fft.fftfreq(len(radii),c*kpc_step)             # 
    posfreqs   = np.where(qz > 0)                                #
    myqz       = qz[posfreqs] #/ (2.0*np.pi)                     #
    dqz        = myqz[1]-myqz[0]                                 #
    z_kpc      = z_radians*D_a_kpc.value                         # As it says -- in kpc!
    rad_kpc    = radians*D_a_kpc.value

    Pdl2y      = sigmaT*D_a_mpc*mpc2cm/mec2 * u.cm**3 / u.keV
    P2invkpc   = sigmaT*mpc2cm / (mec2*1000) * u.cm**3 / u.keV
    #sigTmec    = sigmaT/mec2

    Ns         = np.zeros(radians.shape)

    Wlist = []
    Plist = []
    for i,los in enumerate(radians):
        rads       = np.sqrt(z_radians**2 + los**2)
        r_kpc      = rads * D_a_kpc
        pprof      = gdi.a10_from_m500_z(M500, z,r_kpc).to("keV cm**-3") / c
        Window     = (pprof * P2invkpc).decompose().value / (yProf[i])  # Now in kpc**-1
        #Window     = (pprof * Pdl2y).decompose().value / (D_a_kpc.value *yProf[i])  # Now in kpc**-1
        Wp         = np.fft.fft(Window)*kpc_step*c
        pWt        = np.abs(Wp**2)
        Wlist.append(Window*1.0)
        Plist.append(pWt*1.0)
        NofTheta   = np.sum(pWt)*dqz
        Ns[i]      = NofTheta*1.0
        #print(Window)
        #import pdb;pdb.set_trace()

    return rad_kpc,z_kpc,Wlist,Plist,qz

def get_window_SZ_kpc(M500,z,SmRads=None,Smy=None,rMax=5,arcminstep = 0.01,c=1.0,npts=500,lgRmin=-0.5):

    radians,yProf,theta500 = SSGC.get_A10_yprof(z,M500,npts=npts,rMax=rMax,rBuff=rMax*2,lgRmin=lgRmin)
    if not (SmRads is None):
        unSmy   = yProf*1.0
        unSmRad = radians*1.0
        radians = SmRads*1.0
        yProf   = Smy*1.0

    radii      = np.arange(-theta500*10,theta500*10,arcminstep)  # in Arcminutes
    z_radians  = radii*np.pi/180/60                              # in radians
    D_a_kpc    = MM.get_d_ang(z).to("kpc")
    D_a_mpc    = D_a_kpc.to("Mpc").value
    kpc_step   = arcminstep*np.pi/180/60 *D_a_kpc.value
    qz         = np.fft.fftfreq(len(radii),c*kpc_step)             # 
    posfreqs   = np.where(qz > 0)                                #
    myqz       = qz[posfreqs] #/ (2.0*np.pi)                     #
    dqz        = myqz[1]-myqz[0]                                 # 

    Pdl2y      = sigmaT*D_a_mpc*mpc2cm/mec2 * u.cm**3 / u.keV
    P2invkpc   = sigmaT*mpc2cm / (mec2*1000) * u.cm**3 / u.keV
    #sigTmec    = sigmaT/mec2

    Ns         = np.zeros(radians.shape)
    
    for i,los in enumerate(radians):
        rads       = np.sqrt(z_radians**2 + los**2)
        r_kpc      = rads * D_a_kpc
        pprof      = gdi.a10_from_m500_z(M500, z,r_kpc).to("keV cm**-3") / c
        Window     = (pprof * P2invkpc).decompose().value / (yProf[i])  # Now in kpc**-1
        #Window     = (pprof * Pdl2y).decompose().value / (D_a_kpc.value *yProf[i])  # Now in kpc**-1
        Wp         = np.fft.fft(Window)*kpc_step*c
        pWt        = np.abs(Wp**2)
        NofTheta   = np.sum(pWt)*dqz
        Ns[i]      = NofTheta*1.0
        #print(Window)
        #import pdb;pdb.set_trace()

    return radians,Ns

def get_window_XR_kpc(M500,z,SmRads=None,Smy=None,SoftOnly=True):

    radians,sProf,theta500 = SSGC.get_XR_USBP(z,M500,npts=500,SoftOnly=SoftOnly)
    I_not,theta_c,beta,theta500 = SSGC.get_XR_UniversalBetaPars(z,M500,SoftOnly=SoftOnly)
    if not (SmRads is None):
        unSmy   = yProf*1.0
        unSmRad = radians*1.0
        radians = SmRads*1.0
        yProf   = Smy*1.0
        
    arcminstep = 0.01
    radii      = np.arange(-theta500*10,theta500*10,arcminstep)  # in Arcminutes
    z_radians  = radii*np.pi/180/60                              # in radians
    D_a_kpc    = MM.get_d_ang(z).to("kpc")
    D_a_mpc    = D_a_kpc.to("Mpc").value
    kpc_step   = arcminstep*np.pi/180/60 *D_a_kpc.value
    qz         = np.fft.fftfreq(len(radii),kpc_step)             # 
    posfreqs   = np.where(qz > 0)                                #
    myqz       = qz[posfreqs] #/ (2.0*np.pi)                     #
    dqz        = myqz[1]-myqz[0]                                 # 

    #beta       = 2.0/3.0                                         # Forever and ever
    kpcperam   = D_a_kpc*np.pi/(180*60)                          # kpc per arcminute
    betanorm   = gamma(3*beta-0.5)*gamma(0.5) / gamma(3*beta) 
    theta_kpc  = (theta_c * kpcperam).to("kpc")

    Ns         = np.zeros(radians.shape)
    
    for i,los in enumerate(radians):
        rads       = np.sqrt(z_radians**2 + los**2)
        r_kpc      = rads * D_a_kpc
        t_kpc      = los  * D_a_kpc
        r3d_scaled = (r_kpc/theta_kpc).decompose().value
        r2d_scaled = (t_kpc/theta_kpc).decompose().value
        emmisivity = (1.0 + (r3d_scaled)**2)**(-3.0*beta)          # 
        surface_b  = (1.0 + (r2d_scaled)**2)**(0.5 -3.0*beta)      # Recalculating...kind of
        Window     = emmisivity / (betanorm*beta*theta_kpc.value*surface_b)   # inverse kpc
        Wp         = np.fft.fft(Window)*kpc_step
        pWt        = np.abs(Wp**2)
        NofTheta   = np.sum(pWt)*dqz
        Ns[i]      = NofTheta*1.0
        #print(Window)
        #import pdb;pdb.set_trace()

    return radians,Ns

def get_Neff(radians,Ns,Redges):

    Nannuli = len(Redges)-1
    N_eff   = np.zeros(Nannuli)
    N_var   = np.zeros(Nannuli)
    N_std   = np.zeros(Nannuli)
    Area    = np.hstack(([0.0],radians[1:]**2 - radians[:-1]**2))

    for i in range(Nannuli):
        gi        = (radians >= Redges[i])*(radians < Redges[i+1])
        gr        = radians[gi]
        gN        = np.sqrt(Ns[gi])
        #nume      = 
        N_eff[i]  = ( np.sum(gN*Area[gi])/np.sum(Area[gi]) )**2
        N_var[i]  = np.sum( (gN - N_eff[i])**2 * gr) / np.sum(gr)
    N_std  = np.sqrt(N_var)

    return N_eff,N_std
def get_Windows_M5_z(M5=6.0,z=0.3,npts=2000,lgRmin=-2,rMax=10):

    M500 = M5*1e14 * u.M_sun
    r500, p500                      = gdi.R500_P500_from_M500_z(M500,z)
    r500_kpc                        = r500.to("kpc").value
    rW,NofTheta                     = get_window_SZ_kpc(M500,z,rMax=rMax,npts=2000,lgRmin=-2)         # N has units of kpc
    rWkpc,rZkpc,Window,TildeWsq,kz  = get_FullWindow_SZ_kpc(M500,z,rMax=rMax,npts=2000,lgRmin=-2)

    return rW,NofTheta,rWkpc,rZkpc,Window,TildeWsq,kz,r500_kpc

def recalculate_Neffs(M5=6.0,z=0.3,npts=2000,lgRmin=-2,rMax=10):

    rW,Ns,rWkpc,rZkpc,Window,TildeWsq,kz,r500_kpc = get_Windows_M5_z(M5=M5,z=z,npts=npts,lgRmin=lgRmin,rMax=rMax)

    x500   = (288/1024)*5
    rBox   = r500_kpc*x500
    Nlt500 = (rWkpc < r500_kpc)
    GoodNs = Ns[Nlt500]
    GoodRs = rW[Nlt500]
    Rstack = np.hstack([0,GoodRs])
    Areas  = Rstack[1:]**2 - Rstack[:-1]**2
    Nwtd   = np.sum(GoodNs*Areas)/np.sum(Areas)

    BoxFrac = np.ones(npts)
    BoxRgt   = (rWkpc >= rBox)
    Thetas   = np.arccos(rBox/rWkpc[BoxRgt])
    BoxFrac[BoxRgt]  = (np.pi - 4*Thetas)/np.pi
    BadBox   = (BoxFrac < 0)
    BoxFrac[BadBox] = 0
    FullRads = np.hstack([0,rWkpc])
    FullArea = (FullRads[1:]**2 - FullRads[:-1]**2)*BoxFrac
    Nbox     = np.sum(Ns*FullArea)/np.sum(FullArea)

    print(Nwtd,Nbox)

    return Nwtd,Nbox
