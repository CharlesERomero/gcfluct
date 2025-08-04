import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3, Tcmb0=2.725)
import astropy.units as u
from scipy.interpolate import interp1d
from datetime import date
from tqdm import tqdm
##########################################################
import UtilityFunctions as UF
import image_filtering as imf
import analytic_integrations as ai
import numerical_integration as ni      # For the gNFW profile

##########################################################
#import quick_map_fitting as qmf
#from matplotlib.collections import LineCollection
#import Arevalo_Wrapper as AW
#import pickle
#import MultiGaussBias as MGB
#import cProfile, pstats, io
#from pstats import SortKey
#import get_data_info as gdi
#from scipy.special import gamma

from importlib import reload
#qmf=reload(qmf)

today = date.today()
tdstr = today.strftime("%d%b%Y")

sigmaT_m2 = 6.652e-29 # Thomson cross-section (m**2)
sigmaT    = sigmaT_m2 * 1e4 #                 (cm**2)
mec2      = 511.0     # electron mass *c^2    (keV)
s2f       = 2.0*np.sqrt(2*np.log(2))

##Some conversions + cosmological numbers
mpc2cm    = 3.08568e24
kpc2cm    = 3.08568e21
pc2cm     = 3.0857e18

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di


def get_d_ang(z):

    d_ang = cosmo.comoving_distance(z) / (1.0 + z)

    return d_ang

def make_r2d(nx,ny,pixsize):

    x1   = (np.arange(nx)-nx/2)*pixsize
    y1   = (np.arange(ny)-ny/2)*pixsize
    x    = np.outer(x1,np.ones(ny))
    y    = np.outer(np.ones(nx),y1)
     
    rmap = np.sqrt(x**2 + y**2)

    return rmap

def make_xymap(xsize,ysize,pixsize,cx=None,cy=None,ForceInteger=False):

    xpix = int(np.round((xsize*60)/pixsize))
    ypix = int(np.round((ysize*60)/pixsize))
    if cx is None:
        cx   = xpix//2 if ForceInteger else xpix/2.0
    if cy is None:
        cy   = ypix//2 if ForceInteger else ypix/2.0
    x1   = (np.arange(xpix)-cx)*pixsize
    y1   = (np.arange(ypix)-cy)*pixsize
    x    = np.outer(x1,np.ones(ypix))
    y    = np.outer(np.ones(xpix),y1)
    
    return x,y
    
def make_rmap(xymap):

    x,y  = xymap
    rmap = np.sqrt(x**2 + y**2)

    return rmap

def make_M2_noise(xymap,size,hours,inArcseconds=True,inComptony=True,pixsize=2.0):

    fwhm                  = 10 if inArcseconds else 1.0/6.0
    pixfwhm               = fwhm/pixsize
    myconv                = 60.0 if inArcseconds else 1.0
    rmap                  = make_rmap(xymap) /myconv  # in arcminutes
    rmsroothourmap        = get_radrms(rmap,size,inComptony=inComptony)
    rmsmap                = rmsroothourmap / np.sqrt(hours)
    rgtFOV                = (rmap > size+2.4)
    rgtComp               = (rmap > size+1.1)
    
    rmsmap[rgtComp]      *= (rmap[rgtComp]/(size+1.1))**2
    noiseReal             = np.random.normal(size=rmap.shape)*rmsmap
    noiseReal[rgtFOV]     = 0
    rmsmap[rgtFOV]        = -1

    rmsmap               *= pixfwhm
    noiseReal            *= pixfwhm

    return noiseReal,rmsmap
    
def make_A10_map(xymap,z,M500,inArcseconds=True,rMax=5,yPnz=True):

    #pr = cProfile.Profile()
    #pr.enable()
    myconv                = 3600.0 if inArcseconds else 60.0
    rmap                  = make_rmap(xymap) * np.pi/(180*myconv)  # in radians
    #rBuff = rMax if rMax > 15 else 15.0
    rBuff = rMax*2.0
    radians,yProf,t500    = get_A10_yprof(z,M500,rMax=rMax,rBuff=rBuff)
    rshape                = rmap.shape
    rltrmin               = (rmap < np.min(radians))
    rmap[rltrmin]         = np.min(radians)
    r1d                   = rmap.reshape((np.prod(rshape)))
    fint                  = interp1d(radians,yProf, bounds_error = False, fill_value = 0)
    Profile1D             = fint(r1d)
    yMap                  = Profile1D.reshape(rshape)
    #pr.disable()
    #s = io.StringIO()
    #sortby = SortKey.CUMULATIVE
    #ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    #ps.print_stats()
    #print(s.getvalue())
    #import pdb;pdb.set_trace()

    return yMap


########################################################################################


def reinterpolate(z,M500,Buffrad,yProf,rMax=5,npts=500):

    D_a_mpc          = get_d_ang(z).to("Mpc").value
    Pdl2y            = sigmaT*D_a_mpc*mpc2cm/mec2 * u.cm**3 / u.keV
    r500, p500       = gdi.R500_P500_from_M500_z(M500,z)
    log5r500         = np.log10((rMax*r500/u.kpc).decompose().value)
    rads             = np.logspace(-0.5,log5r500,npts) * u.kpc # 1 Mpc ~ R500, usually
    radians          = (rads / (D_a_mpc*u.Mpc) ).decompose().value    

    newProf          = np.exp(np.interp(np.log(radians),np.log(Buffrad),np.log(yProf)))

    return radians,newProf

def get_A10_yprof(z,M500,npts=500,rMax=5,rBuff=15,lgRmin=-0.5):

    myMax = rBuff if rBuff > rMax else rMax
    radians,unitless_profile,theta500 = get_A10_ulPprof(z,M500,npts=npts,rMax=myMax,lgRmin=lgRmin)
    yProf            = ni.int_profile(radians, unitless_profile,radians)
    if rBuff > rMax:
        inradians         = radians*1.0
        outradians,yProf  = reinterpolate(z,M500,inradians,yProf,rMax=rMax,npts=npts) 
        radians           = outradians*1.0

    return radians,yProf,theta500

def Nelson_alpha(rads,r200m,A=0.452,B=0.841,gNT=1.628):
    
    """
    A    = 0.452 +/- 0.001
    B    = 0.841 +/- 0.0008
    gNT  = 1.628 +/- 0.019
    
    """
    
    myexp = (rads/(B*r200m) ).decompose().value
    g     = (1.0 + np.exp(-myexp**gNT))
    alpha = 1 - A*g
    
    return alpha

def Battaglia_alpha(rads,M200,R500,z,nNT=0.8,nM=0.2,anot=0.18,beta=0.5):
    
    """
    anot = 0.18 +/- 0.06
    nNT  = 0.8  +/- 0.25
    
    """
    
    alpha_z = anot*(1+z)**beta
    Rscale  = (rads / R500).decompose().value
    Mscale  = (M200 / (3e14*u.M_sun)).decompose().value 
    
    alpha   = alpha_z * (Rscale**nNT)*(Mscale**nM)
    
    return alpha

def R500c_to_R200m(R500c,z):
    
    R200c = 1.5*R500c
    OmL    = 1.0 - cosmo.Om0
    cscale = OmL + (1+z)**3 * cosmo.Om0
    mscale = (1+z)**3 * cosmo.Om0
    mVc_zero = (cosmo.Om0)**(-1.0/3.0)
    m_vs_c = (mscale/cscale)**(-1.0/3.0)
    
    R5_2zero = 0.37
    
    scaling = (1.0/R5_2zero)*(m_vs_c / mVc_zero)
    R200m   = R500c * scaling
    
    #print(1.0/scaling,scaling)
    return R200m.to("Mpc")
    
def R500c_to_R200c(R500c,z):
    
    R200c = 1.53*R500c    # 1.53 +/- 0.04 ? .. no dependence on z
    
    return R200c.to("Mpc")

def R500c_to_M200c(R500c,z):
    
    rho_crit = cosmo.critical_density(z)
    R200c    = R500c_to_R200c(R500c,z)
    print("Checks: ",R500c,R200c)
    M200c    = (4.0/3)*np.pi * R200c**3 * rho_crit * 200.0
    M200c    = M200c.to("M_sun")
    
    return M200c

def pNT_from_P_alpha(rads,P,alpha):
    
    pNT = alpha*P / (1 - alpha)
    
    return pNT
    
def Mach_from_alpha(alpha,Adiabatic=5.0/3.0):
    
    pNT_pTh = alpha / (1 - alpha)
    Mach3d  = np.sqrt( pNT_pTh / (Adiabatic/3.0) )
    
    return Mach3d
    
def lnPnt_lnPth(rads,Machs,pTh):
    
    lnMlnr = np.diff(np.log(Machs)) / np.diff(np.log(rads))
    lnPlnr = np.diff(np.log(pTh)) / np.diff(np.log(rads))
    
    dlnPnt_dlnPth = 1 + 2*lnMlnr/lnPlnr
    
    return dlnPnt_dlnPth

def HydroBias(Machs,dlnPnt_dlnPth,Adiabatic=5.0/3.0):
    
    T1 = (-Adiabatic/3)*Machs[1:]**2 * dlnPnt_dlnPth
    T2 = 1.0 + (Adiabatic/3)*Machs[1:]**2 * dlnPnt_dlnPth
    
    b_Mach = T1/T2
    
    return b_Mach

def get_UPP_beamSmoothed_yProf(z,M500,fwhm=10.0,rMax=5,npts=500):

    radians,yProf_uns,t500         = get_A10_yprof(z,M500,rMax=rMax,rBuff=rMax*2,npts=npts)
    BeamUPP,xymap,pixs,t500        = make_UPP_beam_smoothed(M500,z,fwhm=fwhm,Nr500=3.4,npts=npts)
    rmap                           = make_rmap(xymap)
    rbin,ybin,yerr,ycnts           = UF.bin_two2Ds(rmap,BeamUPP,binsize=pixs*2.0)
    fint                           = interp1d(rbin,ybin, bounds_error = False, fill_value = "extrapolate")
    radarcsec                      = radians * 3600 * 180/np.pi
    Profile1D                      = fint(radarcsec)

    return radians,Profile1D,t500

def make_UPP_beam_smoothed(M500,z,fwhm=10.0,Nr500=1.5,rMax=5,npts=500):
    
    #pprof            = gdi.a10_from_m500_z(M500, z,rads).to("keV cm**-3")
    #R500c, p500      = gdi.R500_P500_from_M500_z(M500,z)
    radians,unitless_profile,theta500 = get_A10_yprof(z,M500,rMax=rMax,rBuff=rMax*2,npts=npts)

    xsize = np.round(theta500*2*Nr500)
    ysize = np.round(theta500*2*Nr500)

    pixsize  = fwhm/5.0
    pixfwhm  = fwhm/pixsize
    xymap    = make_xymap(xsize,ysize,pixsize)
    A10map   = make_A10_map(xymap,z,M500,inArcseconds=True,rMax=rMax)         # in Compton y
    BeamA10  = imf.fourier_filtering_2d(A10map,"gauss",pixfwhm)     # in Compton y

    return BeamA10,xymap,pixsize,theta500

def get_Annular_bias_at_linj(lc,slope=3.0,npytag="_InstrumentClusterAgnostic_200pass",outdir=None,CompMask=False,
                             matchThresh=1e-2):

    Pk_bias,myslopes,rings,kbin = get_Annular_biases(npytag=npytag,outdir=outdir,CompMask=CompMask)

    si                          = (np.abs(myslopes-slope) <= matchThresh)
    scaledK                     = np.abs(np.log(kbin*lc*100.0))
    ki                          = (scaledK == np.min(scaledK))

    RingBiases                  = Pk_bias[si,:,ki]

    #print(si,ki,scaledK)
    #import pdb;pdb.set_trace()
    
    return RingBiases.flatten()


def calculated_P2D(image,pixsize,corrN=False,doPlot=False,OutDir=""):

    nx,ny          = image.shape
    k, dkx, dky    = get_freqarr_2d(nx, ny, pixsize, pixsize)
    imgfft         = np.fft.fft2(image)*pixsize
    imgps          = np.abs(imgfft**2) / (nx*ny) if corrN else np.abs(imgfft**2)
    np2            = int(np.round(np.log(nx)/np.log(2))*2.0)
    kb,pb,pe,pcnt  = UF.bin_log2Ds(k,imgps,nbins=np2,witherr=True,withcnt=True)

    if doPlot:
        VF_fig = plt.figure(37,figsize=(5,4),dpi=200)
        VF_fig.clf()
        VF_ax  = VF_fig.add_subplot(111)
        VF_ax.scatter(k,imgps,s=0.5)
        VF_ax.plot(kb,pb,"--k")
        VF_ax.set_xscale("log")
        VF_ax.set_yscale("log")
        VF_fig.tight_layout()
        VF_fig.savefig(OutDir+"QuickPlot_calculatedP2D.png")
        #print(pixsize)
        #import pdb;pdb.set_trace()

    return kb,pb
    
def binned_P2D(nx,ny,kc,p0,slope,pixsize,corrN=False,kdis=1e3,eta_c=4.0,eta_d=1.5):

    k, dkx, dky    = get_freqarr_2d(nx, ny, pixsize, pixsize)
    imgps          = get_P3d(k,slope=slope,kc=kc,p0=p0,kdis=kdis,eta_c=eta_c,eta_d=eta_d)
    np2            = int(np.round(np.log(nx)/np.log(2))*2.0)
    kb,pb,pe,pcnt  = UF.bin_log2Ds(k,imgps,nbins=np2,witherr=True,withcnt=True)

    return kb,pb

def rebinned_P2D(nx,ny,kbin,psbin,pixsize,corrN=False,kdis=1e3):
    

    k, dkx, dky    = get_freqarr_2d(nx, ny, pixsize, pixsize)
    kflat          = k.flatten()
    gki            = (kflat > 0)
    gk             = kflat[gki]

    psout          = np.exp(np.interp(np.log(gk),np.log(kbin),np.log(psbin)))
    psarr          = kflat*0
    psarr[gki]     = psout
    imgps          = psarr.reshape(k.shape) 
    np2            = int(np.round(np.log(nx)/np.log(2))*2.0)
    kb,pb,pe,pcnt  = UF.bin_log2Ds(k,imgps,nbins=np2,witherr=True,withcnt=True)

    return kb,pb
    
def check_nz_dz_importance(kbin,nz,kpc_step,z_kpc,myWindow,slope,kc,p0,c=1):

    Cz         = nz/2.0
    Boxed_z    = (np.arange(nz) - Cz)*kpc_step         # pixsize is in kpc!
    BoxWindow  = np.interp(Boxed_z,z_kpc,myWindow)
    qz         = np.fft.fftfreq(nz,c*kpc_step)             # 
    posfreqs   = np.where(qz > 0)                                #
    myqz       = qz[posfreqs] #/ (2.0*np.pi)                     #
    dqz        = myqz[1]-myqz[0]                                 #
    hr_kpc_stp = np.abs(np.median(np.diff(z_kpc)))
    hrqz       = np.fft.fftfreq(len(z_kpc),c*hr_kpc_stp)             # 
    Wp         = np.fft.fft(myWindow)*hr_kpc_stp*c
    hrTW       = np.abs(Wp**2)
    BoxWp      = np.fft.fft(BoxWindow)*kpc_step*c
    BoxpWt     = np.abs(BoxWp**2)

    k1,hrP     = theory_integrate(kbin,hrqz,hrTW,slope=slope,kc=kc,p0=p0)
    k2,BoxP    = theory_integrate(kbin,qz,BoxpWt,slope=slope,kc=kc,p0=p0)

    ResBias    = BoxP/hrP

    print(ResBias)
    import pdb;pdb.set_trace()

    return ResBias

def integrate_P3D_Window(box,pixsize,rad_kpc,z_kpc,myWindow):

    #mydiff    = np.abs(rad_kpc - myradius)
    #mynpind   = np.where(mydiff == np.min(mydiff))            # Find the closest point to 500 kpc
    #myind     = mynpind[0]
    #myWindow  = Wlist[myind]
    BoxShape  = box.shape
    Cz        = BoxShape[2]/2.0
    Boxed_z   = (np.arange(BoxShape[2]) - Cz)*pixsize         # pixsize is in kpc!
    BoxWindow = np.interp(Boxed_z,z_kpc,myWindow)

    OneBox    = np.outer(np.ones(BoxShape[1]),BoxWindow)
    TwoBox    = np.expand_dims(OneBox,0)
    #ky3 = np.expand_dims(ky2,2)
    #gkx3= np.repeat(kx3,nz,axis=2)
    Windowed  = np.repeat(TwoBox,BoxShape[0],axis=0)
    #Windowed  = np.outer(np.ones(BoxShape[0].T,np.ones(BoxShape[1]).T),BoxWindow)
    dz_in     = z_kpc[1] - z_kpc[0]                           # I have specified a linear griding
    #print("Steps: ",dz_in,pixsize)
    #print(Windowed[0,0,::32])
    Integ     = np.sum(Windowed*box,axis=2)*pixsize

    #print(Cz,pixsize)
    #import pdb;pdb.set_trace()

    return Integ
    
def get_3D_rads(nx, ny, nz, pixsize):

    cx               = nx/2
    cy               = ny/2
    cz               = nz/2
    rads             = get_xyzmap(nx, ny, nz, cx,cy,cz, pixsize, pixsize, pixsize,retR=True)
    bi               = (rads < pixsize/10.0)
    rads[bi]         = pixsize/10.0

    return rads

def make_3D_UPP(z,M500,nx, ny, nz, cx,cy,cz, psx, psy, psz):

    rads             = get_xyzmap(nx, ny, nz, cx,cy,cz, psx, psy, psz,retR=True)
    bi               = (rads < psx/10.0)
    #rads[bi]         = psx/10.0
    rads[bi]         = psx/2.0
    pprof            = gdi.a10_from_m500_z(M500, z,rads*u.kpc) # should return array of same dims
    #import pdb;pdb.set_trace()

    return pprof.to("keV cm**-3").value

def make_3D_products(z,M500,nx=1024,ny=1024,nz=1024,pixsize=1.0,slope=0.0,kc=1.0e-3,p0=1.0e0,
                verbose=False,phase=None,retPhase=False,kin=None):

    cx          = nx/2
    cy          = ny/2
    cz          = nz/2

    A10Pressure = make_3D_UPP(z,M500,nx,ny,nz,cx,cy,cz,pixsize,pixsize,pixsize)
    Pthermal    = A10Pressure    # P * (1 + delta P / P)
    Pint        = np.sum(Pthermal,axis=2)*pixsize # (pixsize = kpc_step)
    Pkpc2y      = sigmaT*kpc2cm/mec2
    Comptony    = Pkpc2y * Pint
    Cy3         = np.expand_dims(Comptony,2)
    #################################################
    Comptony3D  = np.repeat(Cy3,nz,axis=2)

    W           = Pthermal / (Comptony3D*Pkpc2y)
    R3D         = get_3D_rads(nx, ny, nz, pixsize)
    #gi          = np.where(A10Pressure == np.max(A10Pressure))
    #print(gi)
    #import pdb;pdb.set_trace()

    return W,R3D


def make_yfrom_3DP(z,M500,nx=1024,ny=1024,nz=1024,pixsize=1.0,slope=0.0,kc=1.0e-3,p0=1.0e0,
                   verbose=False,phase=None,retPhase=False,kin=None,box=None):

    if box is None:
        if verbose:
            print("Remaking box")
        box, k, newps = make_P3Dbox(nx=nx,ny=ny,nz=nz,pixsize=pixsize,slope=slope,phase=phase,
                                    retPhase=False,verbose=verbose,kc=kc,p0=p0)
    else:
        k = 0
        newps=0
        if verbose:
            print("box supplied")

    cx          = nx/2
    cy          = ny/2
    cz          = nz/2

    A10Pressure = make_3D_UPP(z,M500,nx,ny,nz,cx,cy,cz,pixsize,pixsize,pixsize)
    Pthermal    = A10Pressure*(1.0 + box)    # P * (1 + delta P / P)
    Pint        = np.sum(Pthermal,axis=2)*pixsize # (pixsize = kpc_step)
    Pkpc2y      = sigmaT*kpc2cm/mec2
    Comptony    = Pkpc2y * Pint

    A10_raw     = np.sum(A10Pressure,axis=2)*pixsize # (pixsize = kpc_step)
    A10_y       = Pkpc2y * A10_raw

    #gi          = np.where(A10Pressure == np.max(A10Pressure))
    #print(gi)
    #import pdb;pdb.set_trace()
    print("Done integrating along the line of sight")

    return Comptony, box, k, newps,A10_y

def make_P3Dbox(nx=1024,ny=1024,nz=1024,cx=512,cy=512,pixsize=1.0,slope=-11.0/3.0,kc=1.0e-3,p0=1.0e-2,
                verbose=False,phase=None,retPhase=False,kin=None,CorrN=True,eta_c=4.0,eta_d=1.5,kdis=1e3):

    k      = get_freqarr_3d(nx, ny,nz, pixsize, pixsize,pixsize)
    kmax   = 1.0/pixsize

    P3D    = get_P3d(k,slope=slope,kc=kc,p0=p0,kdis=kdis,eta_c=eta_c,eta_d=eta_d)
    #amp    = np.sqrt(P3D*nx*ny*nz)
    AmpFactor = (nx*ny*nz)/(pixsize**3)
    if verbose:
        print("Size(x1e6): ",(nx*ny*nz)/1e6,"  Pixsize: ",pixsize,"  AmpFactor: ",AmpFactor)
    amp    = np.sqrt(P3D*AmpFactor)
    #amp = p0*k**(slope/2) * np.exp(-kc/2/k)
    #vol = int_pk(p0,slope,kc,1.0/pixsize)
    if phase is None:
        phase  = np.random.uniform(size=(nx,ny,nz))*2*np.pi
    ampC   = 1.0
    newfft = amp* np.exp(1j*phase) / np.sqrt(ampC)
    bi     = np.isnan(newfft)

    numbi  = np.sum(bi)
    #print("Number of bad (NAN) cells: ",numbi)
    #if numbi > 1e3:
    #    import pdb;pdb.set_trace()
    bki    = (k == 0)
    if np.sum(bi) > 0:
        newfft[bi] = 0.0
        P3D[bi] = 0.0
    #P3D[bki] = 0.0
    #newfft[0,0] = 0.0
    newps = np.abs(newfft*np.conjugate(newfft))
    #PTsum = np.sum(newps/pixsize**2)/(nx*ny)
    #if verbose:
    #    print("PTsum: ",PTsum)
    box = np.real(np.fft.ifftn(newfft)) # As of 01 Sep 2023, this seems correct?
##### Sometime earlier this seemed like the necessary scaling:
    #box = np.real(np.fft.ifftn(newfft)/np.sqrt(pixsize))  
    #img = np.real(np.fft.ifft2(newfft/pixsize**2))
    #img = np.abs(np.fft.ifft2(newfft/pixsize**2))
    box *= np.sqrt(2.0)
    varsum   = np.sum(box**2)

    P3Dsum   = np.sum(P3D)/AmpFactor
    if verbose:
        print("VARsum: ",varsum/(nx*ny*nz)," versus: ",P3Dsum/(nx*ny*nz))
        if varsum == 0:
            import pdb;pdb.set_trace()

    dcheck = np.any(np.isnan(newps))
    if dcheck:
        print("Oh no -- NANs!")
        import pdb;pdb.set_trace()

    if retPhase:
        return box, k, newps, phase
    else:
        return box, k, newps

def get_freqarr_3d(nx, ny, nz, psx, psy, psz):
    """
       Compute frequency array for 3D FFT transform

       Parameters
       ----------
       nx : integer
            number of samples in the x direction
       ny : integer
            number of samples in the y direction
       nz : integer
            number of samples in the y direction
       psx: float
            map pixel size in the x direction
       psy: float
            map pixel size in the y direction
       psz: float
            map pixel size in the y direction
       Returns
       -------
       k : float 3D numpy array
           frequency vector
    """
    #kx =  np.outer( np.outer(np.fft.fftfreq(nx),np.ones(ny))/psx, np.ones(nz))
    ky2 =  np.outer(np.ones(nx),np.fft.fftfreq(ny))/psy
    kz2 =  np.outer(np.ones(ny),np.fft.fftfreq(nz))/psz
    kx2 =  np.outer(np.fft.fftfreq(nx),np.ones(ny))/psx
    #kx3 = np.broadcast_to(kx2,(nx,ny,1))
    kx3 = np.expand_dims(kx2,2)
    ky3 = np.expand_dims(ky2,2)
    kz3 = np.expand_dims(kz2,0)
    gkx3= np.repeat(kx3,nz,axis=2)
    gky3= np.repeat(ky3,nz,axis=2)
    gkz3= np.repeat(kz3,nx,axis=0)
    #print(gkx3.shape,gky3.shape,gkz3.shape)
    #import pdb;pdb.set_trace()
    #print(gkx3[:5,:5,0])
    #print(gky3[:5,:5,0])
    #print(gkz3[:5,:5,0])
    #import pdb;pdb.set_trace()
    k  =  np.sqrt(gkx3**2 + gky3**2 + gkz3**2)
    #print(k[:3,:3,:3])

    return k

def get_freqarr_2d(nx, ny, psx, psy):
    """
       Compute frequency array for 2D FFT transform

       Parameters
       ----------
       nx : integer
            number of samples in the x direction
       ny : integer
            number of samples in the y direction
       psx: integer
            map pixel size in the x direction
       psy: integer
            map pixel size in the y direction

       Returns
       -------
       k : float 2D numpy array
           frequency vector
    """
    kx =  np.outer(np.fft.fftfreq(nx),np.zeros(ny).T+1.0)/psx
    ky =  np.outer(np.zeros(nx).T+1.0,np.fft.fftfreq(ny))/psy
    dkx = kx[1:][0]-kx[0:-1][0]
    dky = ky[0][1:]-ky[0][0:-1]
    k  =  np.sqrt(kx*kx + ky*ky)
    #print('dkx, dky:', dkx[0], dky[0])
    return k, dkx[0], dky[0]

def get_xyzmap(nx, ny, nz, cx,cy,cz, psx, psy, psz,retR=False):

    #xpix = int(np.round((xsize*60)/pixsize))
    #ypix = int(np.round((ysize*60)/pixsize))
    #x1   = (np.arange(xpix)-xpix//2)*pixsize
    #y1   = (np.arange(ypix)-ypix//2)*pixsize
    #x    = np.outer(x1,np.ones(ypix))
    #y    = np.outer(np.ones(xpix),y1)


    ky2  = np.outer(np.ones(nx),np.arange(ny)-cy)*psy
    kz2  = np.outer(np.ones(ny),np.arange(nz)-cz)*psz
    kx2  = np.outer(np.arange(nx)-cx,np.ones(ny))*psx
    kx3  = np.expand_dims(kx2,2)
    ky3  = np.expand_dims(ky2,2)
    kz3  = np.expand_dims(kz2,0)
    #################################################
    gx3  = np.repeat(kx3,nz,axis=2)
    gy3  = np.repeat(ky3,nz,axis=2)
    gz3  = np.repeat(kz3,nx,axis=0)

    r3d  =  np.sqrt(gx3**2 + gy3**2 + gz3**2)

    #gi   = np.where(r3d == np.min(r3d))
    #print(gi)
    #import pdb;pdb.set_trace()

    if retR:
        return r3d
    else:
        return gx3,gy3,gz3

    
def get_Beam_PS(karr,fwhm):

    sigma    = fwhm/s2f
    fourfilt = np.exp(-2.0*karr**2 * sigma**2 * np.pi**2)
    PS       = fourfilt**2

    return PS

def make_image(kbin,psbin,nx=1024,ny=1024,cx=512,cy=512,pixsize=1.0,verbose=False):

    k,dkx,dky   = get_freqarr_2d(nx, ny, pixsize, pixsize)
    kflat       = k.flatten()
    gki         = (kflat > 0)
    gk          = kflat[gki]

    psout       = np.exp(np.interp(np.log(gk),np.log(kbin),np.log(psbin)))
    psarr       = kflat*0
    psarr[gki]  = psout
    ps2d        = psarr.reshape(k.shape) * nx*ny

    phase       = np.random.uniform(size=(nx,ny))*2*np.pi
    newfft      = np.sqrt(ps2d) * np.exp(1j*phase)
    newps       = np.abs(newfft*np.conjugate(newfft))
    PTsum       = np.sum(newps/pixsize**2)/(nx*ny)
    if verbose:
        print("PTsum: ",PTsum)
    img = np.real(np.fft.ifft2(newfft/pixsize))
    img *= np.sqrt(2.0)
    varsum = np.sum(img**2)
    if verbose:
        print("VARsum: ",varsum)
        #import pdb;pdb.set_trace()

    return img

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

def make_template_hdul(nx,ny,cntr,pixsize,cx=None,cy=None):
    """
    Return a map of RMS sensitivites based on input set of scans.

    Parameters
    ----------
    nx : int
       Number of pixels along axis 0
    ny : int
       Number of pixels along axis 1
    cntr : array_like
       Two-element object specifying the RA and Dec of the center.
    pixsize : float
       Pixel size, in arcseconds
    cx : float
       The pixel center along axis 0
    cy : float
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
    ### Let's make some WCS information as if we made 1 arcminute pixels about Coma's center:
    w = WCS(naxis=2)
    w.wcs.crpix = [cx,cy]
    w.wcs.cdelt = np.array([-pixsize/3600.0,pixsize/3600.0])
    w.wcs.crval = [cntr[0], cntr[1]]
    #w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    hdr = w.to_header()

    zero_img    = np.zeros((nx,ny))
    Phdu        = fits.PrimaryHDU(zero_img,header=hdr)
    TempHdu     = fits.HDUList([Phdu])

    return TempHdu

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

def rot_trans_grid(x,y,xs,ys,rot_rad):
    """   
    Shift and rotate coordinates

    :param x: coordinate along major axis (a) 
    :type x: class:`numpy.ndarray`
    :param y: coordinate along minor axis (b) 
    :type y: class:`numpy.ndarray`
    :param xs: translation along x-axis
    :type xs: float
    :param ys: translation along y-axis
    :type ys: float
    :param rot_rad: rotation angle, in radians
    :type rot_rad: float

    """

    xnew = (x - xs)*np.cos(rot_rad) + (y - ys)*np.sin(rot_rad)
    ynew = (y - ys)*np.cos(rot_rad) - (x - xs)*np.sin(rot_rad)

    return xnew,ynew

def get_ell_rads(x,y,ella,ellb):
    """   
    Get ellipsoidal radii from x,y standard

    :param x: coordinate along major axis (a) 
    :type x: class:`numpy.ndarray`
    :param y: coordinate along minor axis (b) 
    :type y: class:`numpy.ndarray`
    :param ella: scaling along major axis (should stay 1)
    :type ella: float
    :param ellb: scaling along minor axis
    :type ella: float

    """
 
    xnew = x/ella ; ynew = y/ellb

    return xnew, ynew



def mock_XMM_exposure(xymap,ksec,rot=30,incChipGaps=True):

    ### How to parameterize the radial profile of the relative exposure.

    rads_arcsec = np.logspace(0,3.5,500) # Out to 3000 arcseconds (~almost a a degree)
    beta_expo = 1.0/3.0
    r_c = 6*60 # 5 arcminutes = 300 arcseconds. Half-exposure at roughly 9 arcminutes

    relative_exposure = (1 + (rads_arcsec/r_c)**2)**(-1.5*beta_expo)
    exposure_profile = ksec*relative_exposure*1000

    exposure_map = grid_profile(rads_arcsec, exposure_profile, xymap)

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
