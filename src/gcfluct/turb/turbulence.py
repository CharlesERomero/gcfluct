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
