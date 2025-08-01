#=======================================================================#
#        ANALYTIC FITTING FORMULA FOR THE NUMERICAL RESULT              #
#     Transcribed results from the paper: "AN IMPROVED FORMULA FOR      #
# THE KINEMATICAL SUNYAEV-ZELDOVICH EFFECT FOR CLUSTERS OF GALAXIES"    #
# Nozawa, S. Itoh, N., Suda, Y., Ohhata, Y. 2006, NCimb, 121, 487       #
#                                                                       #
#        http://adsabs.harvard.edu/abs/2006NCimB.121..487N              #
#                                                                       #
#                                                                       #
#       DELTA n(X)                                                      #
#     ------------ = tau K(BETA, BETAZ, THETAE, X)                      #
#         n_0(X)                                                        #
#                                                                       #
#   K(BETA, BETAZ, THETAE, X) =  K1 * [                                 #
#                      beta**2 SUM( B_i * THETAE_i ** i) + // i=0..3    #
#             BETA P_1(cos{TG) SUM( C_i * THETAE_i ** i) + // i=0..4    #
#          BETA**2 P_2(cos{TG) SUM( D_i * THETAE_i ** i) + // i=0..3    #
#                                ]                                      #
#          -- WHERE --                                                  #
#                                                                       #
#         BETA = v/c (fractional speed of light)                        #
#        BETAZ = v_z/c (fractional speed of light along z direction)    #
#           TG = BETAZ/BETA                                             #
#       THETAE = (k_B * T_e) / (m_e * c**2)                             #
#            X = (h * nu) / (k_B * T_0)                                 #
#                                                                       #
#   NOTES:                                                              #
#      The positive z direction is towards us                           #
#            (different from the typical convention)                    #
#      The Nozawa+ 2006 paper claims 2% accuracy for                    #
#            THETAE=0.02 and BETAZ = 1/300                              #
#                                                                       #
#   SECONDARY NOTE:                                                     #
#        The terms which are multiplied by beta**2 (first line of terms #
#   above) should be very minimal (see Nozawa+ 2006). This is still     #
#   technically a kinematical effect, but you'll notice that it is not  #
#   directionally dependent.                                            #
#                                                                       #
#=======================================================================#
# TRANSCRIBED/WRITTEN (CHARLES ROMERO; June 16 2017):                   #
#                                                                       #
# DELTA I = 2 * (k_B*T_CMB)^3 / (h*c)^2 * K(BETA, BETAZ, THETAE, X)     #     
#             * tau                                                     #
#         = 2 * (k_B*T_CMB)^3 / (h*c)^2 * K(BETA, BETAZ, THETAE, X)     #     
#             * y / THETAE                                              #
#                                                                       #
#=======================================================================#
### PRIMARY ROUTINES (TO CALL EXTERNALLY):                              #
#                                                                       #
# (1) kSZ_conv(beta,betaz,llt,llx,ult,ulx,st,sx,rel=True)               #
#     This is really the workhorse for kSZ spectral distortions. It     #
#     will search for the proper range of thetae and X (i.e. Z) values  #
#     to apply the appropriate corrections.                             #
#                                                                       #
#=======================================================================#

import numpy as np
import astropy.units as u
#import get_data_info as gdi
import scipy.interpolate as spint
#import tSZ_spectrum as tsz
import astropy.constants as const
import scipy.constants as spconst

def get_sz_values():
    ########################################################
    ### Astronomical value...
    tcmb = 2.72548*u.K # Kelvin (uncertainty = 0.00057)
    ### Reference:
    ### http://iopscience.iop.org/article/10.1088/0004-637X/707/2/916/meta
    
    ### Standard physical values.
    thom_cross = (spconst.value("Thomson cross section") *u.m**2).to("cm**2")
    m_e_c2 = (const.m_e *const.c**2).to("keV")
    kpctocm = 3.0856776 *10**21
    boltzmann = spconst.value("Boltzmann constant in eV/K")/1000.0 # keV/K  
    planck = spconst.value("Planck constant in eV s")/1000.0 # keV s
    c = const.c
    keVtoJ = (u.keV).to("J") # I think I need this...) 
    Icmb = 2.0 * (boltzmann*tcmb.value)**3 / (planck*c.value)**2
    Icmb *= keVtoJ*u.W *u.m**-2*u.Hz**-1*u.sr**-1 # I_{CMB} in W m^-2 Hz^-1 sr^-1
    JyConv = (u.Jy).to("W * m**-2 Hz**-1")
    Jycmb = Icmb.to("Jy sr**-1")  # I_{CMB} in Jy sr^-1
    MJycmb= Jycmb.to("MJy sr**-1")

    ### The following constants (and conversions) are just the values (in Python):
    sz_cons_values={"thom_cross":thom_cross.value,"m_e_c2":m_e_c2.value,
                    "kpctocm":kpctocm,"boltzmann":boltzmann,
                    "planck":planck,"tcmb":tcmb.value,"c":c.value,}
    ### The following "constants" have units attached (in Python)!
    sz_cons_units={"Icmb":Icmb,"Jycmb":Jycmb,"thom_cross":thom_cross,
                   "m_e_c2":m_e_c2}

    return sz_cons_values, sz_cons_units

### Create some conversion values which may be useful throughout.
szcv,szcu = get_sz_values()
temp_conv = 1.0/szcv['m_e_c2']
freq_conv = (szcv['planck'] *1.0e9)/(szcv['boltzmann']*szcv['tcmb'])


def kSZ_conv(beta,betaz,llt,llx,ult,ulx,st,sx,rel=True):
    """
    Calculates relativistic corrections for kSZ as per Nozawa/Itoh group/papers. 
    BEWARE that you need to divide by theta_e to get a conversion between Compton y
    and Jy/beam as is often prescribed.
    
    Parameters
    __________
    beta- v/c (the fractional speed of light)
    betz- v_z/c (the fractional speed of light along the line of sight)
    llt - Lower limit on Theta_e
    llx - Lower limit on x
    ult - Upper limit on Theta_e
    ulx - Upper limit on x
    st  - Step in Theta_e
    sx  - Step in x
    
    Returns
    -------
    K(BETA, BETAZ, THETAE, X)
    """

    thetae=llt
    tarr = None
    xarr = None
    farr = None
        
    while thetae <= ult:
        x=llx        
        while x <= ulx: 

            if rel == True and beta != 0:
                tarr, xarr, farr = create_arrs_rel(beta, betaz,thetae,x,tarr=tarr,xarr=xarr,farr=farr)
            else:
                tarr, xarr, farr = create_arrs_nonrel(beta, betaz,thetae,x,tarr=tarr,xarr=xarr,farr=farr)


            x += sx  # Closes the loop over X
           
        thetae += st # Closes the loop over thetae

    return tarr, xarr, farr

def ksz_beta_terms(beta, betaz, thetae, X):

#    beta = v / szcv['c']  # v should be in m/s
    SH  = (np.exp(X/2)-np.exp(-X/2))/2
    CH  = (np.exp(X/2)+np.exp(-X/2))/2
    CTH = CH/SH
    
    XT  = X*CTH
    ST  = X/SH
    
    Y0  = -4.0e0+XT
    
    Y1  = -10.0e0+47.0e0*XT/2.0e0-42.0e0*(XT**2.0e0)/5.0e0              \
          +7.0e0*(XT**3.0e0)/10.0e0            \
          +(ST**2.0e0)*(-21.0e0/5.0e0+7.0e0*XT/5.0e0)
    
    Y2  = -15.0e0/2+1023.0e0*XT/8.0e0-868.0e0*(XT**2.0e0)/5.0e0         \
          +329.0e0*(XT**3.0e0)/5.0e0-44.0e0*(XT**4.0e0)/5.0e0           \
          +11.0e0*(XT**5.0e0)/30.0e0+(ST**2.0e0)                        \
          *(-434.0e0/5.0e0+658.0e0*XT/5.0e0
            -242.0e0*(XT**2.0e0)/5.0e0
            +143.0e0*(XT**3.0e0)/30.0e0)+(ST**4.0e0)                    \
        *(-44.0e0/5.0e0+187.0e0*XT/60.0e0)

    Y3  =  15.0e0/2+2505.0e0*XT/8.0e0-7098.0e0*(XT**2.0e0)/5.0e0         \
           +14253.0e0*(XT**3.0e0)/10.0e0-18594.0e0*(XT**4.0e0)/35.0e0    \
           +12059.0e0*(XT**5.0e0)/140.0e0-128.0e0*(XT**6.0e0)/21.0e0     \
           +16.0e0*(XT**7.0e0)/105.0e0+(ST**2.0e0)                       \
           *(-7098.0e0/10.0e0+14253.0e0*XT/5.0e0
             -102267.0e0*(XT**2.0e0)/35.0e0
             +156767.0e0*(XT**3)/140.0e0
             -1216.0e0*(XT**4.0e0)/7.0e0+64.0e0*(XT**5.0e0)/7.0e0)       \
            +(ST**4.0e0)*(-18594.0e0/35.0e0+205003.0e0*XT/280.0e0
                          -1920.0e0*(XT**2.0e0)/7.0e0+1024.0e0*(XT**3.0e0)/35.0e0) \
            +(ST**6.0e0)*(-544.0e0/21.0e0+992.0e0*XT/105.0e0)

    Y4  = -135.0e0/32.0e0+30375.0e0*XT/128.0e0   \
          -62391.0e0*(XT**2.0e0)/10.0e0          \
          +614727.0e0*(XT**3.0e0)/40.0e0         \
          -124389.0e0*(XT**4.0e0)/10.0e0         \
          +355703.0e0*(XT**5.0e0)/80.0e0         \
          -16568.0e0*(XT**6.0e0)/21.0e0          \
          +7516.0e0*(XT**7.0e0)/105.0e0          \
          -22.0e0*(XT**8.0e0)/7.0e0+11.0e0*(XT**9.0e0)/210.0e0   \
          +(ST**2.0e0)*(-62391.0e0/20.0e0+614727.0e0*XT/20.0e0         
                        -1368279.0e0*(XT**2.0e0)/20.0e0         
                        +4624139.0e0*(XT**3.0e0)/80.0e0
                        -157396.0e0*(XT**4.0e0)/7.0e0
                        +30064.0e0*(XT**5.0e0)/7.0e0-2717.0e0*(XT**6.0e0)/7.0e0
                        +2761.0e0*(XT**7.0e0)/210.0e0) \
        +(ST**4.0e0)*(-124389.0e0/10.0e0
                      +6046951.0e0*XT/160.0e0-248520.0*(XT**2.0e0)/7.0e0
                      +481024.0e0*(XT**3.0e0)/35.0e0-15972.0e0*(XT**4.0e0)/7.0e0
                      +18689.0e0*(XT**5.0e0)/140.0e0) \
        +(ST**6.0e0)*(-70414.0e0/21.0e0
                      +465992.0e0*XT/105.0e0-11792.0e0*(XT**2.0e0)/7.0e0
                      +19778.0e0*(XT**3.0e0)/105.0e0) \
        +(ST**8.0e0)*(-682.0e0/7.0e0+7601.0e0*XT/210.0e0)

    K1   = X*np.exp(X)/(np.exp(X)-1)
    
    B0 = Y0/3.0
    B1 = 5.0*Y0/6.0 + 2.0*Y1/3.0
    B2 = 5.0*Y0/8.0 + 1.5*Y1 + Y2
    B3 = -5.0*Y0/8.0 + 1.25*Y1 + 2.5*Y2 + 4.0*Y3/3.0

    #####################################################
    ##  Equation 20; Nozawa+ 2006.                     ##
    ##  Beta * P_1(cos{theta_gamma}) = betaz           ##
    ##  (See Equations 23, 24)                         ##
    #####################################################
    
    result = K1*beta**2 *(B0 + thetae*B1 + thetae**2*B2 + thetae**3*B3)

    return result
    
    
def ksz_C_terms(beta, betaz, thetae, X):

#    beta = v / szcv['c']  # v should be in m/s
    SH  = (np.exp(X/2)-np.exp(-X/2))/2
    CH  = (np.exp(X/2)+np.exp(-X/2))/2
    CTH = CH/SH
    
    XT  = X*CTH
    ST  = X/SH

    C0 = 1.0
    C1 = 10.0 - 47.0*XT/5.0 + 7.0*XT**2/5.0 + 0.7*ST**2
    C2 = 25.0 - 111.7*XT + 84.7*XT**2 - 18.3*XT**3 + 1.1*XT**4 +\
         ST**2 *(847.0/20.0 - 183.0*XT/5.0 + 6.05*XT**2) + 1.1*ST**4
    C3 = 75.0/4.0 - 21873.0*XT/40.0 + 49161.0*XT**2/40.0 -\
         27519.0*XT**3/35.0 + 6684.0*XT**4/35.0 - 3917.0*XT**5/210.0 +\
         64*XT**6/105.0 + ST**2 * (49161/80.0 - 55038*XT/35.0 +\
         36762*XT**2/35.0 - 50921*XT**3/210.0 + 608*XT**4/35.0) +\
         ST**4 * (6684/35.0 - 66589*XT/420.0 + 192*XT**2/7.0) +\
         272*ST**6/105.0
    C4 = -75.0/4.0 -10443.0*XT/8.0 + 359079.0*XT**2/40.0 -\
         938811.0*XT**3/70.0 + 261714.0*XT**4/35.0 - 263259.0*XT**5/140.0 +\
         4772.0*XT**6/21.0 -1336.0*XT**7/105.0 +11*XT**8/42.0 + ST**2 *\
         (359079.0/80.0 - 938811.0*XT/35.0 + 1439427.0*XT**2/35.0 -\
          3422367.0*XT**3/140.0 + 45334*XT**4/7.0 - 5344*XT**5/7.0 +\
          2717.0*XT**6/84.0) + ST**4 *\
          (261714.0/35.0 - 4475403.0*XT/280.0 + 71580.0*XT**2/7.0 -\
           85504.0*XT**3/35.0 + 1331.0*XT**4/7.0) + \
           ST**6 *(20281.0/21.0 - 82832.0*XT/105.0 + 2948.0*XT**2/21.0) + \
           341.0*ST**8/42.0
    
    K1   = X*np.exp(X)/(np.exp(X)-1)

    #####################################################
    ##  Equation 20; Nozawa+ 2006.                     ##
    ##  Beta * P_1(cos{theta_gamma}) = betaz           ##
    ##  (See Equations 23, 24)                         ##
    #####################################################
    
    result = K1*betaz*(C0 + thetae*C1 + thetae**2*C2 + thetae**3*C3 +\
                       thetae**4 *C4)

    return result

def ksz_D_terms(beta, betaz, thetae, X):
    
    SH  = (np.exp(X/2)-np.exp(-X/2))/2
    CH  = (np.exp(X/2)+np.exp(-X/2))/2
    CTH = CH/SH
    
    XT  = X*CTH
    ST  = X/SH

    D0 = -2.0/3.0 + 11*XT/30.0
    D1 = -4.0 + 12.0*XT - 6.0*XT**2 + 19.0*XT**3/30.0 + \
         ST**2 * (-3.0 * 19.0*XT/15.0)
    D2 = -10.0 + 542*XT/5.0 - 843*XT**2/5.0 + 10603.0*XT**3/140.0 -\
         409*XT**4/35.0 + 23*XT**5/42.0 + ST**2 *\
         (-84.3 + 10603*XT/70.0 - 4499*XT**2/70.0 + 299*XT**3/42.0) +\
         ST**4 *(-409/35.0 + 391*XT/84.0)
    D3 = -7.5 + 492.9*XT - 39777.0*XT**2/20.0 + 1199897.0*XT**3/560.0 -\
         4392*XT**4/5.0 + 16364.0*XT**5/105.0 - 3764.0*XT**6/315.0 +\
         101.0*XT**7/315.0 + ST**2 *\
         (-39777.0/40.0 + 119897.0*XT/280.0 - 24156.0*XT**2/5.0 +
          212732.0*XT**3/105.0 - 35758.0*XT**4/105.0 + 404.0*XT**5/21.0) +\
        ST**4 *(-4392.0/5.0 + 139094*XT/105.0 -3764.0*XT**2/7.0 +
                6464.0*XT**3/105.0) + ST**6 *\
        (-15997.0/315.0 + 6262.0*XT/315.0)
    
    K1 = X*np.exp(X)/(np.exp(X)-1)
    P2 = (3.0*(betaz/beta)**2 -1.0)/2.0

    #####################################################
    ##  Equation 20; Nozawa+ 2006.                     ##
    ##  Beta * P_1(cos{theta_gamma}) = betaz           ##
    ##  (See Equations 23, 24)                         ##
    #####################################################
    
    result = K1*beta**2 *P2*(D0 + thetae*D1 + thetae**2*D2 + thetae**3*D3)

    return result

def create_arrs_nonrel(beta,betaz,thetae,x,tarr=None,xarr=None,farr=None):

    factor = x**3 / (np.exp(x) - 1.0)       # Equation 2.16 in Nozawa+ 2000
    K1 = x*np.exp(x)/(np.exp(x)-1)

    if type(tarr) == type(None):
        tarr=np.array(thetae)
        xarr=np.array(x)
        farr=betaz*factor*K1
    else:
        tarr=np.append(tarr,np.array(thetae))
        xarr=np.append(xarr,np.array(x))
        farr=np.append(farr,np.array(betaz*K1)*factor)
        
    return tarr, xarr, farr

def create_arrs_rel(beta,betaz,thetae,x,tarr=None,xarr=None,farr=None):

    factor = x**3 / (np.exp(x) - 1.0)       # Equation 2.16 in Nozawa+ 2000
    dnn = ksz_beta_terms(beta, betaz, thetae, x) +\
          ksz_C_terms(beta, betaz, thetae, x) +\
          ksz_D_terms(beta, betaz, thetae, x)
    
    if type(tarr) == type(None):
        tarr=np.array(thetae)
        xarr=np.array(x)
        farr=np.array(dnn)*factor
    else:
        tarr=np.append(tarr,np.array(thetae))
        xarr=np.append(xarr,np.array(x))
        farr=np.append(farr,np.array(dnn)*factor)
    return tarr, xarr, farr
