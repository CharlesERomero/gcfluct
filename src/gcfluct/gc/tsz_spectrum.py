#=======================================================================#
#        ANALYTIC FITTING FORMULA FOR THE NUMERICAL RESULT              #
#     We have published our paper "RELATIVISTIC CORRECTIONS             #
# TO THE SUNYAEV-ZELDOVICH EFFECT FOR CLUSTERS OF GALAXIES.             #
# IV. ANALYTIC FITTING FORMULA FOR THE NUMERICAL RESULTS"               #
# Nozawa, S., Itoh, N., Kawana, Y., and Kohyama, Y. 2000, ApJ, 536, 31. #
#                                                                       #
# ---   This routine _DOES_ include corrections from               ---  #
# ---   Nozawa+ 2002 and Itoh & Nozawa 2004                        ---  #
# ---   (Note to self: June, 2020)                                 ---  #
#                                                                       #
#     http://adsabs.harvard.edu/abs/2000ApJ...536...31N                 #
#                                                                       #
#                   k_B * T_e                  h * nu                   #
#    For theta_e = -----------      and X = -------------               #
#                   m_e c**2                 k_B * T_CMB                #
#                                                                       #
#       DELTA n(X)                                                      #
#     ------------ = y F(theta_e, X)                                    #
#         n_0(X)                                                        #
#                                                                       #
#      F(theta_e, X) = F1*(Y0+theta_e*Y1 + theta_e^2*Y2                 #
#                            + theta_e^3*Y3 + theta_e^4*Y4) + R         #
#                                                                       #
#      for  0.0 <= theta_e < 0.02                                       #
#                                                                       #
#         R = 0                                                         #
#                                                                       #
#      for 0.02 <= theta_e <= 0.05                                      # 
#                                                                       #
#          /    0.0 < X < 2.5                                           #
#          |                                                            #
#          |       R = 0                                                #
#    AND   |                                                            #
#          |    2.5 <= X <= 20                                          #
#          |                                                            #
#          \       R = Sum_{i,j}^{10} a_{i,j} THETA_e^{i} Z^{j}         #
#                                                                       #
#      Where  THETA_e = 25*(theta_e - 0.01), Z = (X - 2.5)/17.5         #
#                                                                       #
#                                 DEC. 1  1999  YK IN RI-DA-GROUP       #
#=======================================================================#
# ADDITIONAL DOCUMENTATION (CHARLES ROMERO; June 15 2017):              #
#                                                                       #
# The spectral distortion (in intensity units, e.g. Jy sr^-1) is given  #
# by:                                                                   #
#                                                                       #
# DELTA I = 2 [(k_B*T_CMB)^3 / (h*c)^2]* F(theta_e, X) * y / theta_e    #
#                     or Equation 2.34 would be:                        #
# DELTA I = X^3 F(theta_e, X) * y / (exp(X) - 1)                        #
#                                                                       #
#                                                                       #
# where [ F(theta_e, X) * y / theta_e ] is unitless.                    #
# and I_0 = I_CMB = 2 [(k_B*T_CMB)^3 / (h*c)^2]                         #
# Thus:                                                                 #
#                                                                       #
#-----------------------------------------------------------------------#
#> DELTA T = (exp(X)-1) * F(theta_e, X) * y / (theta_e * X * exp(X) )  <#
#-----------------------------------------------------------------------#
#                                                                       #
# THIS PROGRAM RETURNS F(theta_e,X)                                     #
#                                                                       #
# NOTE that this is not what you would naively derive from any of the   #
# Itoh or Nozawa papers where they give a definition for DELTA I        #
#                                                                       #
#=======================================================================#
###                                                                   ###
###    Python conversion done by Charles Romero (11 June 2017)        ###
###    Some additional routines were added too.                       ###
###                                                                   ###
#=======================================================================#
### PRIMARY ROUTINES (TO CALL EXTERNALLY):                              #
#                                                                       #
# (1) tSZ_conv_nonrel(llt,llx,ult,ulx,st,sx)                            #
#     This routine returns the non-relativistic spectral distortion     #
#     (as seen in Carlstrom+ 2002, but using F(theta_e,X) notation as   #
#     indicated above.                                                  #
# (2) tSZ_conv(llt,llx,ult,ulx,st,sx)                                   #
#     This is really the workhorse for tSZ spectral distortions. It     #
#     will search for the proper range of thetae and X (i.e. Z) values  #
#     to apply the appropriate corrections.                             #
#                                                                       #
#=======================================================================#


import numpy as np
#import get_data_info as gdi
import astropy.units as u
import scipy.interpolate as spint
import astropy.constants as const
import scipy.constants as spconst
from numpy.typing import NDArray
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, Union

def get_sz_values():
    """
    Legacy code which works, but needs to be documented. (On the to-do list).

    """
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

def tSZ_conv_nonrel(llt,llx,ult,ulx,st,sx):
    """
    Legacy code which works, but needs to be documented. (On the to-do list).

    """

    thetae=llt
    tarr = None
    xarr = None
    farr = None
        
    while thetae <= ult:
        x=llx        
        while x <= ulx: 

            tarr, xarr, farr = create_arrs_nonrel(thetae,x,tarr=tarr,xarr=xarr,farr=farr)

            x += sx  # Closes the loop over X
           
        thetae += st # Closes the loop over thetae

    return tarr, xarr, farr

def tSZ_conv(llt,llx,ult,ulx,st,sx):
    """
    Calculates relativistic corrections for tSZ as per Nozawa/Itoh group/papers. 
    BEWARE that you need to divide by theta_e to get a conversion between Compton y
    and Jy/beam as is often prescribed.
    
    Parameters
    ----------
    llt : np.floating
        Lower limit on Theta_e
    llx : np.floating
        Lower limit on x
    ult : np.floating
        Upper limit on Theta_e
    ulx : np.floating
        Upper limit on x
    st : np.floating
        Step in Theta_e
    sx : np.floating
        Step in x

    Returns
    -------
    tarr : NDArray[np.floating]
        The array of theta_e values
    xarr : NDArray[np.floating]
        The array of X values
    farr : NDArray[np.floating]
        The array of F(theta_e,x) values
    """

    thetae=llt
    tarr = None
    xarr = None
    farr = None
        
    while thetae <= ult:
        x=llx        
        while x <= ulx: 

            if thetae < 0.02:
                tarr, xarr, farr = create_arrs_lowrel(thetae,x,tarr=tarr,xarr=xarr,farr=farr)
            else:
                if thetae < 0.035:
                    if x > 2.4 and x < 15.0:
                        tarr, xarr, farr = create_arrs_Mrel_Hacc(thetae,x,tarr=tarr,xarr=xarr,farr=farr)
                    else:
                        tarr, xarr, farr = create_arrs_lowrel(thetae,x,tarr=tarr,xarr=xarr,farr=farr)
                else:
                    if thetae < 0.05:
                        if x < 2.5:
                            tarr, xarr, farr = create_arrs_lowrel(thetae,x,tarr=tarr,xarr=xarr,farr=farr)
                        else:
                            tarr, xarr, farr = create_arrs_modrel(thetae,x,tarr=tarr,xarr=xarr,farr=farr)
                    else:
                        if x > 1.2 and x < 17:  # Also has an "upper limit of X=17
                            tarr, xarr, farr = create_arrs_higrel(thetae,x,tarr=tarr,xarr=xarr,farr=farr)
                        else:
                            tarr, xarr, farr = create_arrs_lowrel(thetae,x,tarr=tarr,xarr=xarr,farr=farr)
                        
            x += sx  # Closes the loop over X
           
        thetae += st # Closes the loop over thetae

    return tarr, xarr, farr
 #12   FORMAT (2(E11.5,TR3),E14.6)
 #     STOP
 #     END

#----------------  SUBFUNCTION R(theta_e,X)  -----------------------

def nozawa_r(thetae,x):
    """
    Legacy code which works, but needs to be documented. (On the to-do list).

    """
 
    A=np.array([4.13674e-03,
                -3.31208e-02,   1.10852e-01,  -8.50340e-01,   9.01794e+00,
                -4.66592e+01,   1.29713e+02,  -2.09147e+02,   1.96762e+02,
                -1.00443e+02,   2.15317e+01,  -4.40180e-01,   3.06556e+00,
                -1.04165e+01,   2.57306e+00,   8.52640e+01,  -3.02747e+02,
                5.40230e+02,  -5.58051e+02,   3.10522e+02,  -6.68969e+01,
                -4.02135e+00,   1.04215e+01,  -5.17200e+01,   1.99559e+02,
                -3.79080e+02,   3.46939e+02,   2.24316e+02,  -1.36551e+03,
                2.34610e+03,  -1.98895e+03,   8.05039e+02,  -1.15856e+02,
                -1.04701e+02,   2.89511e+02,  -1.07083e+03,   1.78548e+03,
                -2.22467e+03,   2.27992e+03,  -1.99835e+03,   5.66340e+02,
                -1.33271e+02,   1.22955e+02,   1.03703e+02,   5.62156e+02,
                -4.18708e+02,   2.25922e+03,  -1.83968e+03,   1.36786e+03,
                -7.92453e+02,   1.97510e+03,  -6.95032e+02,   2.44220e+03,
                -1.23225e+03,  -1.35584e+03,  -1.79573e+03,  -1.89408e+03,
                -1.77153e+03,  -3.27372e+03,   8.54365e+02,  -1.25396e+03,
                -1.51541e+03,  -3.26618e+03,  -2.63084e+03,   2.45043e+03,
                5.10306e+03,   3.58624e+03,   9.51532e+03,   1.91833e+03,
                9.66009e+03,   6.12196e+03,   1.12396e+03,   3.46686e+03,
                4.91340e+03,  -2.76135e+02,  -5.50214e+03,  -7.96578e+03,
                -4.52643e+03,  -1.84257e+04,  -9.27276e+03,  -9.39242e+03,
                -1.34916e+04,  -6.12769e+03,   3.49467e+02,   7.13723e+02,
                7.73758e+03,   5.62142e+03,   4.89986e+03,   3.50884e+03,
                1.86382e+04,   1.71457e+04,   1.45701e+03,  -1.32694e+03,
                -5.84720e+03,  -6.47538e+03,  -9.17737e+03,  -7.39415e+03,
                -2.89347e+03,   1.56557e+03,  -1.52319e+03,  -9.69534e+03,
                -1.26259e+04,   5.42746e+03,   2.19713e+04,   2.26855e+04,
                1.43159e+04,   4.00062e+03,   2.78513e+02,  -1.82119e+03,
                -1.42476e+03,   2.82814e+02,   2.03915e+03,   3.22794e+03,
                -3.47781e+03,  -1.34560e+04,  -1.28873e+04,  -6.66119e+03,
                -1.86024e+03,   2.44108e+03,   3.94107e+03,  -1.63878e+03])

    Aij = np.transpose(A.reshape(11,11))
    
    THE     = (thetae-0.01)*100.0/4.0
    Z       = (x-2.5)/17.5

#    THE     = 25.0*(thetae-0.01)
#    Z       = (x-2.4)/17.6
    result  = 0.0

    for i in range(Aij.shape[0]):
        for j in range(Aij.shape[0]):
            result += Aij[i,j] * THE**i * Z**j

    return result

#-------  SUBFUNCTION FINK(theta_e,X)  ----------------------
#      FINK = F1*(Y0+theta_e*Y1+ ... +theta_e^4*Y4)
# This comes from Itoh+ 1998
#------------------------------------------------------------

def fink(thetae,X):
    """
    Legacy code which works, but needs to be documented. (On the to-do list).

    """

    SH  = (np.exp(X/2)-np.exp(-X/2))/2
    CH  = (np.exp(X/2)+np.exp(-X/2))/2
    CTH = CH/SH
    
    XT  = X*CTH
    ST  = X/SH
    
    Y0  = -4.0e0+XT
    
    Y1  = -10.0e0+47.0e0*XT/2.0e0-42.0e0*(XT**2.0e0)/5.0e0              \
          +7.0e0*(XT**3.0e0)/10.0e0                                     \
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

    F1   = thetae*X*np.exp(X)/(np.exp(X)-1)

    result = F1*(Y0+(thetae)*Y1+(thetae**2)*Y2  
                 +(thetae**3)*Y3+(thetae**4)*Y4)

    return result

def fnonrel(thetae,X):
    """
    Legacy code which works, but needs to be documented. (On the to-do list).

    """

    SH  = (np.exp(X/2)-np.exp(-X/2))/2
    CH  = (np.exp(X/2)+np.exp(-X/2))/2
    CTH = CH/SH
    
    XT  = X*CTH
    ST  = X/SH
    
    Y0  = -4.0e0+XT
    F1   = thetae*X*np.exp(X)/(np.exp(X)-1)
    result = F1*Y0

    return result
    
def Jyperbeam_factors(bv):
    """
    Legacy code which works, but needs to be documented. (On the to-do list).

    """
    
    srtosas = (1.0*u.sr).to("arcsec**2")     # Square arcseconds in a steradian
    bpsr = srtosas/bv                        # Beams per steradiann
    Inot = szcu["Jycmb"]
    
    result=Inot.value/bpsr
       
    return result

def TBright_factors(x):
    """
    Legacy code which works, but needs to be documented. (On the to-do list).

    """
    
    Tcmb = szcv["tcmb"]
    f1 = x**3 / (np.exp(x) - 1.0)       # Equation 2.16 in Nozawa+ 2000
    f2 = x*np.exp(x)/(np.exp(x)-1)
    
#    print bpsr,srtosas,Inot
    
    result= Tcmb / (f1*f2)
    
### FACTORS necessary here; I have built this in to analytic formulas
### and it's already built into the data table (from which values may be
### interpolated)
#    result=factors                # Useful for comparing to Figure 2.
    
    return result
    
def tSZ_conv_single(temperature, frequency):
    """
    Legacy code which works, but needs to be documented. (On the to-do list).

    """

    theta = temp_conv * temperature
    x = freq_conv * frequency
    
    ulx= x
    llx= x
    sx = 1.0    # Way more than necessary; thus, we are sure to only have 1 value
    ult= theta
    llt= theta
    st = 1.0    # Way more than necessary; thus, we are sure to only have 1 value

#    import pdb; pdb.set_trace()
    
    tarr, xarr, fofx = tSZ_conv(llt,llx,ult,ulx,st,sx)

    return fofx.item(0) # Return as a scalar
    
def tSZ_conv_range(tlow,thigh,tstep,flow,fhigh,fstep):
    """
    Legacy code which works, but needs to be documented. (On the to-do list).

    """

    llx = freq_conv * flow
    ulx = freq_conv * fhigh
    sx  = freq_conv * fstep    
    llt = temp_conv * tlow
    ult = temp_conv * thigh
    st  = temp_conv * tstep
        
    tarr, xarr, fofx = tSZ_conv(llt,llx,ult,ulx,st,sx)

    temparr = tarr/temp_conv
    freqarr = xarr/freq_conv
    
    return temparr,freqarr,fofx

def itoh_2004_r(thetae,x):
    """
    Legacy code which works, but needs to be documented. (On the to-do list).

    """

    Aij=np.array([[-1.81317E1,  9.97038E1 , -6.07438E1 ,  1.05143E3 , -2.86734E3 ,  7.73353E3 ,
          -8.16644E3 , -5.37712E3 ,  1.52226E4 , 7.18726E3 ,  -1.39548E4 , -2.08464E4 , 1.79040E4],
         [1.68733E2 , -6.07829E2 ,  1.14933E3 , -2.42382E2 , -7.73030E2 ,  5.33993E3 ,
          -4.03443E3 ,  3.00692E3 ,  9.58809E3 , 8.16574E3 ,  -6.13322E3 , -1.48117E4 , 3.43816E4],
         [-6.69883E2,  1.59654E3 , -3.33375E3 , -2.13234E3 , -1.80812E2 ,  3.75605E3 ,
          -4.75180E3 , -4.50495E3 ,  5.38753E3 , 5.03355E3 ,  -1.18396E4 , -8.58473E3 , 3.96316E4],
         [1.56222E3 , -1.78598E3 ,  5.13747E3 ,  4.10404E3 ,  5.54775E2 , -3.89994E3 ,
          -1.22455E3 ,  1.03747E3 ,  4.32237E3 , 1.03805E3 ,  -1.47172E4 , -1.23591E4 , 1.77290E4],
         [-2.34712E3,  2.78197E2 , -5.49648E3 , -5.94988E2 , -1.47060E3 , -2.84032E2 ,
          -1.15352E3 , -1.17893E3 ,  7.01209E3 , 4.75631E3 ,  -5.13807E3 , -8.73615E3 , 9.41580E3],
         [1.92894E3 ,  1.17970E3 ,  3.13650E3 , -2.91121E2 , -1.15006E3 ,  4.17375E3 ,
          -3.31788E2 ,  1.37973E3 , -2.48966E3 , 4.82005E3 ,  -1.06121E4 , -1.19394E4 , 1.34908E4],
         [6.40881E2 , -6.81789E2 ,  1.20037E3 , -3.27298E3 ,  1.02988E2 ,  2.03514E3 ,
          -2.80502E3 ,  8.83880E2 ,  1.68409E3 , 4.26227E3 ,  -6.37868E3 , -1.11597E4 , 1.46861E4],
         [-4.02494E3, -1.37983E3 , -1.65623E3 ,  7.36120E1 ,  2.66656E3 , -2.30516E3 ,
          5.22182E3 , -8.53317E3 ,  3.75800E2 , 8.49249E2 ,  -6.88736E3 , -1.01475E4 , 4.75820E3],
         [4.59247E3 ,  3.04203E3 , -2.11039E3 ,  1.32383E3 ,  1.10646E3 , -3.53827E3 ,
          -1.12073E3 , -5.47633E3 ,  9.85745E3 , 5.72138E3 ,   6.86444E3 , -5.72696E3 , 1.29053E3],
         [-1.61848E3, -1.83704E3 ,  2.06738E3 ,  4.00292E3 , -3.72824E1 ,  9.10086E2 ,
          3.72526E3 ,  3.41895E3 ,  1.31241E3 , 6.68089E3 ,  -4.34269E3 , -5.42296E3 , 2.83445E3],
         [-1.00239E3, -1.24281E3 ,  2.46998E3 , -4.25837E3 , -1.83515E2 , -6.47138E2 ,
          -7.35806E3 , -1.50866E3 , -2.47275E3 , 9.09399E3 ,  -2.75851E3 , -6.75104E3 , 7.00899E2],
         [1.04911E3 ,  2.07475E3 , -3.83953E3 ,  7.79924E2 , -4.08658E3 ,  4.43432E3 ,
          3.23015E2 ,  6.16180E3 , -1.00851E4 , 7.65063E3 ,   1.52880E3 , -6.08330E3 , 1.23369E3],
         [-2.61041E2, -7.22803E2 ,  1.34581E3 ,  5.90851E2 ,  3.32198E2 ,  2.58340E3 ,
          -5.97604E2 , -4.34018E3 , -3.58925E3 , 2.59165E3 ,   6.76140E3 , -6.22138E3 , 4.40668E3]])

    THE     = 10.0*thetae
    Z       = x/20.0
    result  = 0.0

    for i in range(Aij.shape[0]):
        for j in range(Aij.shape[0]):
            result += Aij[i,j] * THE**i * Z**j

    return result

def itoh_2002_r(thetae,x):
    """
    Legacy code which works, but needs to be documented. (On the to-do list).

    """

    Cki=np.array([[2.38348e-4,  7.70060e-4,  7.61858e-3,  -5.11946e-2,  2.46541e-1,  -6.55886e-1,
                   1.04523e0, -9.82508e-1,  5.03123e-1,  -1.08542e-1,  2.26789e-4],
                  [-5.55171e-4, -9.02681e-3,  1.78164e-1,  -1.94593e0,  1.08690e1,  -3.59915e1,
                   7.42310e1, -9.64435e1,  7.67639e1,  -3.41927e1,  6.52867e0],
                  [-4.87242e-3, -1.28519e-2, -1.87023e-1,   1.05512e0, -3.55801e0,   4.42411e0,
                   3.94568e0, -1.99013e1,  2.61111e1,  -1.56504e1,  3.66727e0],
                  [3.15750e-3,  4.58081e-2, -7.16782e-1,   6.63376e0, -3.08818e1,   8.52378e1,
                   -1.46255e2,  1.57577e2, -1.03418e2,   3.76441e1, -5.79648e0],
                  [1.82590e-2,  1.83934e-2,  1.23170e0,  -7.78160e0,  2.76593e1,  -4.88313e1,
                   3.29857e1,  2.49955e1, -6.15775e1,   4.18075e1, -1.01200e1],
                  [-9.21742e-3, -9.91460e-2,  1.04791e0,  -7.79860e0,  2.53107e1,  -4.01387e1,
                   1.82737e1,  3.43644e1, -5.81462e1,   3.44306e1, -7.45573e0],
                  [-2.83553e-2,  3.02605e-2, -2.84910e0,   1.79093e1, -6.14780e1,   1.13587e2,
                   -1.14236e2,  5.43853e1, -4.15730e0,  -4.21105e0,  4.33807e-1],
                  [1.18551e-2,  9.94126e-2, -6.72433e-1,   4.20865e0, -9.40878e0,   1.00474e1,
                   -1.44360e1,  4.60361e1, -7.82655e1,   5.95121e1, -1.68492e1],
                  [2.03055e-2, -6.51622e-2,  2.56521e0,  -1.45222e1,  4.18355e1,  -6.01877e1,
                   4.52390e1, -3.22989e1,  4.68655e1,  -4.31270e1,  1.41021e1],
                  [-5.19957e-3, -3.55883e-2,  1.27246e-1,  -7.18783e-1,  2.08060e0,  -1.28611e1,
                   5.62636e1, -1.27553e2,  1.53193e2,  -9.35633e1,  2.29488e1],
                  [-5.51710e-3,  2.89212e-2, -7.71024e-1,   3.43548e0, -5.04902e0,  -6.82987e0,
                   2.70797e1, -1.99352e1, -1.36621e1,   2.43417e1, -8.74479e0]])
    Cik = np.transpose(Cki)

    THE     = 200.0*(thetae - 0.02)/3.0
    Z       = (x - 8.7)/6.3
    result  = 0.0

    for i in range(Cik.shape[0]):
        for k in range(Cik.shape[0]):
            result += Cik[i,k] * THE**i * Z**k

    return result
    
def itoh_2004_int(theta, x, kind='cubic'):
    """
    Legacy code which works, but needs to be documented. (On the to-do list).

    """

    ref_file ='/home/romero/Python/model_fitting/di.dat'
    ref_data = np.loadtxt(ref_file)
    ref_theta= ref_data[:,0]
    ref_x    = ref_data[:,1]
    ref_conv = ref_data[:,2]

    f = spint.interp2d(ref_theta,ref_x,ref_conv, kind=kind)
#    f = spint.RectBivariateSpline(ref_theta,ref_x,ref_conv)
    
    conv = f(theta,x)

    return conv

def create_arrs_nonrel(thetae,x,tarr=None,xarr=None,farr=None):
    """
    Legacy code which works, but needs to be documented. (On the to-do list).

    """

    factor = x**3 / (np.exp(x) - 1.0)       # Equation 2.16 in Nozawa+ 2000
    
    if type(tarr) == type(None):
        tarr=np.array(thetae)
        xarr=np.array(x)
        farr=np.array(fnonrel(thetae,x))*factor
    else:
        tarr=np.append(tarr,np.array(thetae))
        xarr=np.append(xarr,np.array(x))
        farr=np.append(farr,np.array(fnonrel(thetae,x))*factor)
    return tarr, xarr, farr

def create_arrs_lowrel(thetae,x,tarr=None,xarr=None,farr=None):
    """
    Legacy code which works, but needs to be documented. (On the to-do list).

    """

    factor = x**3 / (np.exp(x) - 1.0)       # Equation 2.16 in Nozawa+ 2000
    
    if type(tarr) == type(None):
        tarr=np.array(thetae)
        xarr=np.array(x)
        farr=np.array(fink(thetae,x))*factor
    else:
        tarr=np.append(tarr,np.array(thetae))
        xarr=np.append(xarr,np.array(x))
        farr=np.append(farr,np.array(fink(thetae,x))*factor)
    return tarr, xarr, farr

def create_arrs_modrel(thetae,x,tarr=None,xarr=None,farr=None):
    """
    Legacy code which works, but needs to be documented. (On the to-do list).

    """

    factor = x**3 / (np.exp(x) - 1.0)       # Equation 2.16 in Nozawa+ 2000
    if type(tarr) == type(None):
        tarr=np.array(thetae)
        xarr=np.array(x)
        farr=np.array(fink(thetae,x)+nozawa_r(thetae,x))*factor
    else:
        tarr=np.append(tarr,np.array(thetae))
        xarr=np.append(xarr,np.array(x))
        farr=np.append(farr,np.array(fink(thetae,x)+nozawa_r(thetae,x))*factor)
    return tarr, xarr, farr

def create_arrs_Mrel_Hacc(thetae,x,tarr=None,xarr=None,farr=None):
    """
    Legacy code which works, but needs to be documented. (On the to-do list).

    """

    factor = x**3 / (np.exp(x) - 1.0)       # Equation 2.16 in Nozawa+ 2000
    if type(tarr) == type(None):
        tarr=np.array(thetae)
        xarr=np.array(x)
        farr=np.array(fink(thetae,x)+itoh_2002_r(thetae,x))*factor
    else:
        tarr=np.append(tarr,np.array(thetae))
        xarr=np.append(xarr,np.array(x))
        farr=np.append(farr,np.array(fink(thetae,x)+itoh_2002_r(thetae,x))*factor)
    return tarr, xarr, farr

def create_arrs_higrel(thetae,x,tarr=None,xarr=None,farr=None):
    """
    Legacy code which works, but needs to be documented. (On the to-do list).

    """
    
    X0 = 3.830*(1.0 + 1.1674*thetae - 0.8533*thetae**2)
    Gfact = thetae * x**2 * np.exp(-x) * (x - X0)
#    Ffact = x**3 / (np.exp(x) - 1.0) 
    if type(tarr) == type(None):
        tarr=np.array(thetae)
        xarr=np.array(x)
        ### No need to use FINK() ??
        farr=np.array(itoh_2004_r(thetae,x))*Gfact
    else:
        tarr=np.append(tarr,np.array(thetae))
        xarr=np.append(xarr,np.array(x))
        farr=np.append(farr,np.array(itoh_2004_r(thetae,x))*Gfact)
    return tarr, xarr, farr

