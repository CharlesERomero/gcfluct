import numpy as np
import scipy.special as sps

def get_multiGauss_FT(k,sigs,norms):
    """
    Add together the Gaussians in the Fourier domain

    Parameters
    ----------
    k : %(dtypes)s
        A one-dimensional array of wavenumbers (e.g. inverse arcseconds). The units of karr must be
        the inverse of the units used for the sigma(s) describing the N-Gaussian (below).
    sigs : %(dtypes)s
        array-like collection of Gaussian standard deviations.
    norms : %(dtypes)s
        array-like collection of Gaussian normalizations. The sum of norms must equal unity
    """

    normtotal = np.sum(norms)
    normcheck = np.abs(normtotal-1.0) <= 1.0e-6  # Confirms total is unity, within some tolerance.
    if not normcheck:
        print("Your total normalization doesn't add up to 1.")
        print(norms)
        import pdb;pdb.set_trace()
            
    g = np.zeros(k.shape)    
    for n,s in zip(norms,sigs):
        t = 2 * np.pi**2 * s**2
        g += n * np.exp(-k**2 * t)
        
    return g

def PSF_PS_from_pars(k,pars):
    """

    Compute the power spectrum for a multi-Gaussian Point Spread Function

    Parameters
    ----------
    k : %(dtypes)s
        A one-dimensional array of wavenumbers (e.g. inverse arcseconds). The units of karr must be
        the inverse of the units used for the sigma(s) describing the N-Gaussian (below).
    pars : %(dtypes)s
        array-like collection of Gaussian parameters. You can have an N-Gaussian beam.
        If 1 Gaussian, pars is a 1-element array of [sigma]
        if 2 Gaussians, pars is a 3-element array of [sigma1, sigma2, norm2], where
        norm2 = np.log(n2) if the beam is described by:
        beam = [1.0 * np.exp(-r**2 / (2*sigma1**2)) + n2* np.exp(-r**2 / (2*sigma2**2)) ]/(1+n2)
        That is, norm1 \equiv 0 = np.log(n1) = np.log(1.0); alternatively n1 \equiv 1
        If 3 Gaussian, pars is a 5-element array: [sigma1, sigma2, sigma3, norm2, norm3].
        beam will have a similar form and always be renormalized by (1+n2+n3 +....+nN)
    """
    
    npars   = len(pars)
    nGauss  = (npars+1)//2
    Psigs   = pars[:nGauss]
    if nGauss > 1:
        Inorms  = np.exp(pars[nGauss:])
        NormTot = np.sum(Inorms)+1.0
        Pnorms  = np.hstack((1.0,Inorms)) / NormTot
    else:
        Pnorms  = [1.0]
    #print(pars[nGauss:])
        
    CompFFT = get_multiGauss_FT(k,Psigs,Pnorms)
    CompPS  = CompFFT**2                        # Square the Fourier transform for the PS
    
    return CompPS

def get_multiGauss_terms(karr,pars,index):
    """

    This corrects for the bias noted in Romero+ 2023.*
    * The equation B13 has a sign error; the quantity (n/2 + 2 - alpha/2), as the exponent of the Gaussian terms,
    should have a minus sign in front of it. (See below: -expo)

    Parameters
    ----------
    karr : %(dtypes)s
        A one-dimensional array of wavenumbers (e.g. inverse arcseconds). The units of karr must be
        the inverse of the units used for the sigma(s) describing the N-Gaussian (below).
    pars :%(dtypes)s
        array-like collection of Gaussian parameters. You can have an N-Gaussian beam.
        If 1 Gaussian, pars is a 1-element array of [sigma]
        if 2 Gaussians, pars is a 3-element array of [sigma1, sigma2, norm2], where
        norm2 = np.log(n2) if the beam is described by:
        beam = [1.0 * np.exp(-r**2 / (2*sigma1**2)) + n2* np.exp(-r**2 / (2*sigma2**2)) ]/(1+n2)
        That is, norm1 \equiv 0 = np.log(n1) = np.log(1.0); alternatively n1 \equiv 1
        If 3 Gaussian, pars is a 5-element array: [sigma1, sigma2, sigma3, norm2, norm3].
        beam will have a similar form and always be renormalized by (1+n2+n3 +....+nN)
    index : float 
         The spectral index assumed. [Equivalent to alpha in get_multiGauss_bias()]

    """
    
    ndim    = 2
    expo    = (ndim/2 + 2 - index/2)
    npars   = len(pars)
    nGauss  = (npars+1)//2
    Psigs   = pars[:nGauss]
    if nGauss > 1:
        Inorms  = np.exp(pars[nGauss:])
        NormTot = np.sum(Inorms)+1.0
        Pnorms  = np.hstack((1.0,Inorms)) / NormTot
    else:
        Pnorms  = [1.0]

    kshape = karr.shape
    if len(kshape) > 1:
        print("karr should be one-dimensional")

    corrs = np.ones(kshape[0])
        
    for i in range(kshape[0]):

        thiscorr = 0.0
        
        for j in range(nGauss):
            k1 = 1.0/(np.sqrt(2)*np.pi*Psigs[j])
            x1 = k1/karr[i]
            n1 = Pnorms[j]
            for k in range(nGauss):      
                k2 = 1.0/(np.sqrt(2)*np.pi*Psigs[k])
                x2 = k2/karr[i]
                n2 = Pnorms[k]
                if j == k:
                    coef = n1*n2
                    nume = x1**2 + 1
                    deno = x1**2
                else:
                    coef = n1*n2
                    nume = (2*x1**2 * x2**2 + x1**2 + x2**2)
                    deno = 2.0*x1**2 * x2**2
                term = coef * (nume/deno)**(-expo)
                thiscorr += term

        corrs[i] = thiscorr*1.0

    return corrs

def get_multiGauss_bias(karr,pars,alpha,ignPSF=False,PBonly=False):

    """
    Corrects for:
    (1) The scalar bias induced by Arevalo+ 2012
    (2) The non-scalar bias induced by a beam *and* use of Arevalo+ 2012
    --- In order for this to be done, some spectral index, alpha MUST be assumed! ---
    --- where P(k) $\propto$ k**-alpha                                              ---
    Note that there is a correction for the power spectrum of the beam (PSF) within THIS term.
    If you want to correct for the PSF and the bias then these terms cancel! In light of this,
    I've added a keyword to allow you to ignore the PS term.
    ---------------------------------------------------------------------------------
    karr    : :%(dtypes)s
        The array of k (wavenumber) values at which to calculate the total bias
    pars    : :%(dtypes)s
        array-like collection of Gaussian parameters. You can have an N-Gaussian beam.
        If 1 Gaussian, pars is a 1-element array of [sigma]
        if 2 Gaussians, pars is a 3-element array of [sigma1, sigma2, norm2], where
        norm2 = np.log(n2) if the beam is described by:
        beam = [1.0 * np.exp(-r**2 / (2*sigma1**2)) + n2* np.exp(-r**2 / (2*sigma2**2)) ]/(1+n2)
        That is, norm1 \equiv 0 = np.log(n1) = np.log(1.0); alternatively n1 \equiv 1
        If 3 Gaussian, pars is a 5-element array: [sigma1, sigma2, sigma3, norm2, norm3].
        beam will have a similar form and always be renormalized by (1+n2+n3 +....+nN)
    alpha   : float 
        The assumed spectral index (convention given above)
    [ignPS] : bool
        As mentioned above, allows you to ignore the PSF power spectrum term with respect
        to this bias. Given that you want to correct for the PSF power spectrum, they will
        cancel each other out when correcting your measured power spectrum. You can bypass
        the additional calculations by ignoring the PSF term here.

    """
    ndim  = 2
    lilg  = (ndim/2 + 2 - alpha/2)
    
    p1      = 2**(alpha/2)
    p2      = get_multiGauss_terms(karr,pars,alpha)
    p3      = sps.gamma(lilg) / sps.gamma(ndim/2 + 2)
    PSF_ps  = 1.0 if ignPSF else PSF_PS_from_pars(karr,pars)
    
    bias  = p2/PSF_ps if PBonly else p1*p2*p3 / PSF_ps

    return bias

def get_multiScale_bias(karr,kc,kdis,alpha):

    """
    Corrects for:
    (1) The scalar bias induced by Arevalo+ 2012
    (2) The non-scalar bias induced by a beam *and* use of Arevalo+ 2012
    --- In order for this to be done, some spectral index, alpha MUST be assumed! ---
    --- where P(k) $\propto$ k**-alpha                                              ---
    Note that there is a correction for the power spectrum of the beam (PSF) within THIS term.
    If you want to correct for the PSF and the bias then these terms cancel! In light of this,
    I've added a keyword to allow you to ignore the PS term.
    ---------------------------------------------------------------------------------
    karr    : :%(dtypes)s
        The array of k (wavenumber) values at which to calculate the total bias
    kc      : float
        np.exp(-kc/k) cutoff for low frequencies
    kdis    : float
        np.exp(-k/kdis) cutoff for high frequencies
    alpha   : float 
        The assumed spectral index (convention given above)
    [ignPS] : bool
        As mentioned above, allows you to ignore the PSF power spectrum term with respect
        to this bias. Given that you want to correct for the PSF power spectrum, they will
        cancel each other out when correcting your measured power spectrum. You can bypass
        the additional calculations by ignoring the PSF term here.

    """


    aeff    = kc / karr - alpha - karr/kdis
    
    ndim    = 2
    lilg    = (ndim/2 + 2 - aeff/2)
    p1      = 2**(aeff/2)
    p2      = sps.gamma(lilg) / sps.gamma(ndim/2 + 2)
    
    bias    = p1*p2

    return bias

def get_multiScale_bias_v2(karr,kc,kdis,alpha,eta_d,eta_c):

    """
    Corrects for:
    (1) The scalar bias induced by Arevalo+ 2012
    (2) The non-scalar bias induced by a beam *and* use of Arevalo+ 2012
    --- In order for this to be done, some spectral index, alpha MUST be assumed! ---
    --- where P(k) $\propto$ k**-alpha                                              ---
    Note that there is a correction for the power spectrum of the beam (PSF) within THIS term.
    If you want to correct for the PSF and the bias then these terms cancel! In light of this,
    I've added a keyword to allow you to ignore the PS term.
    ---------------------------------------------------------------------------------
    karr    : :%(dtypes)s
        The array of k (wavenumber) values at which to calculate the total bias
    kc      : float
        np.exp(-kc/k) cutoff for low frequencies
    kdis    : float
        np.exp(-k/kdis) cutoff for high frequencies
    alpha   : float 
        The assumed spectral index (convention given above)
    [ignPS] : bool
        As mentioned above, allows you to ignore the PSF power spectrum term with respect
        to this bias. Given that you want to correct for the PSF power spectrum, they will
        cancel each other out when correcting your measured power spectrum. You can bypass
        the additional calculations by ignoring the PSF term here.

    """


    aeff    = eta_c*(kc / karr)**(eta_c) - alpha - eta_d* (karr/kdis)**(eta_d)
    
    ndim    = 2
    lilg    = (ndim/2 + 2 + aeff/2)
    flipind = (lilg < 2)
    if np.any(flipind):
        lilg[flipind] = 4-lilg[flipind]

    p1      = 2**(-aeff/2)
    p2      = sps.gamma(lilg) / sps.gamma(ndim/2 + 2)
    
    bias    = p1*p2

    #print(p1)
    #print(p2)
    #import pdb;pdb.set_trace()

    return bias

def get_cutoff_dis_bias_old(kmeas,kc,kdis,alpha,eta_d,eta_c,kmin=3e-5,kmax=1e0,nk=500):

    """
    Corrects for:
    (1) The scalar bias induced by Arevalo+ 2012
    (2) The non-scalar bias induced by a beam *and* use of Arevalo+ 2012
    --- In order for this to be done, some spectral index, alpha MUST be assumed! ---
    --- where P(k) $\propto$ k**-alpha                                              ---
    Note that there is a correction for the power spectrum of the beam (PSF) within THIS term.
    If you want to correct for the PSF and the bias then these terms cancel! In light of this,
    I've added a keyword to allow you to ignore the PS term.
    ---------------------------------------------------------------------------------
    karr    : :%(dtypes)s
        The array of k (wavenumber) values at which to calculate the total bias
    kc      : float
        np.exp(-kc/k) cutoff for low frequencies
    kdis    : float
        np.exp(-k/kdis) cutoff for high frequencies
    alpha   : float 
        The assumed spectral index (convention given above)
    [ignPS] : bool
        As mentioned above, allows you to ignore the PSF power spectrum term with respect
        to this bias. Given that you want to correct for the PSF power spectrum, they will
        cancel each other out when correcting your measured power spectrum. You can bypass
        the additional calculations by ignoring the PSF term here.

    """

    karr    = np.logspace(np.log10(kmin),np.log10(kmax),nk)
    dk      = np.hstack([karr[0],np.diff(karr)])
    #kmeas   = np.logspace(np.log10(kmin)+1,np.log10(kmax)-1,nmeas)
    dlk     = (kmax/kmin)**(1/(nk-1))
    PS      = np.exp(-(kc / karr)**(eta_c))*karr**(-alpha)*np.exp(-(karr/kdis)**(eta_d))

    bias    = []
    for k_r in kmeas:
        #sig       = 1./(np.sqrt(2.*np.pi**2)*k_r)
        Fkernel   = 2*(karr/k_r)**2
        PSkern    = np.exp(-Fkernel) # (e^x)^2 = e^(2x) ...blah blah.
        dPdisdk   = -eta_d* (karr/kdis)**(eta_d)
        dPcutdk   = eta_c*(kc / karr)**(eta_c)
        aeff      = dPcutdk - alpha + dPdisdk - Fkernel # d ln(P) / d ln(k)
        Integr    = np.sum(PS*PSkern*2*np.pi*karr*dk)
        IntExp    = 2 + aeff
        myPS      = np.exp(-(kc / k_r)**(eta_c))*k_r**(-alpha)*np.exp(-(k_r/kdis)**(eta_d))
        Integ     = (dlk**(IntExp/2) - dlk**(-IntExp/2))/IntExp
        Integrand = (PS*2 / (karr) ) *Integ
        bi        = np.isnan(Integrand) + np.isinf(Integrand)
        Integrand[bi] = 0.0
        Integral  = np.sum(Integrand)
        Approx    = myPS * k_r**2 / 2
        bias.append(Approx/Integral)

    #aeff    = eta_c*(kc / karr)**(eta_c-1) - alpha - eta_d* (karr/kdis)**(eta_d-1)
    
    import pdb;pdb.set_trace()

    return bias

def get_cutoff_dis_bias(kmeas,kc,kdis,alpha,eta_d,eta_c,kmin=3e-5,kmax=1e0,nk=500):

    """
    Corrects for:
    (1) The scalar bias induced by Arevalo+ 2012
    (2) The non-scalar bias induced by a beam *and* use of Arevalo+ 2012
    --- In order for this to be done, some spectral index, alpha MUST be assumed! ---
    --- where P(k) $\propto$ k**-alpha                                              ---
    Note that there is a correction for the power spectrum of the beam (PSF) within THIS term.
    If you want to correct for the PSF and the bias then these terms cancel! In light of this,
    I've added a keyword to allow you to ignore the PS term.
    ---------------------------------------------------------------------------------
    karr    : :%(dtypes)s
        The array of k (wavenumber) values at which to calculate the total bias
    kc      : float
        np.exp(-kc/k) cutoff for low frequencies
    kdis    : float
        np.exp(-k/kdis) cutoff for high frequencies
    alpha   : float 
        The assumed spectral index (convention given above)
    [ignPS] : bool
        As mentioned above, allows you to ignore the PSF power spectrum term with respect
        to this bias. Given that you want to correct for the PSF power spectrum, they will
        cancel each other out when correcting your measured power spectrum. You can bypass
        the additional calculations by ignoring the PSF term here.

    """

    karr    = np.logspace(np.log10(kmin),np.log10(kmax),nk)
    dk      = np.hstack([karr[0],np.diff(karr)])
    #kmeas   = np.logspace(np.log10(kmin)+1,np.log10(kmax)-1,nmeas)
    dlk     = (kmax/kmin)**(1/(nk-1))
    PS      = np.exp(-(kc / karr)**(eta_c))*karr**(-alpha)*np.exp(-(karr/kdis)**(eta_d))

    bias    = []
    for k_r in kmeas:
        #sig       = 1./(np.sqrt(2.*np.pi**2)*k_r)
        Fkernel   = 2*(karr/k_r)**2
        PSkern    = np.exp(-Fkernel) # (e^x)^2 = e^(2x) ...blah blah.
        dPdisdk   = -eta_d* (karr/kdis)**(eta_d)
        dPcutdk   = eta_c*(kc / karr)**(eta_c)
        aeff      = 4 + dPcutdk - alpha + dPdisdk - Fkernel # d ln(P) / d ln(k)
        Lintegr   = PS*PSkern*4*karr*dk * (karr/k_r)**4
        Integr    = np.sum(Lintegr)
        Linteg2   = np.hstack([0,PS[:-1]*PSkern[:-1]*2*karr[:-1]*dk[1:] * (karr[:-1]/k_r)**4])
        Integ2    = np.sum(Lintegr+Lintegr)/2
        IntExp    = 2 + aeff
        myPS      = np.exp(-(kc / k_r)**(eta_c))*k_r**(-alpha)*np.exp(-(k_r/kdis)**(eta_d))
        Integ     = (dlk**(IntExp/2) - dlk**(-IntExp/2))/IntExp
        Integrand = (PS*2 * karr**2 / k_r**4 ) *Integ
        bi        = np.isnan(Integrand) + np.isinf(Integrand)
        Integrand[bi] = 0.0
        Integral  = np.sum(Integrand)
        Approx    = myPS * k_r**2 / 2

        ###########################################################################
        # I really would have thought that the power law calculation would have   #
        # better, but it's not the case...for now.                                #
        ###########################################################################
        
        bias.append(Integr/Approx)

    #aeff    = eta_c*(kc / karr)**(eta_c-1) - alpha - eta_d* (karr/kdis)**(eta_d-1)
    
    #import pdb;pdb.set_trace()

    return bias

