# Hemodynamics and bold signal all in one

function hemodynamics!(J, f, x, decay, transit, ϵ)
    """
    Components of x are:
    x[:,1] - neural activity: x
    x[:,2] - vascular signal: s
    x[:,3] - rCBF: ln(f)
    x[:,4] - venous volume: ln(ν)
    x[:,5] - deoxyhemoglobin (dHb): ln(q)
    x[:,6] - BOLD signal
    decay, transit - free parameters, set to 0 for standard parameters.

    This function implements the hymodynamics model (balloon model and neurovascular state eq.) described in: 

        Stephan KE, Weiskopf N, Drysdale PM, Robinson PA, Friston KJ (2007)
        Comparing hemodynamic models with DCM. NeuroImage 38: 387-401.

        adapted from spm_fx_fmri.m in SPM12 by Karl Friston & Klaas Enno Stephan
    """

    #= hemodynamic parameters
        H(1) - signal decay                                   d(ds/dt)/ds)
        H(2) - autoregulation                                 d(ds/dt)/df)
        H(3) - transit time                                   (t0)
        H(4) - exponent for Fout(v)                           (alpha)
        H(5) - resting oxygen extraction                      (E0)
    =#
    H = [0.64, 0.32, 2.00, 0.32, 0.4]

    # exponentiation of hemodynamic state variables
    x[:, 3:5] = exp(x[:, 3:5])

    # signal decay
    κ = H[1]*exp(decay)

    # transit time
    τ = H[3]*exp(transit)

    # Fout = f(v) - outflow
    fv = x[:, 3].^(1/H[4])

    # e = f(f) - oxygen extraction
    ff = (1 - (1 - H[5]).^(1.0./x[:, 3]))/H[5]

    # implement differential state equation f = dx/dt (hemodynamic)
    f[:, 2] = x[:, 1] - κ.*x[:, 2] - H[2]*(x[:, 3] - 1)   # Corresponds to eq (9)
    f[:, 3] = x[:, 2]./x[:, 3]  # Corresponds to eq (10), note the added logarithm (see doc string)
    f[:, 4] = (x[:, 3] - fv)./(τ.*x[:, 4])    # Corresponds to eq (8), note the added logarithm (see doc string)
    f[:, 5] = (ff.*x[:, 3] - fv.*x[:, 5]./x[:, 4])./(τ.*x[:, 5])  # Corresponds to eq (8), note the added logarithm (see doc string)

        # NB: Biophysical constants for 1.5T scanners
    # Time to echo
    TE  = 0.04
    # resting venous volume (%)
    V0  = 4    
    # slope r0 of intravascular relaxation rate R_iv as a function of oxygen 
    # saturation S:  R_iv = r0*[(1 - S)-(1 - S0)] (Hz)
    r0  = 25
    # frequency offset at the outer surface of magnetized vessels (Hz)
    nu0 = 40.3
    # resting oxygen extraction fraction
    E0  = 0.4
    # estimated region-specific ratios of intra- to extra-vascular signal 
    ep  = exp(ϵ)

    # -Coefficients in BOLD signal model
    k1  = 4.3*nu0*E0*TE;
    k2  = ep*r0*E0*TE;
    k3  = 1 - ep;

    # -Output equation of BOLD signal model
    ν   = exp(x[:,4])
    q   = exp(x[:,5])
    f[:,6]   = V0*(k1 - k1*q + k2 - k2*q./ν + k3 - k3*ν)   # BOLD signal

    nd = size(x)[1]
    J[1:nd,1:nd] = Matrix(-sd*I, nd, nd)
    J[1:nd,nd+1:2nd] = diagm(-H(2)*x[:,3])
    J[nd+1:2nd,1:nd] = diagm( x[:,3].^-1)
    J[nd+1:2nd,nd+1:2nd] = diagm(-x[:,2]./x[:,3])
    J[2nd+1:3nd,nd+1:2nd] = diagm( x[:,3]./(tt.*x[:,4]))
    J[2nd+1:3nd,2nd+1:3nd] = diagm(-x[:,4].^(1/H[4] - 1)./(tt*H[4]) - (x[:,4].^-1 .*(x[:,3] - x[:,4].^(1/H[4])))./tt)
    J[3nd+1:4nd,nd+1:2nd] = diagm((x[:,3] + log(1 - H[5]).*(1 - H[5]).^(1.0/x[:,3]) - x[:,3].*(1 - H[5]).^(x[:,3].^-1))./(tt.*x[:,5]*H[5]))
    J[3nd+1:4nd,2nd+1:3nd] = diagm((x[:,4].^(1/H[4] - 1)*(H[4] - 1))./(tt*H[4]))
    J[3nd+1:4nd,3nd+1:4nd] = diagm((x[:,3]./x[:,5]).*((1 - H[5]).^(1.0/x[:,3]) - 1)./(tt*H[5]))
    J[4nd+1:5nd,2nd+1:3nd] = diagm(-V0*(k3.*ν - k2.*q./ν))
    J[4nd+1:5nd,3nd+1:4nd] = diagm(-V0*(k1.*q + k2.*q./ν))
    
    return f
end


# Simulate the BOLD signal
function g_fmri(x, ϵ)
    """
    Simulated BOLD response to input
    FORMAT [g,dgdx] = g_fmri(x, ϵ)
    g          - BOLD response (%)
    x          - hemodynamic state vector, same as above.
    ϵ          - free parameter (note also here as above, actually ln(ϵ)), ratio of intra- to extra-vascular components

    This function implements the BOLD signal model described in: 

    Stephan KE, Weiskopf N, Drysdale PM, Robinson PA, Friston KJ (2007)
    Comparing hemodynamic models with DCM. NeuroImage 38: 387-401.
 
    adapted from spm_gx_fmri.m in SPM12 by Karl Friston & Klaas Enno Stephan

    NB: Biophysical constants for 1.5T scanners:
    TE  = 0.04
    V0  = 4    
    r0  = 25
    nu0 = 40.3
    E0  = 0.4
    """

    # NB: Biophysical constants for 1.5T scanners
    # Time to echo
    TE  = 0.04
    # resting venous volume (%)
    V0  = 4    
    # slope r0 of intravascular relaxation rate R_iv as a function of oxygen 
    # saturation S:  R_iv = r0*[(1 - S)-(1 - S0)] (Hz)
    r0  = 25
    # frequency offset at the outer surface of magnetized vessels (Hz)
    nu0 = 40.3
    # resting oxygen extraction fraction
    E0  = 0.4
    # estimated region-specific ratios of intra- to extra-vascular signal 
    ep  = exp(ϵ)

    # -Coefficients in BOLD signal model
    k1  = 4.3*nu0*E0*TE;
    k2  = ep*r0*E0*TE;
    k3  = 1 - ep;

    # -Output equation of BOLD signal model
    ν   = exp(x[:,4])
    q   = exp(x[:,5])
    g   = V0*(k1 - k1*q + k2 - k2*q./ν + k3 - k3*ν)


    return g
end
