using ForwardDiff: Dual, Partials, jacobian
# using FFTW: ifft
ForwardDiff.can_dual(::Type{Complex{Float64}}) = true

tagtype(::Dual{T,V,N}) where {T,V,N} = T

include("utils/helperfunctions.jl")
include("utils/helperfunctions_AD.jl")

"""
spectralDCM.jl

Main functions to compute a spectral DCM.

transferfunction : computes transfer function of neuronal model as well as measurement model
csd_approx       : approximates CSD based on transfer functions
csd_fmri_mtf     :
diff             : computes Jacobian of model
csd_Q            : computes precision component prior (which erroneously is not used in the SPM12 code for fMRI signals, it is used for other modalities)
matlab_norm      : computes norm consistent with MATLAB's norm function (Julia's is different, at lest for matrices. Haven't tested vectors)
spm_logdet       : mimick SPM12's way to compute the logarithm of the determinant. Sometimes Julia's logdet won't work.
variationalbayes : main routine that computes the variational Bayes estimate of model parameters
"""


function transferfunction(ω, derivatives, params, indices)
    ∂f = derivatives(params[indices[:dspars]])
    ∂f∂x = ∂f[indices[:sts], indices[:sts]]
    ∂f∂u = ∂f[indices[:sts], indices[:u]]
    ∂g∂x = ∂f[indices[:m], indices[:sts]]

    F = eigen(∂f∂x)
    Λ = F.values
    V = F.vectors

    ∂g∂v = ∂g∂x*V
    ∂v∂u = V\∂f∂u              # u is external variable.

    nω = size(ω, 1)            # number of frequencies
    ng = size(∂g∂x, 1)         # number of outputs
    nu = size(∂v∂u, 2)         # number of inputs
    nk = size(V, 2)            # number of modes
    S = zeros(Complex{real(eltype(∂v∂u))}, nω, ng, nu)
    for j = 1:nu
        for i = 1:ng
            for k = 1:nk
                # transfer functions (FFT of kernel)
                Sk = (1im*2*pi*ω .- Λ[k]).^-1
                S[:,i,j] .+= ∂g∂v[i,k]*∂v∂u[k,j]*Sk
            end
        end
    end

    return S
end


"""
    This function implements equation 2 of the spectral DCM paper, Friston et al. 2014 "A DCM for resting state fMRI".
    Note that nomenclature is taken from SPM12 code and it does not seem to coincide with the spectral DCM paper's nomenclature. 
    For instance, Gu should represent the spectral component due to external input according to the paper. However, in the code this represents
    the hidden state fluctuations (which are called Gν in the paper).
    Gn in the code corresponds to Ge in the paper, i.e. the observation noise. In the code global and local components are defined, no such distinction
    is discussed in the paper. In fact the parameter γ, corresponding to local component is not present in the paper.
"""
function csd_approx(ω, derivatives, params, params_idx)
    # priors of spectral parameters
    # ln(α) and ln(β), region specific fluctuations: ln(γ)
    nω = length(ω)
    nd = length(params_idx[:lnγ])
    α = params[params_idx[:lnα]]
    β = params[params_idx[:lnβ]]
    γ = params[params_idx[:lnγ]]
    
    # define function that implements spectra given in equation (2) of the paper "A DCM for resting state fMRI".

    # neuronal fluctuations, intrinsic noise (Gu) (1/f or AR(1) form)
    G = ω.^(-exp(α[2]))    # spectrum of hidden dynamics
    G /= sum(G)
    Gu = zeros(eltype(G), nω, nd, nd)
    Gn = zeros(eltype(G), nω, nd, nd)
    for i = 1:nd
        Gu[:, i, i] .+= exp(α[1])*G
    end
    # region specific observation noise (1/f or AR(1) form)
    G = ω.^(-exp(β[2])/2)
    G /= sum(G)
    for i = 1:nd
        Gn[:,i,i] .+= exp(γ[i])*G
    end

    # global components
    for i = 1:nd
        for j = i:nd
            Gn[:,i,j] .+= exp(β[1])*G
            Gn[:,j,i] = Gn[:,i,j]
        end
    end
    S = transferfunction(ω, derivatives, params, params_idx)   # This is K(ω) in the equations of the spectral DCM paper.

    # predicted cross-spectral density
    G = zeros(eltype(S), nω, nd, nd);
    for i = 1:nω
        G[i,:,:] = S[i,:,:]*Gu[i,:,:]*S[i,:,:]'
    end

    return G + Gn
end


function csd_approx_lfp(ω, derivatives, params, params_idx)
    # priors of spectral parameters
    # ln(α) and ln(β), region specific fluctuations: ln(γ)
    nω = length(ω)
    nd = length(params_idx[:lnγ])
    α = reshape(params[params_idx[:lnα]], nd, nd)
    β = params[params_idx[:lnβ]]
    γ = params[params_idx[:lnγ]]

    # define function that implements spectra given in equation (2) of the paper "A DCM for resting state fMRI".
    Gu = zeros(eltype(α), nω, nd)   # spectrum of neuronal innovations or intrinsic noise or system noise
    Gn = zeros(eltype(β), nω)   # global spectrum of channel noise or observation noise or external noise
    Gs = zeros(eltype(γ), nω)   # region specific spectrum of channel noise or observation noise or external noise
    for i = 1:nd
        Gu[:, i] .+= exp(α[1, i]) .* ω.^(-exp(α[2, i]))
    end
    # global components and region specific observation noise (1/f or AR(1) form)
    Gn = exp(β[1] - 2) * ω.^(-exp(β[2]))
    Gs = exp(γ[1] - 2) * ω.^(-exp(γ[2]))  # this is really oddly implemented in SPM12. Completely unclear how this should be region specific

    S = transferfunction(ω, derivatives, params, params_idx)   # This is K(ω) in the equations of the spectral DCM paper.

    # predicted cross-spectral density
    G = zeros(eltype(S), nω, nd, nd);
    for i = 1:nω
        G[i,:,:] = S[i,:,:]*diagm(Gu[i,:])*S[i,:,:]'
    end

    for i = 1:nd
        G[:,i,i] += Gs
        for j = 1:nd
            G[:,i,j] += Gn
        end
    end

    return G
end

@views function csd_mtf(freqs, p, derivatives, params, params_idx, modality)   # alongside the above realtes to spm_csd_fmri_mtf.m
    if modality == "fMRI"
        G = csd_approx(freqs, derivatives, params, params_idx)

        dt = 1/(2*freqs[end])
        # the following two steps are very opaque. They are taken from the SPM code but it is unclear what the purpose of this transformation and back-transformation is
        # in particular it is also unclear why the order of the MAR is reduced by 1. My best guess is that this procedure smoothens the results.
        # But this does not correspond to any equation in the papers nor is it commented in the SPM12 code. NB: Friston conferms that likely it is
        # to make y well behaved.
        mar = csd2mar(G, freqs, dt, p-1)
        y = mar2csd(mar, freqs)    
    elseif modality == "LFP"
        y = csd_approx_lfp(freqs, derivatives, params, params_idx)
    end
    if real(eltype(y)) <: Dual
        y_vals = Complex.((p->p.value).(real(y)), (p->p.value).(imag(y)))
        y_part = (p->p.partials).(real(y)) + (p->p.partials).(imag(y))*im
        y = map((x1, x2) -> Dual{tagtype(real(y)[1]), ComplexF64, length(x2)}(x1, Partials(Tuple(x2))), y_vals, y_part)
    end
    return y
end


"""
    function setup_sDCM(data, stateevolutionmodel, initcond, csdsetup, priors, hyperpriors, params_idx)

    Interface function to performs variational inference to fit model parameters to empirical cross spectral density.
    The current implementation provides a Variational Laplace fit (see function above `variationalbayes`).

    Arguments:
    - `data`        : dataframe with column names corresponding to the regions of measurement.
    - `model`       : MTK model, including state evolution and measurement.
    - `initcond`    : dictionary of initial conditions, numerical values for all states
    - `csdsetup`    : dictionary of parameters required for the computation of the cross spectral density
    -- `dt`         : sampling interval
    -- `freq`       : frequencies at which to evaluate the CSD
    -- `p`          : order parameter of the multivariate autoregression model
    - `priors`      : dataframe of parameters with the following columns:
    -- `name`       : corresponds to MTK model name
    -- `mean`       : corresponds to prior mean value
    -- `variance`   : corresponds to the prior variances
    - `hyperpriors` : dataframe of parameters with the following columns:
    -- `Πλ_pr`      : prior precision matrix for λ hyperparameter(s)
    -- `μλ_pr`      : prior mean(s) for λ hyperparameter(s)
    - `params_idx`  : indices to separate model parameters from other parameters. Needed for the computation of AD gradient.
    - `modality`    : which modality? Currently fMRI and LFP is available.
"""
function setup_sDCM(data, model, initcond, csdsetup, priors, hyperpriors, params_idx, modality)
    # compute cross-spectral density
    dt = csdsetup[:dt];              # order of MAR. Hard-coded in SPM12 with this value. We will use the same for now.
    ω = csdsetup[:freq];             # frequencies at which the CSD is evaluated
    p = csdsetup[:p];                # order of MAR
    mar = mar_ml(Matrix(data), p);   # compute MAR from time series y and model order p
    y_csd = mar2csd(mar, ω, dt^-1);  # compute cross spectral densities from MAR parameters at specific frequencies freqs, dt^-1 is sampling rate of data
    if modality == "LFP"
        vars = matread("speedandaccuracy/matlab_cmc.mat");
        y_csd = vars["csd"]
    end

    jac_fg = generate_jacobian(model, expression = Val{false})[1]   # compute symbolic jacobian.

    statevals = [v for v in values(initcond)]
    derivatives = par -> jac_fg(statevals, addnontunableparams(par, model), t)

    μθ_pr = vecparam(OrderedDict(priors.name .=> priors.mean))            # note: μθ_po is posterior and μθ_pr is prior
    Σθ_pr = diagm(vecparam(OrderedDict(priors.name .=> priors.variance)))

    ### Collect prior means and covariances ###
    if haskey(hyperpriors, :Q)
        Q = hyperpriors[:Q];
    else
        Q = csd_Q(y_csd);                 # compute functional connectivity prior Q. See Friston etal. 2007 Appendix A
    end
    nq = 1                            # TODO: this is hard-coded, need to make this compliant with csd_Q
    nh = size(Q, 3)                   # number of precision components (this is the same as above, but may differ)

    f = params -> csd_mtf(ω, p, derivatives, params, params_idx, modality)

    np = length(μθ_pr)     # number of parameters
    ny = length(y_csd)     # total number of response variables

    # variational laplace state variables
    vlstate = VLState(
        0,             # iter
        -4,            # log ascent rate
        [-Inf],        # free energy
        [],            # delta free energy
        hyperpriors[:μλ_pr],    # metaparameter, initial condition. TODO: why are we not just using the prior mean?
        zeros(np),     # parameter estimation error ϵ_θ
        [zeros(np), hyperpriors[:μλ_pr]],      # memorize reset state
        μθ_pr,         # parameter posterior mean
        Σθ_pr,         # parameter posterior covariance
        zeros(np),
        zeros(np, np)
    )

    # variational laplace setup
    vlsetup = VLSetup(
        f,                    # function that computes the cross-spectral density at fixed point 'initcond'
        y_csd,                # empirical cross-spectral density
        1e-1,                 # tolerance
        [np, ny, nq, nh],     # number of parameters, number of data points, number of Qs, number of hyperparameters
        [μθ_pr, hyperpriors[:μλ_pr]],          # parameter and hyperparameter prior mean
        [inv(Σθ_pr), hyperpriors[:Πλ_pr]],     # parameter and hyperparameter prior precision matrices
        Q                                      # components of data precision matrix
    )
    return (vlstate, vlsetup)
end


"""
    variationalbayes(idx_A, y, derivatives, w, V, p, priors, niter)

    Computes parameter estimation using variational Laplace that is to a large extend equivalent to the SPM12 implementation
    and provides the exact same values.

    Arguments:
    - `idx_A`: indices of connection weight parameter matrix A in model Jacobian
    - `y`: empirical cross-spectral density (input data)
    - `derivatives`: jacobian of model as well as gradient of observer function
    - `w`: fequencies at which to estimate cross-spectral densities
    - `V`: projection matrix from full parameter space to reduced space that removes parameters with zero variance prior
    - `p`: order of multivariate autoregressive model for estimation of cross-spectral densities from data
    - `priors`: Bayesian priors, mean and variance thereof. Laplace approximation assumes Gaussian distributions
    - `niter`: number of iterations of the optimization procedure
"""
function run_sDCM_iteration!(state::VLState, setup::VLSetup)
    μθ_po = state.μθ_po

    λ = state.λ
    v = state.v
    ϵ_θ = state.ϵ_θ
    dFdθ = state.dFdθ
    dFdθθ = state.dFdθθ

    f = setup.model_at_x0
    y = setup.y_csd              # cross-spectral density
    (np, ny, nq, nh) = setup.systemnums
    (μθ_pr, μλ_pr) = setup.systemvecs
    (Πθ_pr, Πλ_pr) = setup.systemmatrices
    Q = setup.Q

    dfdp = jacobian(f, μθ_po)
    norm_dfdp = matlab_norm(dfdp, Inf);
    revert = isnan(norm_dfdp) || norm_dfdp > exp(32);

    if revert && state.iter > 1
        for i = 1:4
            # reset expansion point and increase regularization
            v = min(v - 2, -4);
            t = exp(v - logdet(dFdθθ)/np)

            # E-Step: update
            if t > exp(16)
                ϵ_θ = ϵ_θ - dFdθθ \ dFdθ    # -inv(dfdx)*f
            else
                ϵ_θ = ϵ_θ + expv(t, dFdθθ, dFdθθ \ dFdθ) - dFdθθ \ dFdθ   # (expm(dfdx*t) - I)*inv(dfdx)*f
            end

            μθ_po = μθ_pr + ϵ_θ

            dfdp = jacobian(f, μθ_po)

            # check for stability
            norm_dfdp = matlab_norm(dfdp, Inf);
            revert = isnan(norm_dfdp) || norm_dfdp > exp(32);

            # break
            if ~revert
                break
            end
        end
    end

    ϵ = reshape(y - f(μθ_po), ny)                   # error
    J = - dfdp   # Jacobian, unclear why we have a minus sign. Helmut: comes from deriving a Gaussian. 

    ## M-step: Fisher scoring scheme to find h = max{F(p,h)} // comment from MATLAB code
    P = zeros(eltype(J), size(Q))
    PΣ = zeros(eltype(J), size(Q))
    JPJ = zeros(real(eltype(J)), size(J, 2), size(J, 2), size(Q, 3))
    dFdλ = zeros(eltype(J), nh)
    dFdλλ = zeros(real(eltype(J)), nh, nh)
    local iΣ, Σλ_po, Σθ_po, ϵ_λ
    for m = 1:8   # 8 seems arbitrary. Numbers of iterations taken from SPM12 code.
        iΣ = zeros(eltype(J), ny, ny)
        for i = 1:nh
            iΣ .+= Q[:, :, i] * exp(λ[i])
        end

        Pp = real(J' * iΣ * J)    # in MATLAB code 'real()' is applied to the resulting matrix product, why is this okay?
        Σθ_po = inv(Pp + Πθ_pr)

        for i = 1:nh
            P[:,:,i] = Q[:,:,i]*exp(λ[i])
            PΣ[:,:,i] = iΣ \ P[:,:,i]
            JPJ[:,:,i] = real(J'*P[:,:,i]*J)      # in MATLAB code 'real()' is applied (see also some lines above)
        end
        for i = 1:nh
            dFdλ[i] = (tr(PΣ[:,:,i])*nq - real(dot(ϵ, P[:,:,i], ϵ)) - tr(Σθ_po * JPJ[:,:,i]))/2
            for j = i:nh
                dFdλλ[i, j] = -real(tr(PΣ[:,:,i] * PΣ[:,:,j]))*nq/2
                dFdλλ[j, i] = dFdλλ[i, j]
            end
        end

        ϵ_λ = λ - μλ_pr
        dFdλ = dFdλ - Πλ_pr*ϵ_λ
        dFdλλ = dFdλλ - Πλ_pr
        Σλ_po = inv(-dFdλλ)

        t = exp(4 - spm_logdet(dFdλλ)/length(λ))
        # E-Step: update
        if t > exp(16)
            dλ = -real(dFdλλ \ dFdλ)
        else
            idFdλλ = inv(dFdλλ)
            dλ = real(exponential!(t * dFdλλ) * idFdλλ*dFdλ - idFdλλ*dFdλ)   # (expm(dfdx*t) - I)*inv(dfdx)*f ~~~ could also be done with expv but doesn't work with Dual.
        end

        dλ = [min(max(x, -1.0), 1.0) for x in dλ]      # probably precaution for numerical instabilities?
        λ = λ + dλ

        dF = dot(dFdλ, dλ)

        # NB: it is unclear as to whether this is being reached. In this first tests iterations seem to be 
        # trapped in a periodic orbit jumping around between 1250 and 940. At that point the results become
        # somewhat arbitrary. The iterations stop at 8, whatever the last value of iΣ etc. is will be carried on.
        if real(dF) < 1e-2
            break
        end
    end

    ## E-Step with Levenberg-Marquardt regularization    // comment from MATLAB code
    L = zeros(real(eltype(iΣ)), 3)
    L[1] = (real(logdet(iΣ))*nq - real(dot(ϵ, iΣ, ϵ)) - ny*log(2pi))/2
    L[2] = (logdet(Πθ_pr * Σθ_po) - dot(ϵ_θ, Πθ_pr, ϵ_θ))/2
    L[3] = (logdet(Πλ_pr * Σλ_po) - dot(ϵ_λ, Πλ_pr, ϵ_λ))/2
    F = sum(L)

    if F > state.F[end] || state.iter < 3
        # accept current state
        state.reset_state = [ϵ_θ, λ]
        append!(state.F, F)
        state.Σθ_po = Σθ_po
        # Conditional update of gradients and curvature
        dFdθ  = -real(J' * iΣ * ϵ) - Πθ_pr * ϵ_θ    # check sign
        dFdθθ = -real(J' * iΣ * J) - Πθ_pr
        # decrease regularization
        v = min(v + 1/2, 4);
    else
        # reset expansion point
        ϵ_θ, λ = state.reset_state
        # and increase regularization
        v = min(v - 2, -4);
    end

    # E-Step: update
    t = exp(v - spm_logdet(dFdθθ)/np)
    if t > exp(16)
        dθ = - inv(dFdθθ) * dFdθ     # -inv(dfdx)*f
    else
        dθ = exponential!(t * dFdθθ) * inv(dFdθθ) * dFdθ - inv(dFdθθ) * dFdθ     # (expm(dfdx*t) - I)*inv(dfdx)*f
    end

    ϵ_θ += dθ
    state.μθ_po = μθ_pr + ϵ_θ
    dF = dot(dFdθ, dθ);

    state.v = v
    state.ϵ_θ = ϵ_θ
    state.λ = λ
    state.dFdθθ = dFdθθ
    state.dFdθ = dFdθ
    append!(state.dF, dF)

    return state
end