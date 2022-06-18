#= Define notational equivalences between SPM12 and this code:

# the following two precision matrices will not be updated by the code,
# they belong to the assumed prior distribution p (fixed, but what if it isn't
# the ground truth?)
ipC = Πθ_p   # precision matrix of prior of parameters p(θ)
ihC = Πλ_p   # precision matrix of prior of hyperparameters p(λ)

Variational distribution parameters:
pE, Ep = θμ, μθ   # prior expectation of parameters (q(θ))
pC, Cp = θΣ, Σθ   # prior covariance of parameters (q(θ))
hE, Eh = λμ, μλ   # prior expectation of hyperparameters (q(λ))
hC, Ch = λΣ, Σλ   # prior covariance of hyperparameters (q(λ))

Σ, iΣ  # data covariance matrix (likelihood), and its inverse (precision of likelihood - use Π only for those precisions that don't change)
Q      # components of iΣ; definition: iΣ = sum(exp(λ)*Q)
=#

# compute Jacobian of rhs w.r.t. variable -> matrix exponential solution (use ExponentialUtilities.jl)
# -> use this numerical integration as solution to the diffeq to then differentiate solution w.r.t. parameters (like sensitivity analysis in Ma et al. 2021)
# -> that Jacobian is used in all the computations of the variational Bayes


# Define priors etc.
# Q, θμ, θΣ, λμ, λΣ

# pE.A = A/128; θμ?
using LinearAlgebra: Eigen
using ForwardDiff: Dual
using ForwardDiff: Partials
using ChainRules: _eigen_norm_phase_fwd!
using Serialization
ForwardDiff.can_dual(::Type{ComplexF64}) = true
Base.eps(z::Complex{T}) where {T<:AbstractFloat} = hypot(eps(real(z)), eps(imag(z)))
Base.signbit(x::Complex{T}) where {T<:AbstractFloat} = real(x) < 0

function LinearAlgebra.eigen(M::Matrix{Dual{T, P, N}}) where {T,P,N}
    np = length(M[1].partials)
    nd = size(M, 1)
    A = (p->p.value).(M)
    F = eigen(A)
    λ, V = F.values, F.vectors
    local ∂λ_agg, ∂V_agg
    for i = 1:np
        dA = (p->p.partials[i]).(M)
        tmp = V \ dA
        ∂K = tmp * V   # V^-1 * dA * V
        ∂Kdiag = @view ∂K[diagind(∂K)]
        ∂λ_tmp = eltype(λ) <: Real ? real.(∂Kdiag) : copy(∂Kdiag)
        ∂K ./= transpose(λ) .- λ
        fill!(∂Kdiag, 0)
        ∂V_tmp = mul!(tmp, V, ∂K)
        _eigen_norm_phase_fwd!(∂V_tmp, A, V)
        if i == 1
            ∂V_agg = ∂V_tmp
            ∂λ_agg = ∂λ_tmp
        else
            ∂V_agg = cat(∂V_agg, ∂V_tmp, dims=3)
            ∂λ_agg = cat(∂λ_agg, ∂λ_tmp, dims=2)
        end
    end
    ∂V = Array{Partials}(undef, nd, nd)
    ∂λ = Array{Partials}(undef, nd)
    for i = 1:nd
        ∂λ[i] = Partials(Tuple(∂λ_agg[i, :]))
        for j = 1:nd
            ∂V[i, j] = Partials(Tuple(∂V_agg[i, j, :]))
        end
    end
    return Eigen(Dual{T}.(F.values, ∂λ), Dual{T}.(F.vectors, ∂V))
end


function transferfunction(x, w, θμ, C, lnϵ, lndecay, lntransit)
    # compute transfer function of Volterra kernels, see fig 1 in friston2014
    # 1. compute jacobian w.r.t. f ; TODO: what is it with this "delay operator" that is set to 1 in "spm_fx_fmri.m"
    # J_x = jacobian(f, x0) # well, no need to perform this for a linear system... we already have it: θμ
    C /= 16.0   # TODO: unclear why it is devided by 16 but see spm_fx_fmri.m:49
    # 2. get jacobian of hemodynamics
    dx = similar(x[:, 2:end])
    J = hemodynamics!(dx, x[:, 2:end], x[:, 1], lndecay, lntransit)[2]
    θμ -= diagm(exp.(diag(θμ))/2 + diag(θμ))
    # if I eventually need also the change of variables rather than just the derivative then here is where to fix it!
    nd = size(θμ, 1)
    J_tot = [θμ zeros(nd, size(J, 2));   # add derivatives w.r.t. neural signal
             [Matrix(1.0I, size(θμ)); zeros(size(J)[1]-nd, size(θμ)[2])] J]

    dfdu = [C;
            zeros(size(J,1), size(C, 2))]

    F = eigen(J_tot)#, sortby=nothing, permute=true)
    Λ = F.values
    V = F.vectors

    # condition unstable eigenmodes
    # if max(w) > 1
    #     s = 1j*imag(s) + real(s) - exp(real(s));
    # else
    #     s = 1j*imag(s) + min(real(s),-1/32);
    # end

    # 3. get jacobian (??) of bold signal, just compute it as is done, but how is this a jacobian, it isn't! if anything it should be a gradient since the BOLD signal is scalar
    #TODO: implement numerical and compare with analytical: J_g = jacobian(bold, x0)
    dgdx = boldsignal(x, lnϵ)[2]
    dgdv = dgdx*V[end-size(dgdx,2)+1:end, :]     # TODO: not a clean solution, also not in the original code since it seems that the code really depends on the ordering of eigenvalues and respectively eigenvectors!
    dvdu = pinv(V)*dfdu

    nw = size(w,1)            # number of frequencies
    ng = size(dgdx,1)         # number of outputs
    nu = size(dfdu,2)         # number of inputs
    nk = size(V,2)            # number of modes
    S = zeros(Complex, nw, ng, nu)

    for j = 1:nu
        for i = 1:ng
            for k = 1:nk
                # transfer functions (FFT of kernel)
                Sk = (1im*2*pi*w .- Λ[k]).^-1
                S[:,i,j] .+= dgdv[i,k]*dvdu[k,j]*Sk
            end
        end
    end
    return S
end

function csd2mar(csd, w, dt, p)
    # TODO: investiagate why SymmetricToeplitz(ccf[1:p, i, j]) is not good to be used but instead need to use Toeplitz(ccf[1:p, i, j], ccf[1:p, j, i])
    # as is done in the original MATLAB code (confront comment there). ccf should be a symmetric matrix so there should be no difference between the
    # Toeplitz matrices but for the second jacobian (J[2], see for loop for i = 1:nJ in function diff) the computation produces subtle differences between
    # the two versions.

    dw = w[2] - w[1]
    w = w/dw
    ns = dt^-1
    N = ceil(Int64, ns/2/dw)
    gj = findall(x -> x > 0 && x < (N + 1), w)
    gi = gj .+ (ceil(Int64, w[1]) - 1)    # TODO: figure out what's the purpose of this!
    g = zeros(ComplexF64, N)

    # transform to cross-correlation function
    ccf = zeros(ComplexF64, N*2+1, size(csd,2), size(csd,3))
    for i = 1:size(csd, 2)
        for j = 1:size(csd, 3)
            g[gi] = csd[gj,i,j]
            f = ifft(g)
            f = ifft(vcat([0.0im; g; conj(g[end:-1:1])]))
            ccf[:,i,j] = real.(fftshift(f))*N*dw
        end
    end

    # MAR coefficients
    N = size(ccf,1)
    m = size(ccf,2)
    n = (N - 1) ÷ 2
    p = min(p, n - 1)
    ccf = ccf[(1:n) .+ n,:,:]
    A = zeros(m*p, m)
    B = zeros(m*p, m*p)
    for i = 1:m
        for j = 1:m
            A[((i-1)*p+1):i*p, j] = ccf[(1:p) .+ 1, i, j]
            B[((i-1)*p+1):i*p, ((j-1)*p+1):j*p] = Toeplitz(ccf[1:p, i, j], vcat(ccf[1,i,j], ccf[2:p, j, i]))  # SymmetricToeplitz(ccf[1:p, i, j])
        end
    end
    a = B\A

    Σ  = ccf[1,:,:] - A'*a   # noise covariance matrix
    lags = [-a[i:p:m*p, :] for i = 1:p]
    mar = Dict([("A", lags), ("Σ", Σ), ("p", p)])
    return mar
end

function mar2csd(mar, freqs, sf)
    Σ = mar["Σ"]
    p = mar["p"]
    A = mar["A"]
    nd = size(Σ, 1)
    w  = 2*pi*freqs/sf    # isn't it already transformed?? Is the original really in Hz? Also clearly freqs[end] is not the sampling frequency of the signal...
    nf = length(w)
	csd = zeros(ComplexF64, nf, nd, nd)
	for i = 1:nf
		af_tmp = I
		for k = 1:p
			af_tmp = af_tmp + A[k] * exp(-im * k * w[i])
		end
		iaf_tmp = inv(af_tmp)
		csd[i,:,:] = iaf_tmp * Σ * iaf_tmp'     # is this really the covariance or rather precision?!
	end
    csd = 2*csd/sf
    return csd
end

function mar2csd(mar, freqs)
    sf = 2*freqs[end]
    Σ = mar["Σ"]
    p = mar["p"]
    A = mar["A"]
    nd = size(Σ, 1)
    w  = 2pi*freqs/sf    # isn't it already transformed?? Is the original really in Hz? Also clearly freqs[end] is not the sampling frequency of the signal...
    nf = length(w)
	csd = zeros(ComplexF64, nf, nd, nd)
	for i = 1:nf
		af_tmp = I
		for k = 1:p
			af_tmp = af_tmp + A[k] * exp(-im * k * w[i])
		end
		iaf_tmp = inv(af_tmp)
		csd[i,:,:] = iaf_tmp * Σ * iaf_tmp'     # is this really the covariance or rather precision?!
	end
    csd = 2*csd/sf
    return csd
end

function csd_approx(x, w, θμ, C, α, β, γ, lnϵ, lndecay, lntransit)
    # priors of spectral parameters
    # ln(α) and ln(β), region specific fluctuations: ln(γ)
    nw = length(w)
    nd = size(θμ, 1)

    # define function that implements spectra given in equation (2) of "A DCM for resting state fMRI".

    # neuronal fluctuations (Gu) (1/f or AR(1) form)
    G = w.^(-exp(β[1]))   # spectrum of hidden dynamics
    G /= sum(G)
    Gu = zeros(typeof(G[1]), nw, nd, nd)
    Gn = zeros(typeof(G[1]), nw, nd, nd)
    for i = 1:nd
        Gu[:, i, i] .+= exp(α[1])*G
    end
    # region specific observation noise (1/f or AR(1) form)
    G = w.^(-exp(β[2])/2)
    G /= sum(G)
    for i = 1:nd
        Gn[:,i,i] .+= exp(γ[i])*G
    end

    # global components
    for i = 1:nd
        for j = i:nd
            Gn[:,i,j] .+= exp(α[2])*G
            Gn[:,j,i] = Gn[:,i,j]
        end
    end
    C = Matrix(I, nd, nd)
    S = transferfunction(x, w, θμ, C, lnϵ, lndecay, lntransit)

    # predicted cross-spectral density
    G = zeros(ComplexF64,nw,nd,nd);
    for i = 1:nw
        G[i,:,:] = S[i,:,:]*Gu[i,:,:]*S[i,:,:]'
    end

    return G + Gn
end

function csd_fmri_mtf(x, freqs, p, param)
    dim = size(x, 1)
    θμ = reshape(param[1:dim^2], dim, dim)
    C = param[(1+dim^2):(3+dim^2)]
    lntransit = param[(4+dim^2):(6+dim^2)]
    lndecay = param[7+dim^2]
    lnϵ = param[8+dim^2]
    α = param[[9+dim^2, 11+dim^2]]
    β = param[[10+dim^2, 12+dim^2]]
    γ = param[(13+dim^2):(15+dim^2)]
    G = csd_approx(x, freqs, θμ, C, α, β, γ, lnϵ, lndecay, lntransit)
    dt  = 1/(2*freqs[end])
    mar = csd2mar(G, freqs, dt, p-1)
    y = mar2csd(mar, freqs)
    return y
end

function diff(U, dx, f, param)
    nJ = size(U, 2)
    y0 = f(param)
    J = zeros(ComplexF64, nJ, size(y0, 1), size(y0, 2), size(y0, 3))
    for i = 1:nJ
        tmp_param = param .+ U[:, i]*dx
        y1 = f(tmp_param)
        J[i,:,:,:] = (y1 .- y0)/dx
    end
    return J, y0
end

function matlab_norm(A, p)
    if p == 1
        return maximum(vec(sum(abs.(A),dims=1)))
    elseif p == Inf
        return maximum(vec(sum(abs.(A),dims=2)))
    elseif p == 2
        print("Not implemented yet!\n")
        return NaN
    end
end


function spm_logdet(M)
    TOL = 1e-16
    s = diag(M)
    if sum(abs.(s)) != sum(abs.(M[:]))
        ~, s, ~ = svd(M)
    end
    return sum(log.(s[(s .> TOL) .& (s .< TOL^-1)]))
end

function csd_Q(csd)
    s = size(csd)
    Qn = length(csd)
    Q = zeros(ComplexF64, Qn, Qn);
    idx = CartesianIndices(csd)
    for Qi  = 1:Qn
        for Qj = 1:Qn
            if idx[Qi][1] == idx[Qj][1]
                Q[Qi,Qj] = csd[idx[Qi][1], idx[Qi][2], idx[Qj][2]]*csd[idx[Qi][1], idx[Qi][3], idx[Qj][3]]
            end
        end
    end
    Q = inv(Q .+ matlab_norm(Q, 1)/32*Matrix(I, size(Q)))   # TODO: MATLAB's and Julia's norm function are different! Reconciliate?
    return Q
    # the following routine is for situations where no Q is given apriori
    # Q = zeros(ny,ny,nr)
    # for i = 1:nr
    #     Q[((i-1)*ns+1):(i*ns), ((i-1)*ns+1):(i*ns), i] = Matrix(1.0I, ns, ns)
    # end

end


function variationalbayes(x, y, w, V, param, priors, niter)
    # extract priors
    Πθ_p = priors[1]
    Πλ_p = priors[2]
    λμ = vec(priors[3])

    # prep stuff
    p = Int(param[1])
    θμ = param[2:end]          # note: μθ is posterior and θμ is prior
    np = size(V, 2)            # number of parameters
    ny = length(y)             # total number of response variables
    ns = size(y, 1)            # number of samples
    nr = ny÷ns                 # number of response components
    nq = 1
    ϵ_θ = zeros(np)  # M.P - θμ # still need to figure out what M.P is for. It doesn't seem to be used further down the road in nlsi_GM, only at the very beginning when p is defined first. Then replace μθ with θμ above.
    μθ = θμ + V*ϵ_θ
    Q = zeros(ny,ny,nr)
    for i = 1:nr
        Q[((i-1)*ns+1):(i*ns), ((i-1)*ns+1):(i*ns), i] = Matrix(1.0I, ns, ns)
    end
    nh = size(Q,3)             # number of precision components (this is the same as above, but may differ)
    λ = 8 * ones(nh)

    dx = exp(-8)
    revert = false
    f_prep = param -> csd_fmri_mtf(x, w, p, param)

    # state variable
    F = -Inf
    F0 = F
    state = Dict("F"=>F, "ϵ_θ" => zeros(np), "λ" => λ, "μθ" => μθ, "Σθ" => inv(Πθ_p))
    v = -4   # log ascent rate
    criterion = [false, false, false, false]
    Σθ = similar(Πθ_p)
    Σλ = similar(Πθ_p)
    iΣ = zeros(ny, ny)
    ϵ_λ = similar(λμ)

    for k = 1:niter
        f = f_prep(μθ)
        dfdp = ForwardDiff.jacobian(f_prep, μθ)
        # dfdp, f = diff(V, dx, f_prep, μθ);
        dfdp = transpose(reshape(dfdp, np, ny))

        norm_dfdp = matlab_norm(dfdp, Inf);
        revert = isnan(norm_dfdp) || norm_dfdp > exp(32);

        if revert && k > 1
            for i = 1:4
                # reset expansion point and increase regularization
                v = min(v - 2,-4);
                t = exp(v - logdet(dFdpp)/np)

                # E-Step: update
                if t > exp(16)
                    ϵ_θ = state["ϵ_θ"] - inv(dFdpp)*dFdp    # -inv(dfdx)*f
                else
                    ϵ_θ = state["ϵ_θ"] + expv(t, dFdpp, inv(dFdpp)*dFdp) -inv(dFdpp)*dFdp   # (expm(dfdx*t) - I)*inv(dfdx)*f
                end

                μθ = θμ + V*ϵ_θ

                dfdp, f = diff(V, dx, f_prep, μθ);
                dfdp = transpose(reshape(dfdp, np, ny))

                # check for stability
                norm_dfdp = matlab_norm(dfdp, Inf);
                revert = isnan(norm_dfdp) || norm_dfdp > exp(32);
        
                # break
                if ~revert
                    break
                end
            end
        end


        ϵ = reshape(y - f, ny)                   # error value
        J = - dfdp   # Jacobian, unclear why we have a minus sign. Helmut: comes from deriving a Gaussian. 


        ## M-step: Fisher scoring scheme to find h = max{F(p,h)} // comment from MATLAB code
        for m = 1:8   # 8 seems arbitrary. Numbers of iterations taken from SPM12 code.
            iΣ = zeros(ny, ny)
            for i = 1:nh
                iΣ .+= Q[:,:,i]*exp(λ[i])
            end

            Σ = inv(iΣ)             # Julia requires conversion to dense matrix before inversion so just use dense from the get-go
            Pp = zeros(ComplexF64, size(Πθ_p))
            mul!(Pp, J', iΣ * J)    # in MATLAB code 'real()' is applied to the resulting matrix product, why?
            Pp = real(Pp)
            Σθ = inv(Pp + Πθ_p)

            P = similar(Q)
            PΣ = similar(Q)
            JPJ = zeros(size(Pp,1), size(Pp,2), size(Q,3))
            for i = 1:nh
                P[:,:,i] = Q[:,:,i]*exp(λ[i])
                PΣ[:,:,i] = P[:,:,i] * Σ
                JPJ[:,:,i] = real(J'*P[:,:,i]*J)      # in MATLAB code 'real()' is applied (see also some lines above), what's the rational?
            end

            dFdh = zeros(nh)
            dFdhh = zeros(nh, nh)
            for i = 1:nh
                dFdh[i] = (tr(PΣ[:,:,i])*nq - real(dot(ϵ,P[:,:,i],ϵ)) - tr(Σθ * JPJ[:,:,i]))/2
                for j = i:nh
                    dFdhh[i, j] = - tr(PΣ[:,:,i] * PΣ[:,:,j])*nq/2
                    dFdhh[j, i] = dFdhh[i, j]
                end
            end

            ϵ_λ = λ - λμ
            dFdh = dFdh - Πλ_p*ϵ_λ
            dFdhh = dFdhh - Πλ_p
            Σλ = inv(-dFdhh)

            t = exp(4 - spm_logdet(dFdhh)/length(λ))
            # E-Step: update
            if t > exp(16)
                dλ = -inv(dFdhh) * dFdh
            else
                dλ = expv(t, dFdhh, inv(dFdhh)*dFdh) -inv(dFdhh)*dFdh   # (expm(dfdx*t) - I)*inv(dfdx)*f
            end

            dλ = [min(max(x, -1.0), 1.0) for x in dλ]      # probably precaution for numerical instabilities?
            λ = λ + dλ

            dF = dot(dFdh,dλ)
            # NB: it is unclear as to whether this is being reached. In this first tests iterations seem to be 
            # trapped in a periodic orbit jumping around between 1250 and 940. At that point the results become
            # somewhat arbitrary. The iterations stop at 8, whatever the last value of iΣ etc. is will be carried on.
            if dF < 1e-2
                break
            end
        end

        ## E-Step with Levenberg-Marquardt regularization    // comment from MATLAB code
        L = zeros(3)
        L[1] = (logdet(iΣ)*nq  - real(ϵ' * iΣ * ϵ) - ny*log(2pi))/2
        L[2] = (logdet(Πθ_p * Σθ) - dot(ϵ_θ, Πθ_p, ϵ_θ))/2
        L[3] = (logdet(Πλ_p * Σλ) - dot(ϵ_λ, Πλ_p, ϵ_λ))/2;
        F = sum(L);

        if k == 1
            F0 = F
        end

        if F > state["F"] || k < 3
            # accept current state
            state["F"] = F
            state["ϵ_θ"] = ϵ_θ
            state["λ"] = λ
            state["Σθ"] = Σθ
            state["μθ"] = μθ
            # Conditional update of gradients and curvature
            dFdp  = -real(J' * iΣ * ϵ) - Πθ_p * ϵ_θ   # check sign!!
            dFdpp = -real(J' * iΣ * J) - Πθ_p
            # decrease regularization
            v = min(v + 1/2,4);
        else
            # reset expansion point
            ϵ_θ = state["ϵ_θ"]
            λ = state["λ"]
            # and increase regularization
            v = min(v - 2,-4);
        end

        # E-Step: update
        t = exp(v - spm_logdet(dFdpp)/np)
        if t > exp(16)
            dθ = - inv(dFdpp)*dFdp    # -inv(dfdx)*f
        else
            dθ = exp(t * dFdpp) * inv(dFdpp)*dFdp - inv(dFdpp)*dFdp   # (expm(dfdx*t) - I)*inv(dfdx)*f
        end

        ϵ_θ += dθ
        μθ = θμ + V*ϵ_θ
        dF  = dot(dFdp, ϵ_θ);

        # convergence condition: reach a change in Free Energy that is smaller than 0.1 four consecutive times
        print("iteration: ", k, " - F:", state["F"] - F0, " - dF predicted:", dF, "\n")
        criterion = vcat(dF < 1e-1, criterion[1:end - 1]);
        if all(criterion)
            print("convergence\n")
            break
        end
    end
    print("iterations terminated\n")
    state["F"] = F
    state["Σθ"] = V*Σθ*V'
    state["μθ"] = μθ
    return state
end