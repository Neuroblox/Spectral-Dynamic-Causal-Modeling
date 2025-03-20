# compute Jacobian of rhs w.r.t. variable -> matrix exponential solution (use ExponentialUtilities.jl)
# -> use this numerical integration as solution to the diffeq to then differentiate solution w.r.t. parameters (like sensitivity analysis in Ma et al. 2021)
# -> that Jacobian is used in all the computations of the variational Bayes

using ForwardDiff: Dual, Partials, jacobian
ForwardDiff.can_dual(::Type{Complex{Float64}}) = true

tagtype(::Dual{T,V,N}) where {T,V,N} = T

include("utils/helperfunctions.jl")
include("utils/helperfunctions_AD.jl")


function transferfunction(x, w, θμ, C, lnϵ, lndecay, lntransit)
    # compute transfer function of Volterra kernels, see fig 1 in friston2014
    # 1. compute jacobian w.r.t. f ; TODO: what is it with this "delay operator" that is set to 1 in "spm_fx_fmri.m"
    # J_x = jacobian(f, x0) # well, no need to perform this for a linear system... we already have it: θμ
    C /= 16.0   # TODO: unclear why it is devided by 16 but see spm_fx_fmri.m:49
    # 2. get jacobian of hemodynamics
    J = hemodynamics_jacobian(x[:, 2:end], lndecay, lntransit)
    θμ -= diagm(exp.(diag(θμ))/2 + diag(θμ))
    # if I eventually need also the change of variables rather than just the derivative then here is where to fix it!
    nd = size(θμ, 1)
    J_tot = [θμ zeros(nd, size(J, 2));   # add derivatives w.r.t. neural signal
             [Matrix(1.0I, size(θμ)); zeros(size(J)[1]-nd, size(θμ)[2])] J]

    dfdu = [C;
            zeros(size(J,1), size(C, 2))]

    F = eigen(J_tot) #, sortby=nothing, permute=true)
    Λ = F.values
    V = F.vectors
    # condition unstable eigenmodes
    # if max(w) > 1
    #     s = 1j*imag(s) + real(s) - exp(real(s));
    # else
    #     s = 1j*imag(s) + min(real(s),-1/32);
    # end

    # 3. get jacobian (??) of bold signal, just compute it as is done, but how is this a jacobian... it isn't! if anything it should be a gradient since the BOLD signal is scalar
    #TODO: implement numerical and compare with analytical: J_g = jacobian(bold, x0)
    dgdx = boldsignal(x, lnϵ)[2]
    dgdv = dgdx * @view V[end-size(dgdx,2)+1:end, :]     # TODO: not a clean solution, also not in the original code since it seems that the code really depends on the ordering of eigenvalues and respectively eigenvectors!
    dvdu = V\dfdu

    nw = size(w,1)            # number of frequencies
    ng = size(dgdx,1)         # number of outputs
    nu = size(dfdu,2)         # number of inputs
    nk = size(V,2)            # number of modes
    # if real(eltype(dvdu)) <: Dual
    #     S = zeros(Dual{tagtype(real(dvdu[1])), ComplexF64, length(real(dvdu[1]).partials)}, nw, ng, nu)    
    # else
    S = zeros(Complex{real(eltype(dvdu))}, nw, ng, nu)
    # end
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
    g = zeros(eltype(csd), N)

    # transform to cross-correlation function
    ccf = zeros(eltype(csd), N*2+1, size(csd,2), size(csd,3))
    for i = 1:size(csd, 2)
        for j = 1:size(csd, 3)
            g[gi] = csd[gj,i,j]
            f = idft(g)
            f = idft(vcat([0.0im; g; conj(g[end:-1:1])]))
            ccf[:,i,j] = real.(fftshift(f))*N*dw
        end
    end

    # MAR coefficients
    N = size(ccf,1)
    m = size(ccf,2)
    n = (N - 1) ÷ 2
    p = min(p, n - 1)
    ccf = ccf[(1:n) .+ n,:,:]
    A = zeros(eltype(csd), m*p, m)
    B = zeros(eltype(csd), m*p, m*p)
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
    w  = 2pi*freqs/sf    # TODO: isn't it already transformed?? Is the original really in Hz? Also clearly freqs[end] is not the sampling frequency of the signal...
    nf = length(w)
	csd = zeros(eltype(Σ), nf, nd, nd)
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

@views function csd_approx(x, w, θμ, C, α, β, γ, lnϵ, lndecay, lntransit)
    # priors of spectral parameters
    # ln(α) and ln(β), region specific fluctuations: ln(γ)
    nw = length(w)
    nd = size(θμ, 1)

    # define function that implements spectra given in equation (2) of "A DCM for resting state fMRI".
    # neuronal fluctuations (Gu) (1/f or AR(1) form)
    G = w.^(-exp(α[2]))   # spectrum of hidden dynamics
    G /= sum(G)
    Gu = zeros(eltype(G), nw, nd, nd)
    Gn = zeros(eltype(G), nw, nd, nd)
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
            Gn[:,i,j] .+= exp(β[1])*G
            Gn[:,j,i] = Gn[:,i,j]
        end
    end
    C = Matrix(I, nd, nd)
    S = transferfunction(x, w, θμ, C, lnϵ, lndecay, lntransit)

    # predicted cross-spectral density
    G = zeros(eltype(S),nw,nd,nd);
    for i = 1:nw
        G[i,:,:] = S[i,:,:]*Gu[i,:,:]*S[i,:,:]'
    end
    G_final = G + Gn
    return G_final
end

@views function csd_fmri_mtf(x, freqs, p, param)
    dim = size(x, 1)
    θμ = reshape(param[1:dim^2], dim, dim)
    C = param[(1+dim^2):(dim+dim^2)]
    lntransit = param[(1+dim+dim^2):(2dim+dim^2)]
    lndecay = param[1+2dim+dim^2]
    lnϵ = param[2+2dim+dim^2]
    α = param[[3+2dim+dim^2, 5+2dim+dim^2]]
    β = param[[4+2dim+dim^2, 6+2dim+dim^2]]
    γ = param[(7+2dim+dim^2):(6+3dim+dim^2)]
    G = csd_approx(x, freqs, θμ, C, α, β, γ, lnϵ, lndecay, lntransit)
    dt  = 1/(2*freqs[end])
    mar = csd2mar(G, freqs, dt, p-1)
    y = mar2csd(mar, freqs)
    if real(eltype(y)) <: Dual
        y_vals = Complex.((p->p.value).(real(y)), (p->p.value).(imag(y)))
        y_part = (p->p.partials).(real(y)) + (p->p.partials).(imag(y))*im
        y = map((x1, x2) -> Dual{tagtype(real(y)[1]), ComplexF64, length(x2)}(x1, Partials(Tuple(x2))), y_vals, y_part)
    end
    return y
end


mutable struct vb_state
    iter::Int
    F::Float64
    λ::Vector{Float64}
    ϵ_θ::Vector{Float64}
    μθ_po::Vector{Float64}
    Σθ::Matrix{Float64}
end

@views function variationalbayes(x, y, w, V, p, priors, niter)
    # extract priors
    Πθ_pr = priors[:Σ][:Πθ_pr]
    Πλ_pr = priors[:Σ][:Πλ_pr]
    μλ_pr = priors[:Σ][:μλ_pr]
    Q = priors[:Σ][:Q]

    # prep stuff
    μθ_pr = vecparam(priors[:μ])      # note: μθ_po is posterior and μθ_pr is prior
    np = size(V, 2)            # number of parameters
    ny = length(y)             # total number of response variables
    nq = 1
    nh = size(Q,3)             # number of precision components (this is the same as above, but may differ)
    λ = 8 * ones(nh)
    ϵ_θ = zeros(np)  # M.P - μθ_pr # still need to figure out what M.P is for. It doesn't seem to be used further down the road in nlsi_GM, only at the very beginning when p is defined first. Then replace μθ with μθ_pr above.
    μθ_po = μθ_pr + V*ϵ_θ
 
    revert = false
    f_prep = param -> csd_fmri_mtf(x, w, p, param)

    # state variable
    F = -Inf
    F0 = F
    previous_F = F
    v = -4   # log ascent rate
    criterion = [false, false, false, false]
    state = vb_state(0, F, λ, zeros(np), μθ_po, inv(Πθ_pr))
    dfdp = zeros(ComplexF64, length(w)*size(x,1)^2, np)
    local ϵ_λ, iΣ, Σλ, Σθ_po, dFdpp, dFdp
    for k = 1:niter
        state.iter = k
        dfdp = ForwardDiff.jacobian(f_prep, μθ_po) * V
        norm_dfdp = opnorm(dfdp, Inf);
        revert = isnan(norm_dfdp) || norm_dfdp > exp(32);

        if revert && k > 1
            for i = 1:4
                # reset expansion point and increase regularization
                v = min(v - 2,-4);
                t = exp(v - logdet(dFdpp)/np)

                # E-Step: update
                if t > exp(16)
                    ϵ_θ = state.ϵ_θ - dFdpp \ dFdp    # -inv(dfdx)*f
                else
                    ϵ_θ = state.ϵ_θ + expv(t, dFdpp, dFdpp \ dFdp) - dFdpp \ dFdp   # (expm(dfdx*t) - I)*inv(dfdx)*f
                end

                μθ_po = μθ_pr + V*ϵ_θ

                # J_test = JacVec(f_prep, μθ_po)
                # dfdp = stack(J_test*v for v in eachcol(V))
                dfdp = ForwardDiff.jacobian(f_prep, μθ_po) * V

                # check for stability
                norm_dfdp = opnorm(dfdp, Inf);
                revert = isnan(norm_dfdp) || norm_dfdp > exp(32);

                # break
                if ~revert
                    break
                end
            end
        end

        f = f_prep(μθ_po)
        ϵ = reshape(y - f, ny)                   # error value
        J = - dfdp   # Jacobian, unclear why we have a minus sign. Helmut: comes from deriving a Gaussian. 

        ## M-step: Fisher scoring scheme to find h = max{F(p,h)} // comment from MATLAB code
        P = zeros(eltype(J), size(Q))
        PΣ = zeros(eltype(J), size(Q))
        JPJ = zeros(real(eltype(J)), size(J,2), size(J,2), size(Q,3))
        dFdλ = zeros(real(eltype(J)), nh)
        dFdλλ = zeros(real(eltype(J)), nh, nh)
        for m = 1:8   # 8 seems arbitrary. Numbers of iterations taken from SPM12 code.
            iΣ = zeros(eltype(J), ny, ny)
            for i = 1:nh
                iΣ .+= Q[:,:,i]*exp(λ[i])
            end

            Pp = real(J' * iΣ * J)    # in MATLAB code 'real()' is applied to the resulting matrix product, why is this okay?
            Σθ_po = inv(Pp + Πθ_pr)

            if nh > 1
                for i = 1:nh
                    P[:,:,i] = Q[:,:,i]*exp(λ[i])
                    PΣ[:,:,i] = real(iΣ \ P[:,:,i])
                    JPJ[:,:,i] = real(J'*P[:,:,i]*J)      # in MATLAB code 'real()' is applied (see also some lines above)
                end
                for i = 1:nh
                    dFdλ[i] = (tr(PΣ[:,:,i])*nq - real(dot(ϵ, P[:,:,i], ϵ)) - tr(Σθ_po * JPJ[:,:,i]))/2
                    for j = i:nh
                        dFdλλ[i, j] = -real(tr(PΣ[:,:,i] * PΣ[:,:,j]))*nq/2
                        dFdλλ[j, i] = dFdλλ[i, j]
                    end
                end
            else
                dFdλ[1, 1] = ny/2 - real(ϵ'*iΣ*ϵ)/2 - tr(Σθ_po * Pp)/2;
                dFdλλ[1, 1] = - ny/2;
            end

            dFdλλ = dFdλλ + diagm(dFdλ);

            ϵ_λ = λ - μλ_pr
            dFdλ = dFdλ - Πλ_pr*ϵ_λ
            dFdλλ = dFdλλ - Πλ_pr
            Σλ = inv(-dFdλλ)

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
        L[3] = (logdet(Πλ_pr * Σλ) - dot(ϵ_λ, Πλ_pr, ϵ_λ))/2
        F = sum(L)

        if k == 1
            F0 = F
        end

        if F > state.F || k < 3
            # accept current state
            state.ϵ_θ = ϵ_θ
            state.λ = λ
            state.Σθ = Σθ_po
            state.μθ_po = μθ_po
            state.F = F
            # Conditional update of gradients and curvature
            dFdp  = -real(J' * iΣ * ϵ) - Πθ_pr * ϵ_θ    # check sign
            dFdpp = -real(J' * iΣ * J) - Πθ_pr
            # decrease regularization
            v = min(v + 1/2, 4);
        else
            # reset expansion point
            ϵ_θ = state.ϵ_θ
            λ = state.λ
            # and increase regularization
            v = min(v - 2, -4);
        end

        # E-Step: update
        t = exp(v - spm_logdet(dFdpp)/np)
        if t > exp(16)
            dθ = - inv(dFdpp) * dFdp    # -inv(dfdx)*f
        else
            dθ = exponential!(t * dFdpp) * inv(dFdpp) * dFdp - inv(dFdpp) * dFdp   # (expm(dfdx*t) - I)*inv(dfdx)*f
        end

        ϵ_θ += dθ
        μθ_po = μθ_pr + V*ϵ_θ
        dF = dot(dFdp, dθ);

        # convergence condition: reach a change in Free Energy that is smaller than 0.1 four consecutive times
        print("iteration: ", k, " - F:", state.F - F0, " - dF predicted:", dF, "\n")
        criterion = vcat(dF < 1e-1, criterion[1:end - 1]);
        if all(criterion)
            print("convergence\n")
            break
        end
    end
    print("iterations terminated\n")
    state.F = F
    state.Σθ = V*Σθ_po*V'
    state.μθ_po = μθ_po
    return state
end