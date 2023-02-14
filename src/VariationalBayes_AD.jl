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
using FFTW: ifft
# using SparseDiffTools
ForwardDiff.can_dual(::Type{Complex{Float64}}) = true
using ChainRules: _eigen_norm_phase_fwd!
# using DualNumbers
using Serialization
tagtype(::Dual{T,V,N}) where {T,V,N} = T

counter = 0 
# Base.eps(z::Complex{T}) where {T<:AbstractFloat} = hypot(eps(real(z)), eps(imag(z)))
# Base.signbit(x::Complex{T}) where {T<:AbstractFloat} = real(x) < 0
# struct NeurobloxTag end

function idft(x::AbstractArray)
    """discrete inverse fourier transform"""
    N = size(x)[1]
    out = Array{eltype(x)}(undef,N)
    for n in 0:N-1
        out[n+1] = 1/N*sum([x[k+1]*exp(2*im*π*k*n/N) for k in 0:N-1])
    end
    return out
end

function FFTW.ifft(x::Array{Complex{Dual{T, P, N}}}) where {T, P, N}
    return ifft(real(x)) + 1im*ifft(imag(x))
end

function FFTW.ifft(x::Array{Dual{T, P, N}}) where {T, P, N}
    v = (tmp->tmp.value).(x)
    iftx = ifft(v)
    iftrp = Array{Partials}(undef, length(x))
    iftip = Array{Partials}(undef, length(x))
    local iftrp_agg, iftip_agg
    for i = 1:N
        dx = (tmp->tmp.partials[i]).(x)
        iftdx = ifft(dx)
        if i == 1
            iftrp_agg = real(iftdx) .* dx
            iftip_agg = (1im * imag(iftdx)) .* dx
        else
            iftrp_agg = cat(iftrp_agg, real(iftdx) .* dx, dims=2)
            iftip_agg = cat(iftip_agg, (1im * imag(iftdx)) .* dx, dims=2)
        end
    end
    for i = 1:length(x)
        iftrp[i] = Partials(Tuple(iftrp_agg[i, :]))
        iftip[i] = Partials(Tuple(iftip_agg[i, :]))
    end
    return Complex.(Dual{T, P, N}.(real(iftx), iftrp), Dual{T, P, N}.(imag(iftx), iftip))
end

function LinearAlgebra.eigen(M::Matrix{Dual{T, P, np}}) where {T, P, np}
    nd = size(M, 1)
    A = (p->p.value).(M)
    F = eigen(A, sortby=nothing, permute=true)
    λ, V = F.values, F.vectors
    local ∂λ_agg, ∂V_agg
    # compute eigenvalue and eigenvector derivatives for all partials
    for i = 1:np
        dA = (p->p.partials[i]).(M)
        tmp = V \ dA
        ∂K = tmp * V   # V^-1 * dA * V
        ∂Kdiag = @view ∂K[diagind(∂K)]
        ∂λ_tmp = eltype(λ) <: Real ? real.(∂Kdiag) : copy(∂Kdiag)   # why do only copy when complex??
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
    # reassemble the aggregated vectors and values into a Partials type
    for i = 1:nd
        ∂λ[i] = Partials(Tuple(∂λ_agg[i, :]))
        for j = 1:nd
            ∂V[i, j] = Partials(Tuple(∂V_agg[i, j, :]))
        end
    end
    if eltype(V) <: Complex
        evals = map((x,y)->Complex(Dual{T, Float64, length(y)}(real(x), Partials(Tuple(real(y)))), 
                                   Dual{T, Float64, length(y)}(imag(x), Partials(Tuple(imag(y))))), F.values, ∂λ)
        evecs = map((x,y)->Complex(Dual{T, Float64, length(y)}(real(x), Partials(Tuple(real(y)))), 
                                   Dual{T, Float64, length(y)}(imag(x), Partials(Tuple(imag(y))))), F.vectors, ∂V)
    else
        evals = Dual{T, Float64, length(∂λ[1])}.(F.values, ∂λ)
        evecs = Dual{T, Float64, length(∂V[1])}.(F.vectors, ∂V)
    end
    return Eigen(evals, evecs)
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
    w  = 2pi*freqs/sf    # isn't it already transformed?? Is the original really in Hz? Also clearly freqs[end] is not the sampling frequency of the signal...
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
    G = w.^(-exp(β[1]))   # spectrum of hidden dynamics
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
    global counter += 1
    Main.csdapproxvars[] = G, Gn, β, α, γ

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

mutable struct vb_state
    iter::Int
    F::Float64
    λ::Vector{Float64}
    ϵ_θ::Vector{Float64}
    μθ::Vector{Float64}
    Σθ::Matrix{Float64}
end

@views function variationalbayes(x, y, w, V, param, priors, niter)
    # extract priors
    Πθ_p = priors[1]
    Πλ_p = priors[2]
    λμ = priors[3]
    Q = priors[4]

    # prep stuff
    p = Int(param[1])
    θμ = param[2:end]          # note: μθ is posterior and θμ is prior
    np = size(V, 2)            # number of parameters
    ny = length(y)             # total number of response variables
    nq = 1
    nh = size(Q,3)             # number of precision components (this is the same as above, but may differ)
    λ = 8 * ones(nh)
    ϵ_θ = zeros(np)  # M.P - θμ # still need to figure out what M.P is for. It doesn't seem to be used further down the road in nlsi_GM, only at the very beginning when p is defined first. Then replace μθ with θμ above.
    μθ = θμ + V*ϵ_θ
 
    revert = false
    f_prep = param -> csd_fmri_mtf(x, w, p, param)

    # state variable
    F = -Inf
    F0 = F
    previous_F = F
    v = -4   # log ascent rate
    criterion = [false, false, false, false]
    state = vb_state(0, F, λ, zeros(np), μθ, inv(Πθ_p))

    local ϵ_λ, iΣ, Σλ, Σθ, dFdpp, dFdp
    for k = 1:niter
        state.iter = k

        f = f_prep(μθ)
        # jac_prototype = Array{Dual{ForwardDiff.Tag{NeurobloxTag, ComplexF64}, ComplexF64, 12}}(undef, ny, np)
        # @show typeof(jac_prototype)
        # dfdp = forwarddiff_color_jacobian(f_prep, μθ) * V
        # forwarddiff_color_jacobian(f,x,ForwardColorJacCache(f,x,chunksize,tag = NeurobloxTag()),nothing)
        dfdp = ForwardDiff.jacobian(f_prep, μθ) * V
        # dfdp = Complex.((p -> p.value).(real(dfdp)), (p -> p.value).(imag(dfdp)))

        norm_dfdp = matlab_norm(dfdp, Inf);
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

                μθ = θμ + V*ϵ_θ

                f = f_prep(μθ)
                # dfdp = forwarddiff_color_jacobian(f_prep, μθ) * V
                dfdp = ForwardDiff.jacobian(f_prep, μθ) * V

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
            iΣ = zeros(eltype(J), ny, ny)
            for i = 1:nh
                iΣ .+= Q[:,:,i]*exp(λ[i])
            end

            Σ = inv(iΣ)             # Julia requires conversion to dense matrix before inversion so just use dense from the get-go
            Pp = real(J' * iΣ * J)    # in MATLAB code 'real()' is applied to the resulting matrix product, why?
            Σθ = inv(Pp + Πθ_p)

            P = zeros(eltype(J), size(Q))
            PΣ = zeros(eltype(J), size(Q))
            JPJ = zeros(real(eltype(J)), size(Pp,1), size(Pp,2), size(Q,3))
            for i = 1:nh
                P[:,:,i] = Q[:,:,i]*exp(λ[i])
                PΣ[:,:,i] = P[:,:,i] * Σ
                JPJ[:,:,i] = real(J'*P[:,:,i]*J)      # in MATLAB code 'real()' is applied (see also some lines above), what's the rational?
            end
            dFdh = zeros(eltype(J), nh)
            dFdhh = zeros(real(eltype(J)), nh, nh)
            for i = 1:nh
                dFdh[i] = (tr(PΣ[:,:,i])*nq - real(dot(ϵ, P[:,:,i], ϵ)) - tr(Σθ * JPJ[:,:,i]))/2
                for j = i:nh
                    dFdhh[i, j] = -real(tr(PΣ[:,:,i] * PΣ[:,:,j]))*nq/2
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
                dλ = -real(dFdhh \ dFdh)
            else
                idFdhh = inv(dFdhh)
                dλ = real(exponential!(t * dFdhh) * idFdhh*dFdh - idFdhh*dFdh)   # (expm(dfdx*t) - I)*inv(dfdx)*f ~~~ could also be done with expv but doesn't work with Dual.
            end

            dλ = [min(max(x, -1.0), 1.0) for x in dλ]      # probably precaution for numerical instabilities?
            λ = λ + dλ

            dF = dot(dFdh, dλ)
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
        L[2] = (logdet(Πθ_p * Σθ) - dot(ϵ_θ, Πθ_p, ϵ_θ))/2
        L[3] = (logdet(Πλ_p * Σλ) - dot(ϵ_λ, Πλ_p, ϵ_λ))/2
        F = sum(L)

        if k == 1
            F0 = F
        end

        if F > state.F || k < 3
            # accept current state
            state.ϵ_θ = ϵ_θ
            state.λ = λ
            state.Σθ = Σθ
            state.μθ = μθ
            state.F = F
            # Conditional update of gradients and curvature
            dFdp  = -real(J' * iΣ * ϵ) - Πθ_p * ϵ_θ    # check sign
            dFdpp = -real(J' * iΣ * J) - Πθ_p
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
        μθ = θμ + V*ϵ_θ
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
    state.Σθ = V*Σθ*V'
    state.μθ = μθ
    return state
end