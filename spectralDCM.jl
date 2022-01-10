# using ForwardDiff: jacobian
using LinearAlgebra
using FFTW
using ToeplitzMatrices
using MAT
using Plots
using Serialization



function hemodynamics!(dx, x, na, decay, transit)
    """
    na     - neural activity
    Components of x are:
    x[:,1] - vascular signal: s
    x[:,2] - rCBF: ln(f)
    x[:,3] - venous volume: ln(ν)
    x[:,4] - deoxyhemoglobin (dHb): ln(q)
    
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
    x[:, 2:4] = exp.(x[:, 2:4])

    # signal decay
    κ = H[1]*exp(decay)

    # transit time
    τ = H[3]*exp.(transit)

    # Fout = f(v) - outflow
    fv = x[:, 2].^(H[4]^-1)

    # e = f(f) - oxygen extraction
    ff = (1.0 .- (1.0 - H[5]).^(x[:, 2].^-1))/H[5]

    # implement differential state equation f = dx/dt (hemodynamic)

    dx[:, 1] = na .- κ.*x[:, 1] .- H[2]*(x[:, 2] .- 1)   # Corresponds to eq (9)
    dx[:, 2] = x[:, 1]./x[:, 2]  # Corresponds to eq (10), note the added logarithm (see doc string)
    dx[:, 3] = (x[:, 2] .- fv)./(τ.*x[:, 3])    # Corresponds to eq (8), note the added logarithm (see doc string)
    dx[:, 4] = (ff.*x[:, 2] .- fv.*x[:, 4]./x[:, 3])./(τ.*x[:, 4])  # Corresponds to eq (8), note the added logarithm (see doc string)

    d = size(x)[1]   # number of dimensions, equals typically number of regions
    J = zeros(4d, 4d)

    J[1:d,1:d] = Matrix(-κ*I, d, d)   # TODO: make it work when κ/decay is a vector. Only solution if-clause? diagm doesn't allow scalars, [κ] would work in that case
    J[1:d,d+1:2d] = diagm(-H[2]*x[:,2])
    J[d+1:2d,1:d] = diagm( x[:,2].^-1)
    J[d+1:2d,d+1:2d] = diagm(-x[:,1]./x[:,2])
    J[2d+1:3d,d+1:2d] = diagm(x[:,2]./(τ.*x[:,3]))
    J[2d+1:3d,2d+1:3d] = diagm(-x[:,3].^(H[4]^-1 - 1)./(τ*H[4]) - (x[:,3].^-1 .*(x[:,2] - x[:,3].^(H[4]^-1)))./τ)
    J[3d+1:4d,d+1:2d] = diagm((x[:,2] .+ log(1 - H[5])*(1 - H[5]).^(x[:,2].^-1) .- x[:,2].*(1 - H[5]).^(x[:,2].^-1))./(τ.*x[:,4]*H[5]))
    J[3d+1:4d,2d+1:3d] = diagm((x[:,3].^(H[4]^-1 - 1)*(H[4] - 1))./(τ*H[4]))
    J[3d+1:4d,3d+1:4d] = diagm((x[:,2]./x[:,4]).*((1 - H[5]).^(x[:,2].^-1) .- 1)./(τ*H[5]))

    return (dx, J)
end

# Simulate the BOLD signal
function boldsignal(x, lnϵ)
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
    ϵ  = exp(lnϵ)

    # -Coefficients in BOLD signal model
    k1  = 4.3*nu0*E0*TE;
    k2  = ϵ*r0*E0*TE;
    k3  = 1 - ϵ;
    # -Output equation of BOLD signal model
    ν   = exp.(x[:,4])
    q   = exp.(x[:,5])
    bold = V0*(k1 .- k1*q .+ k2 .- k2*q./ν .+ k3 .- k3*ν)

    nd = size(x, 1)
    ∇ = zeros(nd, 2nd)
    ∇[1:nd, 1:nd]     = diagm(-V0*(k3*ν .- k2*q./ν))    # TODO: it is unclear why this is the correct gradient, do the algebra... (note this is a gradient per area, not a Jacobian)
    ∇[1:nd, nd+1:2nd] = diagm(-V0*(k1*q .+ k2*q./ν))

    return (bold, ∇)
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

    F = eigen(J_tot, sortby=nothing, permute=true)
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
    # var = matread("../matlabspectrum.mat")
    # V = var["v"]
    # Λ = var["s"]
    dgdv  = dgdx*V[end-size(dgdx,2)+1:end, :]     # TODO: not a clean solution, also not in the original code since it seems that the code really depends on the ordering of eigenvalues and respectively eigenvectors!
    dvdu  = pinv(V)*dfdu

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

    noise_cov  = ccf[1,:,:] - A'*a
    lags = [-a[i:p:m*p, :] for i = 1:p]
    return (lags, noise_cov)
end

function mar2csd(lags, noise_cov, p, freqs)
    dim = size(noise_cov, 1)
    sf = 2*freqs[end]
    w  = 2*pi*freqs/sf    # isn't it already transformed?? Is the original really in Hz?
    nf = length(w)
	csd = zeros(ComplexF64, nf, dim, dim)
	for i = 1:nf
		af_tmp = I
		for k = 1:p
			af_tmp = af_tmp + lags[k] * exp(-im * k * w[i])
		end
		iaf_tmp = inv(af_tmp)
		csd[i,:,:] = iaf_tmp * noise_cov * iaf_tmp'     # is this really the covariance or rather precision?!
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
    Gu = zeros(nw, nd, nd)
    Gn = zeros(nw, nd, nd)
    G = w.^(-exp(β[1]))   # spectrum of hidden dynamics
    G /= sum(G)
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

function csd_fmri_mtf(x, w, p, param)
    dim = size(x, 1)
    θμ = reshape(param[1:dim^2], dim, dim)
    C = param[(1+dim^2):(3+dim^2)]
    lntransit = param[(4+dim^2):(6+dim^2)]
    lndecay = param[7+dim^2]
    lnϵ = param[8+dim^2]
    α[1] = param[9+dim^2]
    β[1] = param[10+dim^2]
    α[2] = param[11+dim^2]
    β[2] = param[12+dim^2]
    γ = param[(13+dim^2):(15+dim^2)]
    G = csd_approx(x, w, θμ, C, α, β, γ, lnϵ, lndecay, lntransit)
    dt  = 1/(2*w[end])
    lags, noise_cov = csd2mar(G, w, dt, p-1)
    y = mar2csd(lags, noise_cov, p-1, w)
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
    norm_matlab = maximum(vec(sum(abs.(Q),dims=2)))
    Q = inv(Q .+ norm_matlab/32*Matrix(I, size(Q)))   # TODO: MATLAB's and Julia's norm function are different! Reconciliate?
    return Q
end


### DEFINE SEVERAL VARIABLES AND PRIORS TO GET STARTED ###

vars = matread("spectralDCM_demodata_notsparse.mat")
# Y_mat = vars["Y"]
y_csd = vars["csd"]
w = vec(vars["M_nosparse"]["Hz"])
θμ = vars["M_nosparse"]["pE"]["A"]    # see table 1 in friston2014 for values of priors 
θΣ = vars["M_nosparse"]["pC"]
λE = vars["M_nosparse"]["hE"]
Πλ_p = vars["M_nosparse"]["ihC"]
np = length(θμ)            # number of parameters
ny = length(y_csd)         # total number of response variables
ns = size(y_csd, 1)        # number of samples
nr = ny÷ns                 # number of response components
nq = 1
Q = zeros(ny,ny,nr)
for i = 1:nr
    Q[((i-1)*ns+1):(i*ns), ((i-1)*ns+1):(i*ns), i] = Matrix(1.0I, ns, ns)
end
nh = size(Q,3)             # number of precision components (this is the same as above, but may differ)
λ = 8 * ones(nh)

idx = findall(x -> x != 0, θΣ)
V = zeros(size(θΣ, 1), length(idx))
order = sortperm(θΣ[idx], rev=true)
idx = idx[order]
for i = 1:length(idx)
    V[idx[i][1], i] = 1.0
end
θΣ = V'*θΣ*V
Πθ_p = inv(θΣ)

dim = size(θμ, 1)
C = zeros(Float64, dim)    # NB: whatever C is defined to be here, it will be replaced in csd_approx. A little strange thing of SPM12
p = 8
α = [0.0, 0.0]
β = [0.0, 0.0]
γ = zeros(Float64, dim)
lnϵ = 0.0                        # BOLD signal parameter
lndecay = 0.0                    # hemodynamic parameter
lntransit = zeros(Float64, dim)  # hemodynamic parameters
x = zeros(Float64, 3, 5)
param = [reshape(θμ, dim^2); C; lntransit; lndecay; lnϵ; α[1]; β[1]; α[2]; β[2]; γ;]
# Strange α and β sorting? yes. This is to be consistent with the SPM12 code while keeping nomenclature consistent with the spectral DCM paper

dx = exp(-8)
f_prep = param -> csd_fmri_mtf(x, w, p, param)



dfdp, f = diff(V, dx, f_prep, param);
# TODO: SPM12 has a stability check here, we omit it for now.

e = reshape(y_csd - f, ny)   # error value
J = - transpose(reshape(dfdp, 19, ny))      # Jacobian, unclear why we have a minus sign. Helmut: comes from deriving a Gaussian. 


## M-step: Fisher scoring scheme to find h = max{F(p,h)} // comment from MATLAB code
iΣ = zeros(ny, ny)  # local iΣ
ϵ_λ = zeros(np)
for m = 1:8   # 8 seems arbitrary. Numbers of iterations taken from SPM12 code.
    tmp = zeros(ny, ny)
    for i = 1:nh
        tmp = tmp .+ Q[:,:,i]*exp(λ[i])
    end
    iΣ = tmp

    Σ = inv(iΣ)        # Julia requires conversion to dense matrix before inversion so just use dense from the get-go
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
        dFdh[i] = (tr(PΣ[:,:,i])*nq - real(e'*P[:,:,i]*e) - tr(Σθ * JPJ[:,:,i]))/2
        for j = i:nh
            dFdhh[i, j] = - tr(PΣ[:,:,i] * PΣ[:,:,j])*nq/2
            dFdhh[j, i] = dFdhh[i, j]
        end
    end

    ϵ_λ = λ - λE
    dFdh = dFdh - Πλ_p*ϵ_λ
    dFdhh = dFdhh - Πλ_p
    Σλ = inv(-dFdhh)

    dλ = -inv(dFdhh) * dFdh    # Gauss-Newton step, transform into Levenberg-Marquardt, see MATLAB code
    dλ = [min(max(x,-1.0),1.0) for x in dλ]      # probably precaution for numerical instabilities?
    λ = λ + dλ

    dF = dot(dFdh,dλ)
    print(dF,"\n")
    # NB: it is unclear as to whether this is being reached. In this first tests iterations seem to be 
    # trapped in a periodic orbit jumping around between 1250 and 940. At that point the results become
    # somewhat arbitrary. The iterations stop at 8, whatever the last value of iΣ etc. is will be carried on.
    if dF < 1e-2
        break
    end
end

## E-Step with Levenberg-Marquardt regularization // comment from MATLAB code
L = zeros(3)
L[1] = (logdet(iΣ)*nq  - real(e'*iΣ*e) - ny*log(2pi))/2   # figure out why dot gives a different result here (dot(e',iΣ,e))
L[2] = (logdet(Πθ_p * Σθ) - dot(ϵ_θ', Πθ_p, ϵ_θ))/2
L[3] = (logdet(Πλ_p * Σλ) - dot(ϵ_λ', Πλ_p, ϵ_λ))/2;
F = sum(L);