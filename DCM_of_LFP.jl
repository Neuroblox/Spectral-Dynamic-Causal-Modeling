# apply spectral DCM to LFP data

using LinearAlgebra
using MKL
using FFTW
using ToeplitzMatrices
using MAT
using ExponentialUtilities
using Serialization
using OrderedCollections
using StatProfilerHTML

function Base.vec(x::T) where (T <: Real)
    return x*ones(1)
end

include("src/hemodynamic_response.jl")        # hemodynamic and BOLD signal model
include("src/VariationalBayes_AD.jl")         # this can be switched between _spm12 and _AD version. There is also a separate ADVI version in VariationalBayes_ADVI.jl
include("src/mar.jl")                         # multivariate auto-regressive model functions



### get data and compute cross spectral density which is the actual input to the spectral DCM ###
vars = matread("speedandaccuracy/matlab0.01_10regions.mat");
y = vars["data"];
nd = size(y, 2);
dt = vars["dt"];
freqs = vec(vars["Hz"]);
p = 8;                               # order of MAR, it is hard-coded in SPM12 with this value. We will just use the same for now.
mar = mar_ml(y, p);                  # compute MAR from time series y and model order p
y_csd = mar2csd(mar, freqs, dt^-1);  # compute cross spectral densities from MAR parameters at specific frequencies freqs, dt^-1 is sampling rate of data
# y_csd = vars["data_csd"][1]
### Define priors and initial conditions ###
x = vars["x"];                       # initial condition of dynamic variabls
Adj = vars["pE"]["A"];                 # initial values of connectivity matrix
θΣ = vars["pC"];                     # prior covariance of parameter values 
λμ = vec(vars["hE"]);                # prior mean of hyperparameters
Πλ_p = vars["ihC"];                  # prior precision matrix of hyperparameters
if typeof(Πλ_p) <: Number            # typically Πλ_p is a matrix, however, if only one hyperparameter is used it will turn out to be a scalar -> transform that to matrix
    Πλ_p *= ones(1,1)
end

########## assemble the model ##########

regions = []
connex = Num[]
@parameters κ=0.0
for ii = 1:nd
    @named nmm = linearneuralmass()
    @named hemo = hemodynamicsMTK(;κ=κ, τ=0.0)
    eqs = [nmm.x ~ hemo.x]
    region = ODESystem(eqs, systems=[nmm, hemo], name=Symbol("r$ii"))

    push!(connex, region.nmm.x)
    push!(regions, region)
end
diagelem = [(i-1)*nd+i for i in 1:nd]
@parameters A[1:length(Adj)] = vec(Adj)

@named model = linearconnectionssymbolic(sys=regions, adj_matrix=[i in diagelem ? -exp(a)/2 : a for (i, a) in enumerate(A)], connector=connex)
nrnmodel = structural_simplify(model)
all_s = states(nrnmodel)

sts = OrderedDict{typeof(all_s[1]), eltype(x)}()
rnames = []
map(x->push!(rnames, split(string(x), "₊")[1]), all_s); 
rnames = unique(rnames);
for (i, r) in enumerate(rnames)
    for (j, s) in enumerate(all_s[r .== map(x -> x[1], split.(string.(all_s), "₊"))])   # TODO: fix this solution, it is not robust!!
        sts[s] = x[i, j]
    end
end

@named bold = boldsignal()
# foo = Ref{Any}()
grad_full = function(p, grad, sts, nd)
    tmp = zeros(typeof(p), nd, length(sts))
    for i in 1:nd
        # need to resort states and then also the gradient, because in bold the variables are sorted differently from nrnmodel
        # tmp[i, (i-1)*5 .+ (4:5)] = grad(sts[vcat([1], (i-1)*5 .+ (4:5))], p, t)[2:3]
        tmp[i, (i-1)*5 .+ (4:5)] = grad(sts[vcat([1], (i-1)*5 .+ (5:-1:4))], p, t)[3:-1:2]
    end
    return tmp
end
jac_f = generate_jacobian(nrnmodel, expression = Val{false})[1]
grad_g = generate_jacobian(bold, expression = Val{false})[1]

statevals = [v for v in values(sts)]
derivatives = Dict(:∂f => par -> jac_f(statevals, par, t),
                   :∂g => par -> grad_full(par, grad_g, statevals, nd))

modelparam = OrderedDict{Any, Any}()
for par in parameters(nrnmodel)
    modelparam[par] = Symbolics.getdefaultval(par)
end
# Noise parameter mean
modelparam[:lnα] = [0.0, 0.0];           # intrinsic fluctuations, ln(α) as in equation 2 of Friston et al. 2014 
modelparam[:lnβ] = [0.0, 0.0];           # global observation noise, ln(β) as above
modelparam[:lnγ] = zeros(Float64, nd);   # region specific observation noise
modelparam[:C] = ones(Float64, nd);     # C as in equation 3. NB: whatever C is defined to be here, it will be replaced in csd_approx. Another strange thing of SPM12...

for par in parameters(bold)
    modelparam[par] = Symbolics.getdefaultval(par)
end

# define prior variances
paramvariance = copy(modelparam)
paramvariance[:C] = zeros(Float64, nd);
paramvariance[:lnγ] = ones(Float64, nd)./64.0;
paramvariance[:lnα] = ones(Float64, length(modelparam[:lnα]))./64.0; 
paramvariance[:lnβ] = ones(Float64, length(modelparam[:lnβ]))./64.0;
for (k, v) in paramvariance
    if occursin("A[", string(k))
        paramvariance[k] = θΣ[1,1]
    elseif occursin("κ", string(k))
        paramvariance[k] = ones(length(v))./256.0;
    elseif occursin("ϵ", string(k))
        paramvariance[k] = 1/256.0;
    elseif occursin("τ", string(k))
        paramvariance[k] = 1/256.0;
    end
end
θΣ = diagm(vecparam(paramvariance));

# depending on the definition of the priors (note that we take it from the SPM12 code), some dimensions are set to 0 and thus are not changed.
# Extract these dimensions and remove them from the remaining computation. I find this a bit odd and further thoughts would be necessary to understand
# to what extend this is a the most reasonable approach. 
idx = findall(x -> x != 0, θΣ);
V = zeros(size(θΣ, 1), length(idx));
order = sortperm(θΣ[idx], rev=true);
idx = idx[order];
for i = 1:length(idx)
    V[idx[i][1], i] = 1.0
end
θΣ = V'*θΣ*V;       # reduce dimension by removing columns and rows that are all 0

Q = csd_Q(y_csd);                 # compute prior of Q, the precision (of the data) components. See Friston etal. 2007 Appendix A

priors = Dict(:μ => modelparam,
              :Σ => Dict(
                         :Πθ_pr => inv(θΣ),           # prior model parameter precision
                         :Πλ_pr => Πλ_p,              # prior metaparameter precision
                         :μλ_pr => vec(vars["hE"]),   # prior metaparameter mean
                         :Q => Q                      # decomposition of model parameter covariance
                        )
             );

# TODO: it would be much nicer getting the positions of the A's without this hacky solution.
# Perhaps even circumvent the need to find the position, see transferfunction_fmri.
jac = calculate_jacobian(nrnmodel)
idx_A = findall(occursin.("A[", string.(jac)))

### Compute the DCM ###
@time results = variationalbayes(idx_A, y_csd, derivatives, freqs, V, p, priors, 128)


# J, iΣ, ϵ, Πθ_pr = foo[]
