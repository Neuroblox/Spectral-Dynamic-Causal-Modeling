using LinearAlgebra
using MKL
using FFTW
using ToeplitzMatrices
using ExponentialUtilities
using ForwardDiff
using OrderedCollections
using MetaGraphs
using Graphs
using ModelingToolkit
using DataFrames
using MAT

#=
TODO:
    - explore adaptive connection strength between neural activity and hemodynamic response
    - are there alternatives to small perturbations of initial conditions to introduce numerical stability?
=#

include("src/utils/typedefinitions.jl")
include("src/VariationalBayes_MTKAD.jl")
include("src/utils/mar.jl")
include("src/models/neuraldynamics_symbolic.jl")
include("src/models/measurement_symbolic.jl")
include("src/utils/MTK_utilities.jl")


### get data and compute cross spectral density which is the actual input to the spectral DCM ###
vars = matread("speedandaccuracy/matlab_cmc.mat");
data = vars["data"];
nd = size(data, 2);
dt = vars["dt"];
freqs = vec(vars["Hz"]);
max_iter = 128                       # maximum number of iterations

# p = 8;                               # order of MAR, it is hard-coded in SPM12 with this value. We will just use the same for now.
# mar = mar_ml(y, p);                  # compute MAR from time series y and model order p
# y_csd = mar2csd(mar, freqs, dt^-1);  # compute cross spectral densities from MAR parameters at specific frequencies freqs, dt^-1 is sampling rate of data
# y_csd = vars["csd"]
### Define priors and initial conditions ###
x = vars["x"];                       # initial condition of dynamic variabls
# Adj = vars["pE"]["A"];                 # initial values of connectivity matrix
# θΣ = vars["pC"];                     # prior covariance of parameter values 
# λμ = vec(vars["hE"]);                # prior mean of hyperparameters
# Πλ_p = vars["ihC"];                  # prior precision matrix of hyperparameters
# if typeof(Πλ_p) <: Number            # typically Πλ_p is a matrix, however, if only one hyperparameter is used it will turn out to be a scalar -> transform that to matrix
#     Πλ_p *= ones(1,1)
# end


########## assemble the model ##########
g = MetaDiGraph()
global_ns = :g # global namespace
regions = Dict()

@parameters lnr = 0.0
@parameters lnτ_ss=0 lnτ_sp=0 lnτ_ii=0 lnτ_dp=0
@parameters C=512.0 [tunable = false]    # TODO: SPM12 has this seemingly arbitrary 512 pre-factor in spm_fx_cmc.m
for ii = 1:nd
    region = CanonicalMicroCircuitBlox(;namespace=global_ns, name=Symbol("r$(ii)₊cmc"), 
                                        τ_ss=exp(lnτ_ss)*0.002, τ_sp=exp(lnτ_sp)*0.002, τ_ii=exp(lnτ_ii)*0.016, τ_dp=exp(lnτ_dp)*0.028, 
                                        r_ss=exp(lnr)*2.0/3, r_sp=exp(lnr)*2.0/3, r_ii=exp(lnr)*2.0/3, r_dp=exp(lnr)*2.0/3)
    add_blox!(g, region)
    regions[ii] = nv(g)    # store index of neural mass model
    input = ExternalInput(;name=Symbol("r$(ii)₊ei"), I=1.0)
    add_blox!(g, input)
    add_edge!(g, nv(g), nv(g) - 1, Dict(:weight => C))

    # add lead field (LFP measurement)
    measurement = LeadField(;name=Symbol("r$(ii)₊lf"))
    add_blox!(g, measurement)
    # connect measurement with neuronal signal
    add_edge!(g, nv(g) - 2, nv(g), Dict(:weight => 1.0))
end

nl = Int((nd^2-nd)/2)   # number of links unidirectional
@parameters a_sp_ss[1:nl] = repeat([0.0], nl) # forward connection parameter sp -> ss
@parameters a_sp_dp[1:nl] = repeat([0.0], nl) # forward connection parameter sp -> dp
@parameters a_dp_sp[1:nl] = repeat([0.0], nl) # backward connection parameter dp -> sp
@parameters a_dp_ii[1:nl] = repeat([0.0], nl) # backward connection parameters dp -> ii

k = 0
for i in 1:nd
    for j in (i+1):nd
        k += 1
        # forward connection matrix
        add_edge!(g, regions[i], regions[j], :weightmatrix,
                [0 exp(a_sp_ss[k]) 0 0;
                0 0 0 0;
                0 0 0 0;
                0 exp(a_sp_dp[k])/2 0 0] * 200)
        # backward connection matrix
        add_edge!(g, regions[j], regions[i], :weightmatrix,
                [0 0 0 0;
                0 0 0 -exp(a_dp_sp[k]);
                0 0 0 -exp(a_dp_ii[k])/2;
                0 0 0 0] * 200)
    end
end

@named fullmodel = system_from_graph(g)
fullmodel = structural_simplify(fullmodel, split=false)

# attribute initial conditions to states
sts, idx_sts = get_dynamic_states(fullmodel)
idx_u = get_idx_tagged_vars(fullmodel, "ext_input")         # get index of external input state
idx_measurement = get_eqidx_tagged_vars(fullmodel, "measurement")  # get index of equation of bold state
initcond = OrderedDict(sts .=> 0.0)
rnames = []
map(x->push!(rnames, split(string(x), "₊")[1]), sts);
rnames = unique(rnames);
for (i, r) in enumerate(rnames)
    for (j, s) in enumerate(sts[r .== map(x -> x[1], split.(string.(sts), "₊"))])
        initcond[s] = x[i, j]
    end
end

modelparam = OrderedDict()
for par in tunable_parameters(fullmodel)
    modelparam[par] = Symbolics.getdefaultval(par)
end
np = length(modelparam)
indices = Dict(:dspars => collect(1:np))
# Noise parameter mean
modelparam[:lnα] = zeros(Float64, 2, nd);           # intrinsic fluctuations, ln(α) as in equation 2 of Friston et al. 2014 
n = length(modelparam[:lnα]);
indices[:lnα] = collect(np+1:np+n);
np += n;
modelparam[:lnβ] = [-16.0, -16.0];           # global observation noise, ln(β) as above
n = length(modelparam[:lnβ]);
indices[:lnβ] = collect(np+1:np+n);
np += n;
modelparam[:lnγ] = [-16.0, -16.0];   # region specific observation noise
indices[:lnγ] = collect(np+1:np+nd);
np += nd
indices[:u] = idx_u
indices[:m] = idx_measurement
indices[:sts] = idx_sts

# define prior variances
paramvariance = copy(modelparam)
paramvariance[:lnα] = ones(Float64, size(modelparam[:lnα]))./128.0; 
paramvariance[:lnβ] = ones(Float64, nd)./128.0;
paramvariance[:lnγ] = ones(Float64, nd)./128.0;
for (k, v) in paramvariance
    if occursin("a_", string(k))
        paramvariance[k] = 1/16.0
    elseif "lnr" == string(k)
        paramvariance[k] = 1/64.0;
    elseif occursin("lnτ", string(k))
        paramvariance[k] = 1/32.0;
    elseif occursin("lf₊L", string(k))
        paramvariance[k] = 64;
    end
end

priors = DataFrame(name=[k for k in keys(modelparam)], mean=[m for m in values(modelparam)], variance=[v for v in values(paramvariance)])
hype = matread("speedandaccuracy/matlab_cmc_hyperpriors.mat");
hyperpriors = Dict(:Πλ_pr => hype["ihC"],            # prior metaparameter precision, needs to be a matrix
                   :μλ_pr => vec(hype["hE"]),              # prior metaparameter mean, needs to be a vector
                   :Q => hype["Q"]
                  );

csdsetup = Dict(:p => 8, :freq => vec(vars["Hz"]), :dt => vars["dt"]);

(state, setup) = setup_sDCM(data, fullmodel, initcond, csdsetup, priors, hyperpriors, indices, "LFP");
for iter in 1:128
    state.iter = iter
    run_sDCM_iteration!(state, setup)
    print("iteration: ", iter, " - F:", state.F[end] - state.F[2], " - dF predicted:", state.dF[end], "\n")
    if iter >= 4
        criterion = state.dF[end-3:end] .< setup.tolerance
        if all(criterion)
            print("convergence\n")
            break
        end
    end
end
