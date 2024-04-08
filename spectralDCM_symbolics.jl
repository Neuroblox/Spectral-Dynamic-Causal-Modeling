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

abstract type AbstractBlox end # Blox is the abstract type for Blox that are displayed in the GUI
abstract type AbstractComponent end
abstract type BloxConnection end

# subtypes of Blox define categories of Blox that are displayed in separate sections of the GUI
abstract type AbstractNeuronBlox <: AbstractBlox end
abstract type NeuralMassBlox <: AbstractBlox end
abstract type CompositeBlox <: AbstractBlox end
abstract type ObserverBlox end   # not AbstractBlox since it should not show up in the GUI

# struct types for Variational Laplace
mutable struct VLState
    iter::Int                    # number of iteration
    v::Float64                   # log ascent rate of SPM style Levenberg-Marquardt optimization
    F::Vector{Float64}           # free energy vector (store at each iteration)
    dF::Vector{Float64}          # predicted free energy changes (store at each iteration)
    λ::Vector{Float64}           # hyperparameter
    ϵ_θ::Vector{Float64}         # prediction error of parameters θ
    reset_state::Vector{Any}     # store state to reset to [ϵ_θ and λ] when the free energy deteriorates
    μθ_po::Vector{Float64}       # posterior expectation value of parameters 
    Σθ_po::Matrix{Float64}       # posterior covariance matrix of parameters
    dFdθ::Vector{Float64}        # free energy gradient w.r.t. parameters
    dFdθθ::Matrix{Float64}       # free energy Hessian w.r.t. parameters
end

struct VLSetup
    model_at_x0                               # model evaluated at initial conditions
    y_csd::Array{Complex}                     # cross-spectral density approximated by fitting MARs to data
    tolerance::Float64                        # convergence criterion
    systemnums::Vector{Int}                   # several integers -> np: n. parameters, ny: n. datapoints, nq: n. Q matrices, nh: n. hyperparameters
    systemvecs::Vector{Vector{Float64}}       # μθ_pr: prior expectation values of parameters and μλ_pr: prior expectation values of hyperparameters
    systemmatrices::Vector{Matrix{Float64}}   # Πθ_pr: prior precision matrix of parameters, Πλ_pr: prior precision matrix of hyperparameters
    Q::Matrix{Complex}                        # linear decomposition of precision matrix of parameters, typically just one matrix, the empirical correlation matrix
end

include("src/VariationalBayes_MTKAD.jl")
include("src/utils/mar.jl")
include("src/models/neuraldynamics_symbolic.jl")
include("src/models/measurement_symbolic.jl")
include("src/utils/MTK_utilities.jl")


### Load data ###
vars = matread("../Spectral-DCM/speedandaccuracy/matlab0.01_3regions.mat");
data = vars["data"];
x = vars["x"];                       # initial condition of dynamic variabls
nr = size(data, 2);                  # number of regions
max_iter = 128                       # maximum number of iterations

########## assemble the model ##########
g = MetaDiGraph()
regions = Dict()

# decay parameter for hemodynamics lnκ and ratio of intra- to extra-vascular components lnϵ is shared across brain regions 
@parameters lnκ=0.0 [tunable = true] lnϵ=0.0 [tunable=true]   # setting tunable=true means that these parameters are optimized
for ii = 1:nr
    region = LinearNeuralMass(;name=Symbol("r$(ii)₊lm"))
    add_blox!(g, region)
    regions[ii] = 2ii - 1    # store index of neural mass model
    # add hemodynamic response model and observation model (BOLD signal)
    measurement = BalloonModel(;name=Symbol("r$(ii)₊bm"), lnκ=lnκ, lnϵ=lnϵ)
    add_blox!(g, measurement)
    # connect measurement with neuronal signal
    add_edge!(g, 2ii - 1, 2ii, Dict(:weight => 1.0))
end

# add symbolic weights
@parameters A[1:length(vars["pE"]["A"])] = vec(vars["pE"]["A"]) [tunable = true]
for (i, idx) in enumerate(CartesianIndices(vars["pE"]["A"]))
    if idx[1] == idx[2]
        add_edge!(g, regions[idx[1]], regions[idx[2]], :weight, -exp(A[i])/2)  # treatement of diagonal elements in SPM12
    else
        add_edge!(g, regions[idx[2]], regions[idx[1]], :weight, A[i])
    end
end

# compose model
@named fullmodel = system_from_graph(g)
fullmodel = structural_simplify(fullmodel, split=false)

# attribute initial conditions to states
ds_states, idx_u, idx_bold = get_dynamic_states(fullmodel)
initcond = OrderedDict(ds_states .=> 0.0)
rnames = []
map(x->push!(rnames, split(string(x), "₊")[1]), ds_states);
rnames = unique(rnames);
for (i, r) in enumerate(rnames)
    for (j, s) in enumerate(ds_states[r .== map(x -> x[1], split.(string.(ds_states), "₊"))])
        initcond[s] = x[i, j]
    end
end

modelparam = OrderedDict()
for par in tunable_parameters(fullmodel)
    modelparam[par] = Symbolics.getdefaultval(par)
end
np = length(modelparam)
params_idx = Dict(:dspars => collect(1:np))
# Noise parameter mean
modelparam[:lnα] = [0.0, 0.0];           # intrinsic fluctuations, ln(α) as in equation 2 of Friston et al. 2014 
n = length(modelparam[:lnα]);
params_idx[:lnα] = collect(np+1:np+n);
np += n;
modelparam[:lnβ] = [0.0, 0.0];           # global observation noise, ln(β) as above
n = length(modelparam[:lnβ]);
params_idx[:lnβ] = collect(np+1:np+n);
np += n;
modelparam[:lnγ] = zeros(Float64, nr);   # region specific observation noise
params_idx[:lnγ] = collect(np+1:np+nr);
np += nr
params_idx[:u] = idx_u
params_idx[:bold] = idx_bold

# define prior variances
paramvariance = copy(modelparam)
paramvariance[:lnγ] = ones(Float64, nr)./64.0;
paramvariance[:lnα] = ones(Float64, length(modelparam[:lnα]))./64.0; 
paramvariance[:lnβ] = ones(Float64, length(modelparam[:lnβ]))./64.0;
for (k, v) in paramvariance
    if occursin("A[", string(k))
        paramvariance[k] = vars["pC"][1, 1]
    elseif occursin("κ", string(k))
        paramvariance[k] = ones(length(v))./256.0;
    elseif occursin("ϵ", string(k))
        paramvariance[k] = 1/256.0;
    elseif occursin("τ", string(k))
        paramvariance[k] = 1/256.0;
    end
end

priors = DataFrame(name=[k for k in keys(modelparam)], mean=[m for m in values(modelparam)], variance=[v for v in values(paramvariance)])
hyperpriors = Dict(:Πλ_pr => vars["ihC"]*ones(1, 1),   # prior metaparameter precision, needs to be a matrix
                   :μλ_pr => [vars["hE"]]              # prior metaparameter mean, needs to be a vector
                  );

csdsetup = Dict(:p => 8, :freq => vec(vars["Hz"]), :dt => vars["dt"]);

(state, setup) = setup_sDCM(data, fullmodel, initcond, csdsetup, priors, hyperpriors, params_idx);
for iter in 1:max_iter
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
print("maxixmum iterations reached\n")
