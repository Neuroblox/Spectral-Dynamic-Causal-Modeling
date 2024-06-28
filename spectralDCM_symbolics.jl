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

### Load data ###
vars = matread("../Spectral-DCM/speedandaccuracy/matlab0.01_3regions.mat");
data = vars["data"];
x = vars["x"];                       # initial condition of dynamic variabls
nr = size(data, 2);                  # number of regions
max_iter = 128                       # maximum number of iterations

########## assemble the model ##########
g = MetaDiGraph()
regions = Dict()
foo = Ref{Any}()
# decay parameter for hemodynamics lnκ and ratio of intra- to extra-vascular components lnϵ is shared across brain regions 
@parameters lnκ=0.0 [tunable = true] lnϵ=0.0 [tunable=true] C=1/16 [tunable = false]   # setting tunable=true means that these parameters are optimized
for ii = 1:nr
    region = LinearNeuralMass(;name=Symbol("r$(ii)₊lm"))
    add_blox!(g, region)
    regions[ii] = nv(g)    # store index of neural mass model
    input = ExternalInput(;name=Symbol("r$(ii)₊ei"), I=1.0)
    add_blox!(g, input)
    add_edge!(g, nv(g), nv(g) - 1, Dict(:weight => C))

    # add hemodynamic response model and observation model (BOLD signal)
    measurement = BalloonModel(;name=Symbol("r$(ii)₊bm"), lnκ=lnκ, lnϵ=lnϵ)
    add_blox!(g, measurement)
    # connect measurement with neuronal signal
    add_edge!(g, nv(g) - 2, nv(g), Dict(:weight => 1.0))
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
sts, idx_sts = get_dynamic_states(fullmodel)
idx_u = get_idx_tagged_vars(fullmodel, "ext_input")         # get index of external input state
idx_bold = get_eqidx_tagged_vars(fullmodel, "measurement")  # get index of equation of bold state
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
modelparam[:lnα] = [0.0, 0.0];           # intrinsic fluctuations, ln(α) as in equation 2 of Friston et al. 2014 
n = length(modelparam[:lnα]);
indices[:lnα] = collect(np+1:np+n);
np += n;
modelparam[:lnβ] = [0.0, 0.0];           # global observation noise, ln(β) as above
n = length(modelparam[:lnβ]);
indices[:lnβ] = collect(np+1:np+n);
np += n;
modelparam[:lnγ] = zeros(Float64, nr);   # region specific observation noise
indices[:lnγ] = collect(np+1:np+nr);
np += nr
indices[:u] = idx_u
indices[:m] = idx_bold
indices[:sts] = idx_sts

# define prior variances
paramvariance = copy(modelparam)
paramvariance[:lnα] = ones(Float64, length(modelparam[:lnα]))./64.0; 
paramvariance[:lnβ] = ones(Float64, length(modelparam[:lnβ]))./64.0;
paramvariance[:lnγ] = ones(Float64, nr)./64.0;
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

(state, setup) = setup_sDCM(data, fullmodel, initcond, csdsetup, priors, hyperpriors, indices, "fMRI");
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