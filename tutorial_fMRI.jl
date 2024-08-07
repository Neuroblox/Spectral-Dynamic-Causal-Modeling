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
using DifferentialEquations
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
include("src/models/neuraldynamics_MTK.jl")
include("src/models/measurement_MTK.jl")
include("src/utils/MTK_utilities.jl")


########## Assemble model ##########

# Step 1: define the graph, add blocks. Note that spectral DCM (as is implemented in SPM12) requires an input blocks
# Step 2: simulate the model
# Step 3: compute cross spectral density
# Step 4: setup the DCM
# Step 5: estimate

nr = 3             # number of regions
g = MetaDiGraph()
regions = Dict()   # this dictionary is used to keep track of the neural mass block index to more easily connect to other blocks

# decay parameter for hemodynamics lnκ and ratio of intra- to extra-vascular components lnϵ is shared across brain regions 
@parameters lnκ=0.0 lnϵ=0.0 
@parameters C=1/16   # note that C=1/16 is taken from SPM12 and stabilizes the balloon model simulation. Alternatively the noise or the weight of the edge connecting neuronal activity and balloon model can be reduced.
for i = 1:nr
    region = LinearNeuralMass(;name=Symbol("r$(i)₊lm"))
    add_blox!(g, region)
    regions[i] = nv(g)         # store index of neural mass model
    input = OUBlox(;name=Symbol("r$(i)₊ou"), σ=0.1)
    add_blox!(g, input)
    add_edge!(g, nv(g), regions[i], Dict(:weight => C))

    # add hemodynamic response model and observation model (BOLD signal)
    measurement = BalloonModel(;name=Symbol("r$(i)₊bm"), lnκ=lnκ, lnϵ=lnϵ)
    add_blox!(g, measurement)
    # connect measurement with neuronal signal
    add_edge!(g, regions[i], nv(g), Dict(:weight => 1.0))
end

A_true = randn(nr, nr)
A_true -= diagm(map(a -> sum(abs, a), eachrow(A_true)))    # ensure diagonal dominance of matrix
# add symbolic weights
@parameters A[1:nr^2] = vec(A_true) [tunable = true]
for (i, idx) in enumerate(CartesianIndices(A_true))
    if idx[1] == idx[2]
        add_edge!(g, regions[idx[1]], regions[idx[2]], :weight, A[i])  # -exp(A[i])/2: treatement of diagonal elements in SPM12 to make diagonal dominance (see Gershgorin Theorem) more likely but it is not guaranteed
    else
        add_edge!(g, regions[idx[2]], regions[idx[1]], :weight, A[i])
    end
end

# compose model
@named simmodel = system_from_graph(g)
simmodel = structural_simplify(simmodel, split=false)

# simulate model
tspan = (0.0, 512.0)
prob = SDEProblem(simmodel, [], tspan)
dt = 2.0
sol = solve(prob, saveat=dt);

### plot simulation results ###
using Plots

# plot bold signal time series
idx_m = get_idx_tagged_vars(simmodel, "measurement")    # get index of bold signal
Plots.plot(sol, idxs=idx_m)

# estimate and plot cross-spectrum
solm = Matrix(sol)'
data = solm[:, idx_m]
p = 8;
mar = mar_ml(data, p)
ns = size(data, 1)
freq = range(min(128, ns*dt)^-1, max(8, 2*dt)^-1, 32)
csd = mar2csd(mar, freq, dt^-1)
Plots.plot(freq, real.(csd[:, 2, 2]))


########## fit model ##########

# assemble model for fitting
g = MetaDiGraph()
regions = Dict()   # this dictionary is used to keep track of the neural mass block index to more easily connect to other blocks

# decay parameter for hemodynamics lnκ and ratio of intra- to extra-vascular components lnϵ is shared across brain regions 
@parameters lnκ=0.0 [tunable=false] lnϵ=0.0 [tunable=false] lnτ=0.0 [tunable=false] 
@parameters C=1/16 [tunable=false]   # note that C=1/16 is taken from SPM12 and stabilizes the balloon model simulation. Alternatively the noise or the weight of the edge connecting neuronal activity and balloon model can be reduced.
for i = 1:nr
    region = LinearNeuralMass(;name=Symbol("r$(i)₊lm"))
    add_blox!(g, region)
    regions[i] = nv(g)         # store index of neural mass model
    input = ExternalInput(;name=Symbol("r$(i)₊ei"))
    add_blox!(g, input)
    add_edge!(g, nv(g), regions[i], Dict(:weight => C))

    # add hemodynamic response model and observation model (BOLD signal)
    measurement = BalloonModel(;name=Symbol("r$(i)₊bm"), lnτ=lnτ, lnκ=lnκ, lnϵ=lnϵ)
    add_blox!(g, measurement)
    # connect measurement with neuronal signal
    add_edge!(g, regions[i], nv(g), Dict(:weight => 1.0))
end

A_prior = 0.01*randn(nr, nr)
A_prior -= diagm(diag(A_prior))    # ensure diagonal dominance of matrix
# add symbolic weights
@parameters A[1:nr^2] = vec(A_prior) [tunable = true]
for (i, idx) in enumerate(CartesianIndices(A_prior))
    if idx[1] == idx[2]
        add_edge!(g, regions[idx[1]], regions[idx[2]], :weight, -exp(A[i])/2)  # -exp(A[i])/2: treatement of diagonal elements in SPM12 to make diagonal dominance (see Gershgorin Theorem) more likely but it is not guaranteed
    else
        add_edge!(g, regions[idx[2]], regions[idx[1]], :weight, A[i])
    end
end

# compose model
@named fitmodel = system_from_graph(g)
fitmodel = structural_simplify(fitmodel, split=false)


max_iter = 128            # maximum number of iterations

# set which parameters to fit of simmodel, instead of building a wholly new model 
# pars = tunable_parameters(simmodel)
# settunable = (p, tag) -> setmetadata(p, ModelingToolkit.VariableTunable, tag)
# p_new = map(p -> settunable(p, ~occursin("₊ou₊", string(p))), pars)
# @named fullmodel_infer = System(equations(simmodel), ModelingToolkit.get_iv(simmodel), unknowns(simmodel), addnontunableparams(p_new, simmodel))
# fullmodel_infer = structural_simplify(fullmodel_infer, split=false)

# attribute initial conditions to states
sts, idx_sts = get_dynamic_states(fitmodel)
idx_u = get_idx_tagged_vars(fitmodel, "ext_input")         # get index of external input state
idx_bold = get_eqidx_tagged_vars(fitmodel, "measurement")  # get index of equation of bold state

# the following step is needed if the model's Jacobian would give degenerate eigenvalues if expanded around 0 (which is the default expansion)
perturbedfp = Dict(sts .=> abs.(0.001*rand(length(sts))))     # slight noise to avoid issues with Automatic Differentiation. TODO: find different solution, this is hacky.

modelparam = OrderedDict()
for par in tunable_parameters(fitmodel)
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
        paramvariance[k] = ones(length(v));
    elseif occursin("κ", string(k))
        paramvariance[k] = ones(length(v))./256.0;
    elseif occursin("ϵ", string(k))
        paramvariance[k] = 1/256.0;
    elseif occursin("τ", string(k))
        paramvariance[k] = 1/256.0;
    end
end

priors = DataFrame(name=[k for k in keys(modelparam)], mean=[m for m in values(modelparam)], variance=[v for v in values(paramvariance)])
hyperpriors = Dict(:Πλ_pr => 128.0*ones(1, 1),   # prior metaparameter precision, needs to be a matrix
                   :μλ_pr => [8.0]               # prior metaparameter mean, needs to be a vector
                  );

csdsetup = Dict(:p => p, :freq => freq, :dt => dt);

(state, setup) = setup_sDCM(data, fitmodel, perturbedfp, csdsetup, priors, hyperpriors, indices, "fMRI");

# HACK: on machines with very small amounts of RAM, Julia can run out of stack space while compiling the code called in this loop
# this should be rewritten to abuse the compiler less, but for now, an easy solution is just to run it with more allocated stack space.
with_stack(f, n) = fetch(schedule(Task(f,n)))

with_stack(5_000_000) do  # 5MB of stack space
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
end

# plot results
using CairoMakie

results = unvecparam(state.μθ_po, modelparam)
μA = state.μθ_po[1:nr^2]
μA[1:(nr+1):nr^2] = -exp.(μA[1:(nr+1):nr^2])/2    # transform into a 
ΣA = state.Σθ_po[1:nr^2, 1:nr^2]

xlabels = string.(collect(keys(modelparam))[1:nr^2])
CairoMakie.barplot(1:nr^2, μA,
    axis = (xticks = (1:nr^2, xlabels),
    title = "Effective connectivity")
)