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
include("src/models/neuraldynamics_MTK.jl")
include("src/models/measurement_MTK.jl")
include("src/utils/MTK_utilities.jl")


### get data and compute cross spectral density which is the actual input to the spectral DCM ###
vars = matread("speedandaccuracy/matlab_cmc.mat");
data = vars["data"];
nd = size(data, 2);
dt = vars["dt"];
freqs = vec(vars["Hz"]);
max_iter = 128                       # maximum number of iterations

### Define priors and initial conditions ###
x = vars["x"];                       # initial condition of dynamic variabls

########## assemble the model ##########
g = MetaDiGraph()
global_ns = :g # global namespace
regions = Dict()

@parameters lnr = 0.0
@parameters lnτ_ss=0 lnτ_sp=0 lnτ_ii=0 lnτ_dp=0
@parameters C=512.0 [tunable = false]    # TODO: SPM12 has this seemingly arbitrary 512 pre-factor in spm_fx_cmc.m. Can we understand why?
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
@parameters a_sp_ss[1:nl] = repeat([1/2.0], nl) # forward connection parameter sp -> ss: sim value 1/2
@parameters a_sp_dp[1:nl] = repeat([3/2.0], nl) # forward connection parameter sp -> dp: sim value 3/2
@parameters a_dp_sp[1:nl] = repeat([1/16.0], nl)  # backward connection parameter dp -> sp: sim value 1/16
@parameters a_dp_ii[1:nl] = repeat([3.0], nl) # backward connection parameters dp -> ii: sim value 3

k = 0
for i in 1:nd
    for j in (i+1):nd
        k += 1
        # forward connection matrix
        add_edge!(g, regions[i], regions[j], :weightmatrix,
                [0 exp(a_sp_ss[k]) 0 0;            # connection from sp to ss
                0 0 0 0;
                0 0 0 0;
                0 exp(a_sp_dp[k])/2 0 0] * 200)    # connection from sp to dp
        # backward connection matrix
        add_edge!(g, regions[j], regions[i], :weightmatrix,
                [0 0 0 0;
                0 0 0 -exp(a_dp_sp[k]);            # connection from dp to sp
                0 0 0 -exp(a_dp_ii[k])/2;          # connection from dp to ii
                0 0 0 0] * 200)
    end
end

@named fullmodel = system_from_graph(g)
fullmodel = structural_simplify(fullmodel, split=false)

# attribute initial conditions to states
sts, idx_sts = get_dynamic_states(fullmodel)
idx_u = get_idx_tagged_vars(fullmodel, "ext_input")                # get index of external input state
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
np = sum(tunable_parameters(fullmodel); init=0) do par
    val = Symbolics.getdefaultval(par)
    modelparam[par] = val
    length(val)
end
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
# HACK: on machines with very small amounts of RAM, Julia can run out of stack space while compiling the code called in this loop
# this should be rewritten to abuse the compiler less, but for now, an easy solution is just to run it with more allocated stack space.
with_stack(f, n) = fetch(schedule(Task(f,n)))

with_stack(5_000_000) do  # 5MB of stack space
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
end