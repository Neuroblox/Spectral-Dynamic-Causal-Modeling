using LinearAlgebra
using MKL
using FFTW
using ToeplitzMatrices
using MAT
using ExponentialUtilities
using ForwardDiff

using Turing
using Distributions
using Flux

### a few packages relevant for speed tests and profiling ###
using Serialization
using StatProfilerHTML


# simple dispatch for vec to deal with 1xN matrices
function Base.vec(x::T) where (T <: Real)
    return x*ones(1)
end

include("../src/hemodynamic_response.jl")     # hemodynamic and BOLD signal model
include("../src/mar.jl")                      # multivariate auto-regressive model functions
include("../src/VariationalBayes_AD.jl")

function wrapperfunction(vars, iter)
    y = vars["data"];
    dt = vars["dt"];
    freqs = vec(vars["Hz"]);
    p = 8;                               # order of MAR, it is hard-coded in SPM12 with this value. We will just use the same for now.
    mar = mar_ml(y, p);                  # compute MAR from time series y and model order p
    y_csd = mar2csd(mar, freqs, dt^-1);  # compute cross spectral densities from MAR parameters at specific frequencies freqs, dt^-1 is sampling rate of data

    ### Define priors and initial conditions ###
    x = vars["x"];                       # initial condition of dynamic variabls
    A = vars["pE"]["A"];                 # initial values of connectivity matrix
    θΣ = vars["pC"];                     # prior covariance of parameter values 
    λμ = vec(vars["hE"]);                # prior mean of hyperparameters
    Πλ_p = vars["ihC"];                  # prior precision matrix of hyperparameters
    if typeof(Πλ_p) <: Number            # typically Πλ_p is a matrix, however, if only one hyperparameter is used it will turn out to be a scalar -> transform that to matrix
        Πλ_p *= ones(1,1)
    end

    # depending on the definition of the priors (note that we take it from the SPM12 code), some dimensions are set to 0 and thus are not changed.
    # Extract these dimensions and remove them from the remaining computation. I find this a bit odd and further thoughts would be necessary to understand
    # to what extend this is legitimate. 
    idx = findall(x -> x != 0, θΣ);
    V = zeros(size(θΣ, 1), length(idx));
    order = sortperm(θΣ[idx], rev=true);
    idx = idx[order];
    for i = 1:length(idx)
        V[idx[i][1], i] = 1.0
    end
    θΣ = V'*θΣ*V;       # reduce dimension by removing columns and rows that are all 0
    Πθ_p = inv(θΣ);

    # define a few more initial values of parameters of the model
    dim = size(A, 1);
    C = zeros(Float64, dim);          # C as in equation 3. NB: whatever C is defined to be here, it will be replaced in csd_approx. Another little strange thing of SPM12...
    lnα = [0.0, 0.0];                 # ln(α) as in equation 2 
    lnβ = [0.0, 0.0];                 # ln(β) as in equation 2
    lnγ = zeros(Float64, dim);        # region specific observation noise parameter
    lnϵ = 0.0;                        # BOLD signal parameter
    lndecay = 0.0;                    # hemodynamic parameter
    lntransit = zeros(Float64, dim);  # hemodynamic parameters
    param = [p; reshape(A, dim^2); C; lntransit; lndecay; lnϵ; lnα[1]; lnβ[1]; lnα[2]; lnβ[2]; lnγ;];
    # Strange α and β sorting? yes. This is to be consistent with the SPM12 code while keeping nomenclature consistent with the spectral DCM paper
    Q = csd_Q(y_csd);                 # compute prior of Q, the precision (of the data) components. See Friston etal. 2007 Appendix A
    priors = [Πθ_p, Πλ_p, λμ, Q];

    ### Compute the DCM ###
    results = variationalbayes(x, y_csd, freqs, V, param, priors, iter)
    return results
end

# function wrapperfunction_ADVI(vars, samples, steps)
#     y = vars["data"];
#     dt = vars["dt"];
#     w = vec(vars["Hz"]);
#     p = 8;
#     mar = mar_ml(y, p);
#     y_csd = mar2csd(mar, w, dt^-1);
#     x = vars["x"];           # initial condition of dynamic variabls
#     dim = size(x, 1);
#     Σ = vars["pC"]
# 
# 
#     @model function fitADVI_csd(csd_data)
#         # set priors of variable parameters
#         # Σ ~ InverseWishart(σ_μ, σ_σ)
#         # define all priors of parameters
#         # A ~ MvNormal(reshape(vars["pE"]["A"], dim^2), Matrix(I, dim^2, dim^2))
#         A ~ MvNormal(reshape(vars["pE"]["A"], dim^2), Σ[1:dim^2, 1:dim^2])
#         C = zeros(dim);    # NB: whatever C is defined to be here, it will be replaced in csd_approx. A little strange thing of SPM12
#         idx = dim^2 + dim;
#         lntransit ~ MvNormal(vec(vars["pE"]["transit"]), Σ[idx .+ (1:dim), idx .+ (1:dim)])
#         idx += dim;
#         lndecay ~ Normal(only(vars["pE"]["decay"]), only(Σ[idx+1,idx+1]))
#         lnϵ ~ Normal(only(vars["pE"]["epsilon"]), only(Σ[idx+2,idx+2]))
#         idx += 2;
#         α ~ MvNormal(vec(vars["pE"]["a"]), Σ[idx .+ (1:2), idx .+ (1:2)])
#         idx += 2;
#         β ~ MvNormal(vec(vars["pE"]["b"]), Σ[idx .+ (1:2), idx .+ (1:2)])
#         idx += 2;
#         γ ~ MvNormal(vec(vars["pE"]["c"]), Σ[idx .+ (1:dim), idx .+ (1:dim)])
#         # compute cross spectral density
#         param = [A; C; lntransit; lndecay; lnϵ; α[1]; β[1]; α[2]; β[2]; γ];
#         # observations
#         csd = csd_fmri_mtf(x, w, p, param)
#         if eltype(csd) <: Dual
#             csd = (p->p.value).(csd)
#         end
# 
# 
#         csd_real = real(vec(csd_data))
#         csd_imag = imag(vec(csd_data))
#         csd_real ~ MvNormal(real(vec(csd)), Matrix(1.0I, length(csd), length(csd)))
#         csd_imag ~ MvNormal(imag(vec(csd)), Matrix(1.0I, length(csd), length(csd)))
#     end
# 
# 
#     # ADVI
#     modelEMn = fitADVI_csd(y_csd)
#     Turing.setadbackend(:forwarddiff)
#     advi = ADVI(samples, steps)
#     setchunksize(8)
#     q = vi(modelEMn, advi, optimizer=Turing.Variational.DecayedADAGrad(1e-2));
#     return (q, advi, modelEMn)
# end
# 
# # csdapproxvars = Ref{Any}()
# ADVIsteps = 1000
# ADVIsamples = 10
# local vals
# n = 3
# for iter = 15
#     vals = matread("matlab0.01_" * string(n) * "regions.mat");
#     t_juliaADVI = @elapsed (q, advi, model) = wrapperfunction_ADVI(vals, ADVIsamples, ADVIsteps)
#     serialize("ADVIADA" * string(iter) * "_sa" * string(ADVIsamples) * "_st" * string(ADVIsteps) * "_0.01_r" * string(n) * ".dat", (q, advi, model, t_juliaADVI))
# end

# Note for 3 regions
# if I use the native version I improve drastically in parameters and free energy over correct Q and random inits
# if I introduce random initial conditions I get worse parameters and the free energy doubles
# with Q and random inits the accuracy decreases and the free energy halves as compared to the first version
# vals = matread("/home/david/Projects/neuroblox/codes/Spectral-DCM/speedandaccuracy/matlab_3regions.mat");
# q = deserialize("ADVI_3regions.dat")[1]
# z = rand(q, 1000);
# avg = vec(mean(z; dims = 2))
# A_true = vals["true_params"]["A"]
# #A_true = [0 0.0254; 0.0084 0]
# d = size(A_true, 1)
# rms = abs.(reshape(avg[1:d^2], d, d) - A_true)
# rms_Laplace = abs.(vals["Ep"]["A"] - A_true)




# using StatsPlots

# samples_j = []
# samples_m = []
# N = 10000
# for i = 2:2
#     vals = matread("/home/david/Projects/neuroblox/codes/Spectral-DCM/speedandaccuracy/nregions2.mat");
#     matlab = rand(Normal.(vec(vals["Ep"]["A"]), collect(diag(vals["Cp"][1:i^2, 1:i^2]))), N)
#     z = rand(deserialize("ADVI_" * string(i) * "regions_old.dat")[1], N)
#     push!(samples_j, z[1:i^2, :])
#     push!(samples_m, matlab)
# end
# X = [:a12, :a21]
# StatsPlots.violin(["a12"], samples[1][2,:])
# StatsPlots.violin!(["a21"], samples[1][3,:])

for n in vcat(2:10, 15, 30)
    vals = matread("matlab0.01_" * string(n) *"regions.mat");
    include("../src/VariationalBayes_AD.jl")      # this can be switched between _spm12 and _AD version. There is also a separate ADVI version in VariationalBayes_ADVI.jl
    wrapperfunction(vals, 1)
    t_juliaAD = @elapsed res_AD = wrapperfunction(vals, 128)
    include("../src/VariationalBayes_spm12.jl")      # this can be switched between _spm12 and _AD version. There is also a separate ADVI version in VariationalBayes_ADVI.jl
    wrapperfunction(vals, 1)
    t_juliaSPM = @elapsed res_spm = wrapperfunction(vals, 128)
    @show t_juliaAD, t_juliaSPM

    matwrite("PMn" * string(n) * ".mat", Dict(
        "t_mat" => vals["matcomptime"],
        "F_mat" => vals["F"],
        "t_jad" => t_juliaAD,
        "F_jad" => res_AD.F,
        "t_jspm" => t_juliaSPM,
        "F_jspm" => res_spm.F,
        "iter_spm" => res_spm.iter,
        "iter_ad" => res_AD.iter
    ); compress = true)    
end


# file = matopen("speedandaccuracy/nregions" * n * ".mat")
# t_matlab = read(file, "matcomptime")
# close(file)
# iter = 20


### Profiling ###
include("../src/VariationalBayes_AD.jl")

n = 3
vars = matread("speedandaccuracy/matlab0.01_" * string(n) *"regions.mat");
y = vars["data"];
dt = vars["dt"];
freqs = vec(vars["Hz"]);
p = 8;                               # order of MAR, it is hard-coded in SPM12 with this value. We will just use the same for now.
mar = mar_ml(y, p);                  # compute MAR from time series y and model order p
y_csd = mar2csd(mar, freqs, dt^-1);  # compute cross spectral densities from MAR parameters at specific frequencies freqs, dt^-1 is sampling rate of data

### Define priors and initial conditions ###
x = vars["x"];                       # initial condition of dynamic variabls
A = vars["pE"]["A"];                 # initial values of connectivity matrix
θΣ = vars["pC"];                     # prior covariance of parameter values 
λμ = vec(vars["hE"]);                # prior mean of hyperparameters
Πλ_p = vars["ihC"];                  # prior precision matrix of hyperparameters
if typeof(Πλ_p) <: Number            # typically Πλ_p is a matrix, however, if only one hyperparameter is used it will turn out to be a scalar -> transform that to matrix
    Πλ_p *= ones(1,1)
end

# depending on the definition of the priors (note that we take it from the SPM12 code), some dimensions are set to 0 and thus are not changed.
# Extract these dimensions and remove them from the remaining computation. I find this a bit odd and further thoughts would be necessary to understand
# to what extend this is legitimate. 
idx = findall(x -> x != 0, θΣ);
V = zeros(size(θΣ, 1), length(idx));
order = sortperm(θΣ[idx], rev=true);
idx = idx[order];
for i = 1:length(idx)
    V[idx[i][1], i] = 1.0
end
θΣ = V'*θΣ*V;       # reduce dimension by removing columns and rows that are all 0
Πθ_p = inv(θΣ);

# define a few more initial values of parameters of the model
dim = size(A, 1);
C = zeros(Float64, dim);          # C as in equation 3. NB: whatever C is defined to be here, it will be replaced in csd_approx. Another little strange thing of SPM12...
lnα = [0.0, 0.0];                 # ln(α) as in equation 2 
lnβ = [0.0, 0.0];                 # ln(β) as in equation 2
lnγ = zeros(Float64, dim);        # region specific observation noise parameter
lnϵ = 0.0;                        # BOLD signal parameter
lndecay = 0.0;                    # hemodynamic parameter
lntransit = zeros(Float64, dim);  # hemodynamic parameters
param = [p; reshape(A, dim^2); C; lntransit; lndecay; lnϵ; lnα[1]; lnβ[1]; lnα[2]; lnβ[2]; lnγ;];
# Strange α and β sorting? yes. This is to be consistent with the SPM12 code while keeping nomenclature consistent with the spectral DCM paper
Q = csd_Q(y_csd);                 # compute prior of Q, the precision (of the data) components. See Friston etal. 2007 Appendix A
priors = [Πθ_p, Πλ_p, λμ, Q];
variationalbayes(x, y_csd, freqs, V, param, priors, 26)

@profilehtml results = variationalbayes(x, y_csd, freqs, V, param, priors, 26)
