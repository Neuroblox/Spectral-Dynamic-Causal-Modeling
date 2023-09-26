using LinearAlgebra
using MKL
using FFTW
using ToeplitzMatrices
using MAT
using ExponentialUtilities
using ForwardDiff
using OrderedCollections

using Turing
using Distributions
using Flux

### a few packages relevant for speed tests and profiling ###
using JLD2


# simple dispatch for vec to deal with 1xN matrices
function Base.vec(x::T) where (T <: Real)
    return x*ones(1)
end

include("../src/hemodynamic_response.jl")     # hemodynamic and BOLD signal model
include("../src/mar.jl")                      # multivariate auto-regressive model functions
include("../src/VariationalBayes_AD.jl")


function wrapperfunction_ADVI(vars, samples, steps)
    y = vars["data"];
    dt = vars["dt"];
    w = vec(vars["Hz"]);
    p = 8;
    mar = mar_ml(y, p);
    y_csd = mar2csd(mar, w, dt^-1);
    x = vars["x"];           # initial condition of dynamic variabls
    dim = size(x, 1);
    Σ = vars["pC"]


    @model function fitADVI_csd(csd_data)
        # set priors of variable parameters
        # Σ ~ InverseWishart(σ_μ, σ_σ)
        # define all priors of parameters
        # A ~ MvNormal(reshape(vars["pE"]["A"], dim^2), Matrix(I, dim^2, dim^2))
        A ~ MvNormal(reshape(vars["pE"]["A"], dim^2), Σ[1:dim^2, 1:dim^2])
        C = zeros(dim);    # NB: whatever C is defined to be here, it will be replaced in csd_approx. A little strange thing of SPM12
        idx = dim^2 + dim;
        lntransit ~ MvNormal(vec(vars["pE"]["transit"]), Σ[idx .+ (1:dim), idx .+ (1:dim)])
        idx += dim;
        lndecay ~ Normal(only(vars["pE"]["decay"]), only(Σ[idx+1,idx+1]))
        lnϵ ~ Normal(only(vars["pE"]["epsilon"]), only(Σ[idx+2,idx+2]))
        idx += 2;
        α ~ MvNormal(vec(vars["pE"]["a"]), Σ[idx .+ (1:2), idx .+ (1:2)])
        idx += 2;
        β ~ MvNormal(vec(vars["pE"]["b"]), Σ[idx .+ (1:2), idx .+ (1:2)])
        idx += 2;
        γ ~ MvNormal(vec(vars["pE"]["c"]), Σ[idx .+ (1:dim), idx .+ (1:dim)])
        # compute cross spectral density
        param = [A; C; lntransit; lndecay; lnϵ; α[1]; β[1]; α[2]; β[2]; γ];
        # observations
        csd = csd_fmri_mtf(x, w, p, param)
        if eltype(csd) <: Dual
            csd = (p->p.value).(csd)
        end


        csd_real = real(vec(csd_data))
        csd_imag = imag(vec(csd_data))
        csd_real ~ MvNormal(real(vec(csd)), Matrix(1.0I, length(csd), length(csd)))
        csd_imag ~ MvNormal(imag(vec(csd)), Matrix(1.0I, length(csd), length(csd)))
    end


    # ADVI
    modelEMn = fitADVI_csd(y_csd)
    Turing.setadbackend(:forwarddiff)
    advi = ADVI(samples, steps)
    setchunksize(8)
    q = vi(modelEMn, advi);
    return (q, advi, modelEMn)
end

# csdapproxvars = Ref{Any}()
ADVIsteps = 1000
ADVIsamples = 10
local vals
n = 3
for iter = 9
    vals = matread("speedandaccuracy/matlab_" * string(n) * "regions.mat");
    t_juliaADVI = @elapsed (q, advi, model) = wrapperfunction_ADVI(vals, ADVIsamples, ADVIsteps)
    save_object("speedandaccuracy/newADVI/ADVI" * string(iter) * "_sa" * string(ADVIsamples) * "_st" * string(ADVIsteps) * "_r" * string(n) * ".jld2", (q, advi, model, t_juliaADVI))
end

# Note for 3 regions
# if I use the native version I improve drastically in parameters and free energy over correct Q and random inits
# if I introduce random initial conditions I get worse parameters and the free energy doubles
# with Q and random inits the accuracy decreases and the free energy halves as compared to the first version
# vals = matread("/home/david/Projects/neuroblox/codes/Spectral-DCM/speedandaccuracy/matlab_3regions.mat");
# q = deserialize("ADVI_3regions.dat")[1]
# z = rand(q, 1000);
# avg = vec(mean(z; dims = 2))
# A_true = vals["true_params"]["A"]
# # A_true = [0 0.0254; 0.0084 0]
# d = size(A_true, 1)
# rms = abs.(reshape(avg[1:d^2], d, d) - A_true)
# rms_Laplace = abs.(vals["Ep"]["A"] - A_true)
