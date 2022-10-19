using Turing
using Distributions
using MAT
using LinearAlgebra
using MKL
using FFTW
using ToeplitzMatrices
using ExponentialUtilities
using ForwardDiff

include("hemodynamic_response.jl")
include("VariationalBayes_AD.jl")
include("mar.jl")

function Base.vec(x::T) where (T <: Real)
    return x*ones(1)
end

vars = matread("/home/david/Projects/neuroblox/codes/Spectral-DCM/spectralDCM_demodata.mat");
# vars = matread("/home/david/Projects/neuroblox/data/fMRIdata/Bernal-Casas/timeseries_D1.mat")
y = vars["data"];
dt = vars["dt"];
w = vec(vars["Hz"]);
p = 8;
mar = mar_ml(y, p);
y_csd = mar2csd(mar, w, dt^-1);
x = vars["x"];           # initial condition of dynamic variabls
dim = size(x, 1);

σ_μ = 1.0
σ_σ = 1.0


@model function fitADVI_csd(csd_data)
    # set priors of variable parameters
    # Σ ~ InverseWishart(σ_μ, σ_σ)
    # define all priors of parameters
    α ~ MvNormal([0.0, 0.0], Matrix(I, 2, 2))
    β1 ~ Uniform(0.0, 2.0)
    β2 ~ Uniform(0.0, 2.0)
    γ ~ MvNormal(zeros(dim), Matrix(I, dim, dim))
    lnϵ ~ Normal(0.0, 1.0)
    lndecay ~ Normal(0.0, 1.0)
    lntransit ~ MvNormal(zeros(dim), Matrix(I, dim, dim))
    A ~ MvNormal(reshape(vars["pE"]["A"], dim^2), Matrix(I, dim^2, dim^2))
    C = zeros(dim);    # NB: whatever C is defined to be here, it will be replaced in csd_approx. A little strange thing of SPM12
    # compute cross spectral density
    param = [A; C; lntransit; lndecay; lnϵ; α[1]; β1; α[2]; β2; γ];
    # observations
    csd = csd_fmri_mtf(x, w, p, param)
    csd_real = real(csd_data)
    csd_imag = imag(csd_data)
    for i = 1:length(csd_data)
        csd_real[i] ~ Normal(real(csd[i]), 0.5)   # models sampling noise not observational noise (see paper)
        csd_imag[i] ~ Normal(imag(csd[i]), 0.5)
    end
    # data = vec(csd_data)
    # data ~ MvNormal(vec(csd), Matrix((1.0 + 1.0im)I, length(csd), length(csd)))
end

# ADVI
modelEMn = fitADVI_csd(y_csd)
Turing.setadbackend(:forwarddiff)
advi = ADVI(10, 5000)
setchunksize(4)
q = vi(modelEMn, advi);

# sampling
z = rand(q, 1000);
avg = vec(mean(z; dims = 2))
zstd = vec(std(z; dims = 2))
