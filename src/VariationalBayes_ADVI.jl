using Turing
using Distributions
using MAT
using LinearAlgebra
using MKL
using FFTW
using ToeplitzMatrices
using ExponentialUtilities
using ForwardDiff
using Plots

include("hemodynamic_response.jl")
include("VariationalBayes_AD.jl")
include("mar.jl")

function Base.vec(x::T) where (T <: Real)
    return x*ones(1)
end

vars = matread("/home/david/Projects/neuroblox/codes/Spectral-DCM/speedandaccuracy/nregions3.mat");
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

foo = Ref{Any}()


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
    A ~ MvNormal(reshape(vars["pE"]["A"], dim^2), vars["pC"][1:9, 1:9])
    C = zeros(dim);    # NB: whatever C is defined to be here, it will be replaced in csd_approx. A little strange thing of SPM12
    # compute cross spectral density
    param = [A; C; lntransit; lndecay; lnϵ; α[1]; β1; α[2]; β2; γ];
    # observations
    csd = csd_fmri_mtf(x, w, p, param)
    if eltype(csd) <: Dual
        csd = (p->p.value).(csd)
    end

    csd_real = real(vec(csd_data))
    csd_imag = imag(vec(csd_data))
    # for i = 1:length(csd_data)
    #     csd_real[i] ~ Normal(real(csd[i]), 0.5)   # models sampling noise not observational noise (see paper)
    #     csd_imag[i] ~ Normal(imag(csd[i]), 0.5)
    # end
    # data = vec(csd_data)
    # Main.foo[] = csd_sim_imag, csd_sim_real, csd_real, csd_imag, csd
    csd_real ~ MvNormal(real(vec(csd)), Matrix(1.0I, length(csd), length(csd)))
    csd_imag ~ MvNormal(imag(vec(csd)), Matrix(1.0I, length(csd), length(csd)))
    # data ~ MvNormal(vec(csd), Matrix((1.0 + 1.0im)I, length(csd), length(csd)))
end

# ADVI
modelEMn = fitADVI_csd(y_csd)
Turing.setadbackend(:forwarddiff)
advi = ADVI(50, 1000)
setchunksize(8)
q = vi(modelEMn, advi);

chain = sample(modelEMn, NUTS(), 1000);

# sampling
z = rand(q, 10000);
avg = vec(mean(z; dims = 2))
zstd = vec(std(z; dims = 2))
el = 597
p1 = histogram(z[14, :]; bins=100, normed=true, alpha=0.2, color=:blue, label="")
density!(z[14, :]; label="s (ADVI)", color=:blue, linewidth=2)