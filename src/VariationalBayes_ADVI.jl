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
using Random
Random.seed!(3);


include("hemodynamic_response.jl")
include("VariationalBayes_AD.jl")
include("mar.jl")

function Base.vec(x::T) where (T <: Real)
    return x*ones(1)
end

vars = matread("/home/david/Projects/neuroblox/codes/Spectral-DCM/speedandaccuracy/nregions3.mat");
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

foo = Ref{Any}()
backintime = Ref{Any}()

# ADVI
modelEMn = fitADVI_csd(y_csd)
Turing.setadbackend(:forwarddiff)

ADVIsteps = 1000
ADVIsamples = 10
advi = ADVI(ADVIsamples, ADVIsteps) 
setchunksize(8)
q = vi(modelEMn, advi);
iter = 1
serialize("ADVIdata" * string(iter) * "_sa" * string(ADVIsamples) * "_st" * string(ADVIsteps) * ".dat", (q, advi, model))

# chain = sample(modelEMn, NUTS(), 1000);

# sampling
# z = rand(q, 10000);
# avg = vec(mean(z; dims = 2))
# zstd = vec(std(z; dims = 2))
# el = 597
# p1 = histogram(z[14, :]; bins=100, normed=true, alpha=0.2, color=:blue, label="")
# density!(z[14, :]; label="s (ADVI)", color=:blue, linewidth=2)
