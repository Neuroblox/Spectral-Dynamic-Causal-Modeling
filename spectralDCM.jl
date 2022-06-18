using LinearAlgebra
using MKL
using FFTW
using ToeplitzMatrices
using MAT
using ExponentialUtilities
using ForwardDiff

# using Serialization
# using Profile
# using ProfileView
# using BenchmarkTools

include("src/hemodynamic_response.jl")
include("src/VariationalBayes_AD.jl")
include("src/mar.jl")

function Base.vec(x::T) where (T <: Real)
    return x*ones(1)
end
### DEFINE SEVERAL VARIABLES AND PRIORS TO GET STARTED ###
vars = matread("/home/david/Projects/neuroblox/codes/Spectral-DCM/spectralDCM_demodata.mat");
# vars = matread("/home/david/Projects/neuroblox/data/fMRIdata/Bernal-Casas/timeseries_D1.mat")
# function speedtest(vars)
    y = vars["data"];
    dt = vars["dt"];
    freqs = vec(vars["Hz"]);
    p = 8;
    mar = mar_ml(y,p);
    y_csd = mar2csd(mar, freqs, dt^-1);

    x = vars["x"];           # initial condition of dynamic variabls
    A = vars["pE"]["A"];
    A += Matrix(1.0I,size(A));       # add some randomness to the prior to avoid degenerate eigenvalues later on
    A .*= 0.01 .+ 0.1*randn(size(A));
    θΣ = vars["pC"];
    λμ = vec(vars["hE"]);
    Πλ_p = vars["ihC"];
    if typeof(Πλ_p) <: Number
        Πλ_p *= ones(1,1)
    end

    idx = findall(x -> x != 0, θΣ);
    V = zeros(size(θΣ, 1), length(idx));
    order = sortperm(θΣ[idx], rev=true);
    idx = idx[order];
    for i = 1:length(idx)
        V[idx[i][1], i] = 1.0
    end
    θΣ = V'*θΣ*V;
    Πθ_p = inv(θΣ);

    dim = size(A, 1);
    C = zeros(Float64, dim);    # NB: whatever C is defined to be here, it will be replaced in csd_approx. A little strange thing of SPM12
    α = [0.0, 0.0];
    β = [0.0, 0.0];
    γ = zeros(Float64, dim);
    lnϵ = 0.0;                        # BOLD signal parameter
    lndecay = 0.0;                    # hemodynamic parameter
    lntransit = zeros(Float64, dim);  # hemodynamic parameters
    param = [p; reshape(A, dim^2); C; lntransit; lndecay; lnϵ; α[1]; β[1]; α[2]; β[2]; γ;];
    # Strange α and β sorting? yes. This is to be consistent with the SPM12 code while keeping nomenclature consistent with the spectral DCM paper
    Q = csd_Q(y_csd);
    priors = [Πθ_p, Πλ_p, λμ, Q];

    results = variationalbayes(x, y_csd, freqs, V, param, priors, 128)
    # return results
# end

# speedtest(vars)
# # ProfileView.@profview results = variationalbayes(x, y_csd, freqs, V, param, priors, 26)

# # res = @benchmark speedtest(vars)
# # t_julia = mean(res.times./10^9);
# t_julia = @elapsed speedtest(vars)
# n = "14"
# file = matopen("data_speedtest/nregions" * n * ".mat")
# t_matlab = read(file, "matcomptime")
# close(file)
# iter = 20
# matwrite("data_speedtest/n" * n * ".mat", Dict(
# 	"t_matlab" => t_matlab,
# 	"t_julia" => t_julia,
#     "iter" => iter
# ); compress = true)
