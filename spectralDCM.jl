using LinearAlgebra
using FFTW
using ToeplitzMatrices
using MAT
using ExponentialUtilities
using Serialization

include("hemodynamic_response.jl")
include("VariationalBayes_for_spectralDCM.jl")


function csd_Q(csd)
    s = size(csd)
    Qn = length(csd)
    Q = zeros(ComplexF64, Qn, Qn);
    idx = CartesianIndices(csd)
    for Qi  = 1:Qn
        for Qj = 1:Qn
            if idx[Qi][1] == idx[Qj][1]
                Q[Qi,Qj] = csd[idx[Qi][1], idx[Qi][2], idx[Qj][2]]*csd[idx[Qi][1], idx[Qi][3], idx[Qj][3]]
            end
        end
    end
    Q = inv(Q .+ matlab_norm(Q, 1)/32*Matrix(I, size(Q)))   # TODO: MATLAB's and Julia's norm function are different! Reconciliate?
    return Q
end


### DEFINE SEVERAL VARIABLES AND PRIORS TO GET STARTED ###

vars = matread("/home/david/Projects/neuroblox/codes/Spectral-DCM/spectralDCM_demodata_notsparse.mat")
y_csd = vars["csd"];
w = vec(vars["M_nosparse"]["Hz"]);
A = vars["M_nosparse"]["pE"]["A"];    # see table 1 in friston2014 for values of priors 
θΣ = vars["M_nosparse"]["pC"];
λμ = vec(vars["M_nosparse"]["hE"]);
Πλ_p = vars["M_nosparse"]["ihC"];

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
p = 8;
α = [0.0, 0.0];
β = [0.0, 0.0];
γ = zeros(Float64, dim);
lnϵ = 0.0;                        # BOLD signal parameter
lndecay = 0.0;                    # hemodynamic parameter
lntransit = zeros(Float64, dim);  # hemodynamic parameters
x = zeros(Float64, 3, 5);
param = [p; reshape(A, dim^2); C; lntransit; lndecay; lnϵ; α[1]; β[1]; α[2]; β[2]; γ;]
# Strange α and β sorting? yes. This is to be consistent with the SPM12 code while keeping nomenclature consistent with the spectral DCM paper
priors = [Πθ_p, Πλ_p, λμ]


results = VariationalBayes(x, y_csd, w, V, param, priors, 5)


res = matread("/home/david/Projects/neuroblox/data/LFPdata/Session LP052410/Stationary Data/Labeled - Pre/results.mat")
rp = zeros(size(res["results_int"],1), 6)
m = zeros(3,3)
for i = 1:size(rp,1)
    rp[i,:] = vec(res["results_int"][i,4])[[2 3 4 6 7 8]]
    m .+= res["results_int"][i,4]
end
plot(rp, label=["a_11" "a_12" "a_13" "a_21" "a_22" "a_23" "a_31" "a_32" "a_33"], lw=2)
plot(rp, label=["a_12" "a_13" "a_21" "a_23" "a_31" "a_32"], lw=2)
xlabel!("samples interval")
ylabel!("parameter value")
title!("20000 samples window - off-diagonal elements")
savefig("steadycheck_offdiag.png")