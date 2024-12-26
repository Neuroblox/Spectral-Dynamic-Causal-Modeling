using LinearAlgebra
using MKL
using FFTW
using ToeplitzMatrices
using MAT
using ExponentialUtilities
using ForwardDiff
using BenchmarkTools
using OrderedCollections
using SparseDiffTools

include("src/models/hemodynamic_response.jl")     # hemodynamic and BOLD signal model
include("src/VariationalBayes_spm12.jl")             # this can be switched between _spm12 and _AD version. There is also a separate ADVI version in VariationalBayes_ADVI.jl
include("src/utils/mar.jl")                       # multivariate auto-regressive model functions

### get data and compute cross spectral density which is the actual input to the spectral DCM ###
vars = matread("./speedandaccuracy/spm12_demo.mat");

y = vars["data"];
nd = size(y, 2);
dt = vars["dt"];
freqs = vec(vars["Hz"]);
p = 8;                               # order of MAR, it is hard-coded in SPM12 with this value. We will just use the same for now.
mar = mar_ml(y, p);                  # compute MAR from time series y and model order p
y_csd = mar2csd(mar, freqs, dt^-1);  # compute cross spectral densities from MAR parameters at specific frequencies freqs, dt^-1 is sampling rate of data
y_csd = vars["csd"];
### Define priors and initial conditions ###
x = vars["x"];                       # initial condition of dynamic variabls
# x .+= abs.(0.1randn(size(x)...))
θΣ = vars["pC"];                     # prior covariance of parameter values 
# θΣ[1:nd^2, 1:nd^2] = Matrix(I, nd^2, nd^2)
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

Πλ_p = vars["ihC"];                  # prior precision matrix of hyperparameters
if typeof(Πλ_p) <: Number            # typically Πλ_p is a matrix, however, if only one hyperparameter is used it will turn out to be a scalar -> transform that to matrix
    Πλ_p *= ones(1, 1)
end

Q = csd_Q(y_csd);                 # compute prior of Q, the precision (of the data) components. See Friston etal. 2007 Appendix A

A = vars["pE"]["A"]
# A = (A + Matrix(I, size(A)...)) .* (1 .+ 0.1*randn(size(A)...))

priors = Dict(:μ => OrderedDict{Any, Any}(
                                             :A => A,      # prior mean of connectivity matrix
                                             :C => ones(Float64, nd),    # C as in equation 3. NB: whatever C is defined to be here, it will be replaced in csd_approx. Another strange thing of SPM12...
                                             :lnτ => zeros(Float64, nd), # hemodynamic transit parameter
                                             :lnκ => 0.0,                # hemodynamic decay time
                                             :lnϵ => 0.0,                # BOLD signal ratio between intra- and extravascular signal
                                             :lnα => [0.0, 0.0],         # intrinsic fluctuations, ln(α) as in equation 2 of Friston et al. 2014
                                             :lnβ => [0.0, 0.0],         # global observation noise, ln(β) as above
                                             :lnγ => zeros(Float64, nd)  # region specific observation noise
                                            ),
              :Σ => Dict(
                         :Πθ_pr => inv(θΣ),           # prior model parameter precision
                         :Πλ_pr => Πλ_p,              # prior metaparameter precision
                         :μλ_pr => [vars["hE"]],      # prior metaparameter mean
                         :Q => Q                      # decomposition of model parameter covariance
                        )
             );

### Compute the DCM ###
@time results = variationalbayes(x, y_csd, freqs, V, p, priors, 128);
