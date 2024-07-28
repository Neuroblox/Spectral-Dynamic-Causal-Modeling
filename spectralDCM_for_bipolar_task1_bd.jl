using LinearAlgebra
using MKL
using FFTW
using ToeplitzMatrices
using MAT
using ExponentialUtilities
using ForwardDiff
using Statistics
#using Plots


# simple dispatch for vec to deal with 1xN matrices
function Base.vec(x::T) where (T <: Real)
    return x*ones(1)
end

include("src/hemodynamic_response.jl")     # hemodynamic and BOLD signal model
include("src/VariationalBayes_spm12.jl")      # this can be switched between _spm12 and _AD version. There is also a separate ADVI version in VariationalBayes_ADVI.jl
include("src/mar.jl")                      # multivariate auto-regressive model functions
file_list = readdir("/shared/home/anthony/bipolar_dcm_temp/task1_7roi_dcm_bd/")
file_list = file_list
#file_list = file_list[2:end]
for file in file_list
#file = "sub-1659988_first_try_DCM.mat"
#file = "hmm_4roi_noisy_v4.mat"
# for field in [3, 7]
# fitField = 9.4

# for regionNum in [4]
#     for runNum in 2:50

### get data and compute cross spectral density which is the actual input to the spectral DCM ###
#vars = matread("/Users/achesebro/Documents/GitHub/extra_hemodynamic_sims/matlab"*string(field)*"T_"*string(regionNum)*"regions_v"*lpad(runNum, 2, "0")*".mat");
vars = matread("/shared/home/anthony/bipolar_dcm_temp/task1_7roi_dcm_bd/"*file);
y = vars["data"];
dt = vars["dt"];
freqs = vec(vars["Hz"]);
p = 8;                               # order of MAR, it is hard-coded in SPM12 with this value. We will just use the same for now.

# detrending
y = y .- mean(y, dims=1)
# rescaling to a scale of 4 as is done in SPM12:
y = y.*((std(y[:])*4)^(-1))
println("data loaded")

mar = mar_ml(y, p);                  # compute MAR from time series y and model order p
#mar = estimate_mar(y, p, 1)
y_csd = mar2csd(mar, freqs, dt^-1);  # compute cross spectral densities from MAR parameters at specific frequencies freqs, dt^-1 is sampling rate of data
#y_csd2 = vars["csd"]
### Define priors and initial conditions ###
x = vars["x"];                       # initial condition of dynamic variabls
A = vars["pE"]["A"];                 # initial values of connectivity matrix
θΣ = vars["pC"];                     # prior covariance of parameter values 
λμ = vec(vars["hE"]);                # prior mean of hyperparameters
Πλ_p = vars["ihC"];                  # prior precision matrix of hyperparameters
if typeof(Πλ_p) <: Number            # typically Πλ_p is a matrix, however, if only one hyperparameter is used it will turn out to be a scalar -> transform that to matrix
    Πλ_p *= ones(1, 1)
end
println("priors loaded")

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
println("priors reduced")

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
# param = [p; reshape(A, dim^2); C; lntransit; lndecay; lnϵ; reshape(lnα, dim^2); lnβ; lnγ;];
# Strange α and β sorting? yes. This is to be consistent with the SPM12 code while keeping nomenclature consistent with the spectral DCM paper
Q = csd_Q(y_csd);                 # compute prior of Q, the precision (of the data) components. See Friston etal. 2007 Appendix A
priors = [Πθ_p, Πλ_p, λμ, Q];
println("priors defined")

### Compute the DCM ###
@time results = variationalbayes(x, y_csd, freqs, V, param, priors, 128)


file = matopen("/shared/home/anthony/bipolar_dcm_temp/task1_7roi_dcm_bd_results/julia_fit_"*file, "w")
write(file, "juliaFit", results.μθ_po)
close(file)
end
# end
# end
# end