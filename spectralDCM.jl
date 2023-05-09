using LinearAlgebra
using MKL
using FFTW
using ToeplitzMatrices
using MAT
using ExponentialUtilities
using ForwardDiff
using BenchmarkTools

# Questions for Chris:
# 1. what does it mean to cache *
# 2. cached zeros just by moving them one loop out
# 3. how do we deal with ADVI problem?
# 4. maybe also @views
# 5. how about removing bound checks?



# simple dispatch for vec to deal with 1xN matrices
function Base.vec(x::T) where (T <: Real)
    return x*ones(1)
end

include("src/hemodynamic_response.jl")     # hemodynamic and BOLD signal model
include("src/VariationalBayes_AD.jl")      # this can be switched between _spm12 and _AD version. There is also a separate ADVI version in VariationalBayes_ADVI.jl
include("src/mar.jl")                      # multivariate auto-regressive model functions

foo = Ref{Any}()
backintime = Ref{Any}()

### get data and compute cross spectral density which is the actual input to the spectral DCM ###
vars = matread("speedandaccuracy/matlab0.01_3regions.mat");
y = vars["data"];
dt = vars["dt"];
freqs = vec(vars["Hz"]);
p = 8;                               # order of MAR, it is hard-coded in SPM12 with this value. We will just use the same for now.
mar = mar_ml(y, p);                  # compute MAR from time series y and model order p
y_csd = mar2csd(mar, freqs, dt^-1);  # compute cross spectral densities from MAR parameters at specific frequencies freqs, dt^-1 is sampling rate of data
y_csd = vars["csd"]
### Define priors and initial conditions ###
x = vars["x"];                       # initial condition of dynamic variabls
A = vars["pE"]["A"];                 # initial values of connectivity matrix
θΣ = vars["pC"];                     # prior covariance of parameter values 
λμ = vec(vars["hE"]);                # prior mean of hyperparameters
Πλ_p = vars["ihC"];                  # prior precision matrix of hyperparameters
if typeof(Πλ_p) <: Number            # typically Πλ_p is a matrix, however, if only one hyperparameter is used it will turn out to be a scalar -> transform that to matrix
    Πλ_p *= ones(1, 1)
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
# param = [p; reshape(A, dim^2); C; lntransit; lndecay; lnϵ; reshape(lnα, dim^2); lnβ; lnγ;];
# Strange α and β sorting? yes. This is to be consistent with the SPM12 code while keeping nomenclature consistent with the spectral DCM paper
Q = csd_Q(y_csd);                 # compute prior of Q, the precision (of the data) components. See Friston etal. 2007 Appendix A
priors = [Πθ_p, Πλ_p, λμ, Q];

### Compute the DCM ###
@time results = variationalbayes(x, y_csd, freqs, V, param, priors, 128)


# @benchmark bar1 = LinearAlgebra.eigen(J_tot)
# function test(J)
#     F = DifferentiableEigen.eigen(J_tot)
#     Λ = DifferentiableEigen.arr2Comp(F[1], size(J_tot, 1))
#     V = DifferentiableEigen.arr2Comp(F[2], size(J_tot))
#     return Eigen(Λ, V)
# end
# @benchmark bar2 = test(J_tot)

N = 100

function test1()
    for i = 1:100
        for j = 1:10
            A = zeros(Float64, N, N, 10)
            A[:,:,j] = rand(N, N)
        end
    end
end

function test2()
    A = zeros(Int, N, N, 10)
    for i = 1:100
        A = Float64.(A)
        for j = 1:10
            A[:,:,j] = rand(N, N)
        end
    end
end

function test3()
    for i = 1:100
        A = zeros(Float64, N, N, 10)
        for j = 1:10
            A[:,:,j] = rand(N, N)
        end
    end
end

@benchmark test1()
@benchmark test2()
@benchmark test3()