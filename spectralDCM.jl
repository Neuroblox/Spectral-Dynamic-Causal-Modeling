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
Random.seed!(101);
# Questions for Chris:
# 1. what does it mean to cache *
# 2. cached zeros just by moving them one loop out
# 3. how do we deal with ADVI problem?
# 4. maybe also @views
# 5. how about removing bound checks?
# 6. should I go through the functions and declare types?



# simple dispatch for vec to deal with 1xN matrices
function Base.vec(x::T) where (T <: Real)
    return x*ones(1)
end

include("src/hemodynamic_response.jl")     # hemodynamic and BOLD signal model
include("src/VariationalBayes_AD.jl")      # this can be switched between _spm12 and _AD version. There is also a separate ADVI version in VariationalBayes_ADVI.jl
include("src/mar.jl")                      # multivariate auto-regressive model functions

### get data and compute cross spectral density which is the actual input to the spectral DCM ###
vars = matread("../Spectral-DCM/speedandaccuracy/Anthony/test2.mat");
y = vars["data"];
nd = size(y, 2);
dt = vars["dt"];
freqs = vec(vars["Hz"]);
p = 8;                               # order of MAR, it is hard-coded in SPM12 with this value. We will just use the same for now.
mar = mar_ml(y, p);                  # compute MAR from time series y and model order p
y_csd = mar2csd(mar, freqs, dt^-1);  # compute cross spectral densities from MAR parameters at specific frequencies freqs, dt^-1 is sampling rate of data
# y_csd = vars["csd"];
### Define priors and initial conditions ###
x = vars["x"];                       # initial condition of dynamic variabls
x .+= abs.(0.1randn(size(x)...))
θΣ = vars["pC"];                     # prior covariance of parameter values 
θΣ[1:nd^2, 1:nd^2] = Matrix(I, nd^2, nd^2)
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
A = (A + Matrix(I, size(A)...)) .* (1 .+ 0.1*randn(size(A)...))

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
                         :μλ_pr => vec(vars["hE"]),   # prior metaparameter mean
                         :Q => Q                      # decomposition of model parameter covariance
                        )
             );

### Compute the DCM ###
@time results = variationalbayes(x, y_csd, freqs, V, p, priors, 128);


# @benchmark bar1 = LinearAlgebra.eigen(J_tot)
# function test(J)
#     F = DifferentiableEigen.eigen(J_tot)
#     Λ = DifferentiableEigen.arr2Comp(F[1], size(J_tot, 1))
#     V = DifferentiableEigen.arr2Comp(F[2], size(J_tot))
#     return Eigen(Λ, V)
# end
# @benchmark bar2 = test(J_tot)

# N = 100

# function test1()
#     for i = 1:100
#         for j = 1:10
#             A = zeros(Float64, N, N, 10)
#             A[:,:,j] = rand(N, N)
#         end
#     end
# end

# function test2()
#     A = zeros(Int, N, N, 10)
#     for i = 1:100
#         A = Float64.(A)
#         for j = 1:10
#             A[:,:,j] = rand(N, N)
#         end
#     end
# end

# function test3()
#     for i = 1:100
#         A = zeros(Float64, N, N, 10)
#         for j = 1:10
#             A[:,:,j] = rand(N, N)
#         end
#     end
# end

# @benchmark test1()
# @benchmark test2()
# @benchmark test3()

# function f(y,x) # in-place
#   global fcalls += 1
#   for i in 2:length(x)-1
#     y[i] = x[i-1] - 2x[i] + x[i+1]
#   end
#   y[1] = -2x[1] + x[2]
#   y[end] = x[end-1] - 2x[end]
#   nothing
# end





# using SparseDiffTools
# using ForwardDiff

# fcalls = 0
# function g(x) # out-of-place
#     global fcalls += 1
#     y = zero(x)
#     for i in 2:length(x)-1
#       y[i] = x[i-1] - 2x[i] + x[i+1]
#     end
#     y[1] = -2x[1] + x[2]
#     y[end] = x[end-1] - 2x[end]
#     y
# end


# using Plots

# nvarset = [10, 50, 100, 250, 500]
# ncolset = [10, 50, 100, 250, 500]
# nvarset = vcat(2,10:10:50)
# ncolset = vcat(2, 10:10:50)

# speed_Jfree = zeros(length(nvarset)*length(ncolset))
# speed_Jbased = zeros(length(nvarset)*length(ncolset))
# for (i, (nvars, ncols)) in enumerate(Iterators.product(nvarset, ncolset))
#     x = rand(nvars)
#     J = JacVec(g, x)
#     V = rand(nvars, ncols)
#     tmp = @benchmark stack(J*c for c in eachcol(V))
#     speed_Jfree[i] = mean(tmp.times)*1e-6
#     tmp = @benchmark ForwardDiff.jacobian(g, x)*V
#     speed_Jbased[i] = mean(tmp.times)*1e-6
# end
# p1 = contourf(nvarset, ncolset, speed_Jbased, title="Matrix-based", color=:turbo, clim=(0.0,0.04));
# p2 = contourf(nvarset, ncolset, speed_Jfree, xlabel="number of variables", title="Matrix-free", color=:turbo, clim=(0.0,0.04));
# plot(p1, p2, layout=(2, 1), ylabel="number of columns")


# res = similar(v)
# v = foo[][3][:, 1]
# J = JacVec(foo[][1], foo[][2])
# mul!(res, J, v)
# J* foo[][3]
# dfdp = zeros(Complex, 288, 21)
# for i = 1:size(foo[][3], 2)
#     dfdp[:, i] = J*foo[][3][:, i]
# end