using ForwardDiff: jacobian
# using GenericLinearAlgebra
using LinearAlgebra
using FFTW
using ToeplitzMatrices


function transferfunction(x, w, θμ, C, lnϵ, lndecay, lntransit)
    # compute transfer function of Volterra kernels, see fig 1 in friston2014
    # 1. compute jacobian w.r.t. f ; TODO: what is it with this "delay operator" that is set to 1 in "spm_fx_fmri.m"
    # J_x = jacobian(f, x0) # well, no need to perform this for a linear system... we already have it: θμ
    C /= 16.0   # TODO: unclear why it is devided by 16 but see spm_fx_fmri.m:49
    # 2. get jacobian of hemodynamics

	@show typeof(θμ)
    F = eigen(θμ)   #  , sortby=nothing, permute=false, scale=false)
	@show typeof(F)
    return F.values
end

w = rand(32)
θμ = randn(3,3)
dim = size(θμ, 1)
C = ones(Float64, dim)
lnϵ = 0.0                       # BOLD signal parameter
lndecay, lntransit = [0.0, 0.0]    # hemodynamic parameters
x = zeros(Float64, 3, 5)


testfun = foo -> transferfunction(x, w, foo, C, lnϵ, lndecay, lntransit)
J_test = jacobian(testfun, θμ)