using ExponentialUtilities: expv
using ForwardDiff: jacobian

function f!(dx, x::Vector, θ::Vector, t)
    @show typeof(x) typeof(θ) typeof(t)
    dx[1] = θ[1] * (x[2] - x[1])
    dx[2] = x[1] * (θ[2] - x[3]) - x[2]
    dx[3] = x[1] * x[2] - θ[3] * x[3]
end

function fx_int(f!, x, t, θ)
    fx! = (dx, x) -> f!(dx, x, θ, 0)
    dx = similar(x)
    J_x = x -> jacobian(fx!, dx, x)
    x_int = expv(t, J_x(x), x)
    return x_int
end

μθ = [10.0, 28.0, 8/3]     # parameter
x0 = [1., 1., 1.]           # initial condition
dt = 0.0001                 # temporal resolution

# redefine to an unary function (https://juliadiff.org/ForwardDiff.jl/stable/user/limitations/)
f_θ = θ -> fx_int(f!, x0, dt, θ)
# compute jacobian of differential equation solution w.r.t. parameters
J_θ = θ -> jacobian(f_θ, θ)
J_θ(μθ)