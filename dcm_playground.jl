using ModelingToolkit, OrdinaryDiffEq, Plots

# https://juliapackages.com/p/modelingtoolkit
# https://mtk.sciml.ai/stable/systems/ODESystem/




##### Nonlinear System #####

# define Lorenz system 
@parameters t θ1 θ2 θ3
@variables x1(t) x2(t) x3(t)
D = Differential(t)

eqs = [D(x1) ~ θ1 * (x2 - x1),
       D(x2) ~ x1 * (θ2 - x3) - x2,
       D(x3) ~ x1 * x2 - θ3 * x3]

sys = ODESystem(eqs)
# sys = ode_order_lowering(sys)

x0 = [x1 => 1.0,
      x2 => 0.0,
      x3 => 0.0]

p  = [θ1 => 10.0, 
      θ2 => 28.0,
      θ3 => 8/3]

# function lorenz!(dx,x,p,t)
#     dx[1] = p[1] * (x[2] - x[1])
#     dx[2] = x[1] * (p[2] - x[3]) - x[2]
#     dx[3] = x[1] * x[2] - p[3] * x[3]
# end

# x0 = [1.0, 0.0, 0.0]
# p = [10.0, 28.0, 8/3]

tspan = (0.0,100.0)
prob = ODEProblem(sys, x0, tspan, p, jac=true)
nonlinsol = solve(prob, Tsit5())

plot(nonlinsol, vars=(1,2))


##### Linear System #####
using StaticArrays, DifferentialEquations
A  = @SMatrix [-0.5 -0.2  0.0
                0.4 -0.5 -0.3
                0.0  0.2 -0.5]
u0 = rand(3)
tspan = (0.0, 10.0)
f(u,p,t) = A*u

linprob = ODEProblem(f, u0, tspan)
linsol = solve(linprob, saveat=0.1)
plot(linsol)

