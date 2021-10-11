using ModelingToolkit, OrdinaryDiffEq, Plots

# https://juliapackages.com/p/modelingtoolkit
# https://mtk.sciml.ai/stable/systems/ODESystem/


# Simulate the BOLD signal
function spm_gx_fmri(x, ϵ)
    """
    Simulated BOLD response to input      
    FORMAT [g,dgdx] = spm_gx_fmri(x,u,P,M)
    g          - BOLD response (%)
    x          - state vector     (see spm_fx_fmri)
    P          - Parameter vector (see spm_fx_fmri)

    This function implements the BOLD signal model described in: 

    Stephan KE, Weiskopf N, Drysdale PM, Robinson PA, Friston KJ (2007)
    Comparing hemodynamic models with DCM. NeuroImage 38: 387-401.
    Does this need to stay?? : Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging
 
    adapted from MATLAB version by Karl Friston & Klaas Enno Stephan
    """

    # NB: Biophysical constants for 1.5T
    # Time to echo
    TE  = 0.04
    # resting venous volume (%)
    V0  = 4    
    # slope r0 of intravascular relaxation rate R_iv as a function of oxygen 
    # saturation S:  R_iv = r0*[(1 - S)-(1 - S0)] (Hz)
    r0  = 25
    # frequency offset at the outer surface of magnetized vessels (Hz)
    nu0 = 40.3
    # resting oxygen extraction fraction
    E0  = 0.4
    # estimated region-specific ratios of intra- to extra-vascular signal 
    ep  = exp(ϵ)

    # -Coefficients in BOLD signal model
    k1  = 4.3*nu0*E0*TE;
    k2  = ep*r0*E0*TE;
    k3  = 1 - ep;

    # -Output equation of BOLD signal model
    v   = exp(x(:,4));
    q   = exp(x(:,5));
    g   = V0*(k1 - k1.*q + k2 - k2.*q./v + k3 - k3.*v);


    #= -derivative dgdx
    [n m]      = size(x);
    dgdx       = cell(1,m);
    [dgdx{:}]  = deal(sparse(n,n));
    dgdx{1,4}  = diag(-V0*(k3.*v - k2.*q./v));
    dgdx{1,5}  = diag(-V0*(k1.*q + k2.*q./v));
    dgdx       = spm_cat(dgdx);
    =#
end


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

