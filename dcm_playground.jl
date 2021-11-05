using ModelingToolkit, OrdinaryDiffEq, Plots

# https://juliapackages.com/p/modelingtoolkit
# https://mtk.sciml.ai/stable/systems/ODESystem/

function hemodynamics(x, decay, transit)
      """
      Components of x are:
      x[:,1] - neural activity: x
      x[:,2] - vascular signal: s
      x[:,3] - rCBF: ln(f)
      x[:,4] - venous volume: ln(ν)
      x[:,5] - deoxyhemoglobin (dHb): ln(q)
      decay, transit - free parameters, set to 0 for standard parameters.
      
      This function implements the hymodynamics model (balloon model and neurovascular state eq.) described in: 
      
          Stephan KE, Weiskopf N, Drysdale PM, Robinson PA, Friston KJ (2007)
          Comparing hemodynamic models with DCM. NeuroImage 38: 387-401.
      
          adapted from spm_fx_fmri.m in SPM12 by Karl Friston & Klaas Enno Stephan
      """
      
      #= hemodynamic parameters
          H(1) - signal decay                                   d(ds/dt)/ds)
          H(2) - autoregulation                                 d(ds/dt)/df)
          H(3) - transit time                                   (t0)
          H(4) - exponent for Fout(v)                           (alpha)
          H(5) - resting oxygen extraction                      (E0)
      =#
      H = [0.64, 0.32, 2.00, 0.32, 0.4];
      
      # exponentiation of hemodynamic state variables
      x[:, 3:5] = exp(x[:, 3:5])
      
      # signal decay
      κ = H[1]*exp(decay)
      
      # transit time
      τ = H[3]*exp(transit)
      
      # Fout = f(v) - outflow
      fv = x[:, 3].^(1/H[4])
      
      # e = f(f) - oxygen extraction
      ff = (1 - (1 - H[5]).^(1.0./x[:, 3]))/H[5]
      
      
      # implement differential state equation f = dx/dt (hemodynamic)
      f[:, 2] = x[:, 1] - κ.*x[:, 2] - H[2]*(x[:, 3] - 1)   # Corresponds to eq (9)
      f[:, 3] = x[:, 2]./x[:, 3]  # Corresponds to eq (10), note the added logarithm (see doc string)
      f[:, 4] = (x[:, 3] - fv)./(τ.*x[:, 4])    # Corresponds to eq (8), note the added logarithm (see doc string)
      f[:, 5] = (ff.*x[:, 3] - fv.*x[:, 5]./x[:, 4])./(τ.*x[:, 5])  # Corresponds to eq (8), note the added logarithm (see doc string)
      
      return f
end

# Simulate the BOLD signal
function g_fmri(x, ϵ)
      """
      Simulated BOLD response to input
      FORMAT [g,dgdx] = g_fmri(x, ϵ)
      g          - BOLD response (%)
      x          - hemodynamic state vector, same as above.
      ϵ          - free parameter (note also here as above, actually ln(ϵ)), ratio of intra- to extra-vascular components
  
      This function implements the BOLD signal model described in: 
  
      Stephan KE, Weiskopf N, Drysdale PM, Robinson PA, Friston KJ (2007)
      Comparing hemodynamic models with DCM. NeuroImage 38: 387-401.
   
      adapted from spm_gx_fmri.m in SPM12 by Karl Friston & Klaas Enno Stephan
  
      NB: Biophysical constants for 1.5T scanners:
      TE  = 0.04
      V0  = 4    
      r0  = 25
      nu0 = 40.3
      E0  = 0.4
      """
  
      # NB: Biophysical constants for 1.5T scanners
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
      ν   = exp(x[:,4])
      q   = exp(x[:,5])
      g   = V0*(k1 - k1*q + k2 - k2*q./ν + k3 - k3*ν)
  
  
      #= -derivative dgdx   # still need to figure out - these derivatives are strange!
      n = size(x)[1];
      dgdx = cell(1,m);
      [dgdx{:}]  = deal(sparse(n,n));
      dgdx{1,4}  = diag(-V0*(k3.*v - k2.*q./v));
      dgdx{1,5}  = diag(-V0*(k1.*q + k2.*q./v));
      dgdx       = spm_cat(dgdx);
      =#
  
      return g
  end

using MAT
vars = matread("spectralDCM_demodata.mat")
Y_mat = vars["Y"]
y_csd = vars["csd"]

# priors of spectral parameters
# log(α) and log(β)

# define function that implements spectra given in equation (2) of "A DCM for resting state fMRI".

G = w.^(-exp(β[1]))   # spectrum of hidden dynamics
# spectrum of visible units
# compute Volterra kernels
# 1. compute jacobian w.r.t. f, 2. compute also delay operator, 3. compute jacobian w.r.t. g
J_x = jacobian(f, x0)
bold = x -> g_fmri(hemodynamics(x, 0, 0), 0)
J_g = jacobian(bold, x0)   # compare with analytical solution

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

