""" This code computes the moments of the posterior p.d.f. of the parameters of a
nonlinear model specified by IS(P,M,U) under Gaussian assumptions. [MATLAB code comment]
"""

using Plots
using LinearAlgebra: det, inv, tr
using ExponentialUtilities: expv
using ForwardDiff: jacobian
#=
Cite when using ForwardDiff:
@article{RevelsLubinPapamarkou2016,
    title = {Forward-Mode Automatic Differentiation in {J}ulia},
   author = {{Revels}, J. and {Lubin}, M. and {Papamarkou}, T.},
  journal = {arXiv:1607.07892 [cs.MS]},
     year = {2016},
      url = {https://arxiv.org/abs/1607.07892}
}
=#
using DifferentialEquations: ODEProblem, solve
#=
Cite when using DifferentialEquations:
@article{rackauckas2017differentialequations,
  title={Differentialequations.jl--a performant and feature-rich ecosystem for solving differential equations in julia},
  author={Rackauckas, Christopher and Nie, Qing},
  journal={Journal of Open Research Software},
  volume={5},
  number={1},
  year={2017},
  publisher={Ubiquity Press}
}
=#


#= Define notational equivalences between MATLAB code and Julia code:

# the following two precision matrices will not be updated by the code,
# they belong to the assumed prior distribution p (fixed, but what if it isn't
# the ground truth?)
ipC = Πθ_p   # precision matrix of prior of parameters p(θ)
ihC = Πλ_p   # precision matrix of prior of hyperparameters p(λ)

Variational distribution parameters:
pE, Ep = θμ, μθ   # prior expectation of parameters (q(θ))
pC, Cp = θΣ, Σθ   # prior covariance of parameters (q(θ))
hE, Eh = λμ, μλ   # prior expectation of hyperparameters (q(λ))
hC, Ch = λΣ, Σλ   # prior covariance of hyperparameters (q(λ))

Σ, iΣ  # data covariance matrix (likelihood), and its inverse (precision of likelihood)
Q      # components of iΣ; definition: iΣ = sum(exp(λ)*Q)
=#

# compute Jacobian of rhs w.r.t. variable -> matrix exponential solution (use ExponentialUtilities.jl)
# -> use this numerical integration as solution to the diffeq to then differentiate solution w.r.t. parameters (like sensitivity analysis in Ma et al. 2021)
# -> that Jacobian is used in all the computations of the variational Bayes


# Define priors etc.
# Q, θμ, θΣ, λμ, λΣ

# pE.A = A/128; θμ?


# toy model dynamical system - Lorenz system
function f!(dx, x::Vector, θ::Vector, t)
    @show typeof(x) typeof(θ) typeof(t)
    dx[1] = θ[1] * (x[2] - x[1])
    dx[2] = x[1] * (θ[2] - x[3]) - x[2]
    dx[3] = x[1] * x[2] - θ[3] * x[3]
end

# define function that linearizes and integrates dynamical system
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

# Compute prediction error
e = y - f

np = length(μθ)            # number of parameters
nx = length(x0)             # number of variables
nh = length(μλ)            # number of hyperparameters

## M-step: Fisher scoring scheme to find h = max{F(p,h)} // comment from MATLAB code
iΣ = zeros(np, np)
for i = 1:nh
    iΣ = iΣ + Q[i]*exp(h[i])
end

J = J_θ(μθ)
Pp = similar(J)
mul!(Pp, transpose(J), iΣ * J)    # in MATLAB code 'real()' is applied to the resulting matrix product, why?
Σθ = inv(Pp + Πθ_p)
Σ = inv(iΣ)

P = similar(Q)
PΣ = similar(Q)
JPJ = similar(Q)
for i = 1:nh
    P[i] = Q[i]*exp(h[i])
    PS[i] = P[i] * S
    JPJ[i] = transpose(J)*P[i]*J      # in MATLAB code 'real()' is applied (see also some lines above), why?
end

dFdh = zeros(nh)
dFdhh = zeros(nh, nh)
for i = 1:nh
    dFdh[i] = (tr(PS[i])*nq - transpose(e)*P[i]*e - tr(Cp * JPJ[i]))/2
    for j = i:nh
        dFdhh[i, j] = - tr(PS[i] * PS[j])*nq/2
        dFdhh[j, i] = dFdhh[i, j]
    end
end

d = λ - λE
dFdh = dFdh - iΣ_λ*d;
dFdhh = dFdhh - iΣ_λ
Σ_λ = inv(-dFdhh)    # why is there a minus here??

dλ = -inv(dFdhdh) * dFdh    # Gauss-Newton step, transform into Levenberg-Marquardt, see MATLAB code
dλ = min(max(dλ,-1),1)      # probably precaution for numerical errors?
λ = λ + dλ

## E-Step with Levenberg-Marquardt regularization // comment from MATLAB code
L = zeros(3)
L[1] = (log(det(iΣ))*nq  - transpose(e) * iΣ * e - nx*log(2pi))/2
L[2] = (log(det(iΣ_θ * Σ_θ)) - transpose(p) * iΣ_θ *p)/2
L[3] = (log(det(ihC*Ch)) - transpose(d) * iΣ_λ * d)/2;
F = sum(L);

# MISSING: check if energy increases or not 

# Conditional update of gradients and curvature
dFdp  = -transpose(J)*iΣ*e - iΣ_θ * p   # check sign!!
dFdpp = -transpose(J)*iΣ*J - iΣ_θ

dp = -inv(dFdpp) * dFdp    # also here: transform to Levenberg-Marquardt
p = p + dp
Ep = pE + p



# in original code: check for stability (of numerical differential)

# E-step update only before computing prediction error after one iteration completed and some instability is encountered (when differentiating)

# if F has increased, update gradients and curvatures for E-Step
#----------------------------------------------------------------------
if F > C.F || k < 3
    
    # accept current estimates
    #------------------------------------------------------------------
    C.p   = p;
    C.h   = h;
    C.F   = F;
    C.L   = L;
    C.Cp  = Cp;
    
    # E-Step: Conditional update of gradients and curvature
    #------------------------------------------------------------------
    dFdp  = -real(J'*iS*e) - ipC*p;
    dFdpp = -real(J'*iS*J) - ipC;
    
    # decrease regularization
    #------------------------------------------------------------------
    v     = min(v + 1/2,4);
    str   = "EM:(+)";
    
else
    
    # reset expansion point
    #------------------------------------------------------------------
    p     = C.p;
    h     = C.h;
    Cp    = C.Cp;
    
    # and increase regularization
    #------------------------------------------------------------------
    v     = min(v - 2,-4);
    str   = "EM:(-)";
    
end





function Q = spm_dcm_csd_Q(csd)
#= 

Note that this routine is by far the slowest part of the processing pipeline.
A nested for loop with 30752^2 iterations is the cause. This stems from a CSD 
computation of a matrix with 32x31x31 elements. Why such a large matrix if there
are 31 areas? Figure out. However, some coarse estimation gives that it shouldn't take
more than ~6h... hmmm...


    % Precision of cross spectral density
    % FORMAT Q = spm_dcm_csd_Q(csd)
    % 
    % csd{i}   - [cell] Array of complex cross spectra
    % Q        - normalised precision
    %--------------------------------------------------------------------------
    % This routine returns the precision of complex cross spectra based upon
    % the asymptotic results described in Camba-Mendez & Kapetanios (2005):
    % In particular, the scaled difference between the sample spectral density
    % (g) and the predicted density (G);
    %  
    % e = vec(g - G)
    % 
    % is asymptotically complex normal, where the covariance between e(i,j) and
    % e(u,v) is given by Q/h and:
    % 
    % Q = G(i,u)*G(j,u):  h = 2*m + 1
    %  
    % Here m represent the number of averages from a very long time series.The
    % inverse of the covariance is thus a scaled precision, where the
    % hyperparameter (h) plays the role of the degrees of freedom (e.g., the
    % number of averages comprising the estimate). In this routine, we use the
    % sample spectral density to create a frequency specific precision matrix
    % for the vectorised spectral densities - under the assumption that the
    % former of this sample spectral density resembles the predicted spectral
    % density (which will become increasingly plausible with convergence).
    %
    % Camba-Mendez, G., & Kapetanios, G. (2005). Estimating the Rank of the
    % Spectral Density Matrix. Journal of Time Series Analysis, 26(1), 37-48.
    % doi: 10.1111/j.1467-9892.2005.00389.x
    %__________________________________________________________________________
    % Copyright (C) 2018 Wellcome Trust Centre for Neuroimaging
=#
