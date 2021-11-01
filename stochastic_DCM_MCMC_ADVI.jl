"""

First part of this code generates a synthetic BOLD signal for N brain regions
using linear model for neuonal activity and Balloon model for haemodynamic activity.
Effective connectivity is denoted by matrix A.

Second part uses the synthetic BOLD signal to estimate the effective connectivity A.
Both MCMC sampling and ADVI methods are used.

Currently the synthetic data is generated for very low state noise 
  and both methods of estimation are breaking down.


"""

import Pkg

Pkg.add("DifferentialEquations")
Pkg.add("Plots")
Pkg.add("Turing")
Pkg.add("Distributions")
Pkg.add("MCMCChains")
Pkg.add("StatsPlots")

using DifferentialEquations
using DifferentialEquations.EnsembleAnalysis
using Turing
using Distributions
using MCMCChains
using StatsPlots
using Random
using Plots
using Turing: Variational

Random.seed!(141)

#ϕ = 0.1
N=3

#Linear neuronal activity + haemodynamic model
function neural_linear(du,u,p,t)
   

    x=u[1:N,1]
    s=u[(N+1):2*N,1]
    f=u[(2*N+1):3*N,1]
    ν=u[(3*N+1):4*N,1]
    q=u[(4*N+1):5*N,1]

    A=p[1]
    """
    stim = [0.05
            0
            0]
    """
    κ,γ,τ,α,E₀ = [0.64, 0.32, 2.00, 0.32, 0.4]

    for i in 1:N
        input =  A[i:i,:]*x
        ff = (1-(1-E₀)^(1/f[i]))/E₀

        du[i] = input[1] 
        du[N+i] = x[i] - κ*s[i] -γ*(f[i]-1)
        du[2*N+i] = s[i]
        du[3*N+i] = (f[i] - ν[i]^(1/α))/τ
        du[4*N+i] = (f[i]*ff - ν[i]^(1/α)*q[i]/ν[i])/τ

    end
end

#state noise in neuronal activity only
function state_noise(du,u,p,t)
    x=u
    A=p[1]
    ϕ = p[2]
    κ,γ,τ,α,E₀ = [0.64, 0.32, 2.00, 0.32, 0.4]
     
    du[:].=0

    for i in 1:N
    du[i] = ϕ
 
    end

end

#u0 = [rand(N,1);zeros(4*N,1)]
u0 = rand(5*N,1)
tspan = [0.0, 1000]

#Effective connectivity
A = [-0.62 0.0 0.0
      0.92 0.0 0.0
      0.38 0.47 -0.51]
     
#state noise
ϕ = 0.01    

p = [A,ϕ]     
prob1 = SDEProblem(neural_linear, state_noise, u0, tspan, p)      
sol = solve(prob1,SOSRI(),dt=0.001,saveat=0.01)

ensembleprob = EnsembleProblem(prob1)
  @time data = solve(ensembleprob,SOSRI(),saveat=0.01,trajectories=100)
  plot(EnsembleSummary(data))
#sol = solve(prob1,SRIW1(),dt=0.0001,saveat=0.01,adaptive=false)
#sol = solve(prob1,EM(),dt=0.0001,saveat=0.01,adaptive=false)
ar=Array(data)

#exctracing ν and q state variables from the ensemble solution
ν_data = ar[3*N+1:4*N,1,:,:]
q_data = ar[4*N+1:5*N,1,:,:]
t = [i for i = 0:0.01:1000]

plot(t,ν_data[3,:,10:10])
plot(sol)

#fMRI BOLD signal calculated from ν and q
function g_fmri(ν_data,q_data,ϵ)

    # NB: Biophysical constants for 1.5T scanners
    # Time to echo
    TE  = 0.04
    # resting venous volume (%)
    V₀  = 4    
    # slope r0 of intravascular relaxation rate R_iv as a function of oxygen 
    # saturation S:  R_iv = r0*[(1 - S)-(1 - S0)] (Hz)
    r₀  = 25
    # frequency offset at the outer surface of magnetized vessels (Hz)
    ν₀ = 40.3
    # resting oxygen extraction fraction
    E₀  = 0.4
    # estimated region-specific ratios of intra- to extra-vascular signal 
   # ep  = exp(ϵ)

    # -Coefficients in BOLD signal model
    k1  = 4.3*ν₀*E₀*TE;
    k2  = ϵ*r₀*E₀*TE;
    k3  = 1 - ϵ;
    g   = V₀.*(k1 .- k1.*q_data .+ k2 .- k2.*(q_data./ν_data) .+ k3 .- k3.*ν_data)
    return g
end

g_data = g_fmri(ν_data,q_data,0.1)

"""
pl=plot(t,g[3,:,1:1])

for i = 1:100
    plot!(pl,t,g[3,:,i:i])
end
plot!(t,ar[3,1,:,1:1])
"""

#defining the model for parameter estimation
Turing.setadbackend(:forwarddiff)

  @model function fitdcm(g_data,prob1)
    σ ~ InverseGamma(2,3)

    #A = zeros(N,N)
    for i = 1:N
        for j=1:N
         A[i,j] ~ Uniform(-1,1)

        end
    end
    
    ϕ ~ Uniform(0,0.01)
    
    p = [A,ϕ]
    prob = remake(prob1,p=p)
    predicted_sol = solve(prob,SOSRI(),dt=0.001,saveat=0.01)

    ar_sol=Array(predicted_sol)
    ν_pred = ar_sol[3*N+1:4*N,1,:]
    q_pred = ar_sol[4*N+1:5*N,1,:]

    g_pred = g_fmri(ν_pred,q_pred,0.1)
    
    if predicted.retcode != :Success
      Turing.acclogp!(_varinfo, -Inf)
    end
    for j in 1:size(g_data,3)
    for i = 1:length(predicted_sol)
      g_data[:,i,j] ~ MvNormal(g_pred[:,i],σ)
    end
    end
  end

  model = fitdcm(g_data,prob1)
  # model = fitvp(sol,prob1)
 init_A = -1 .+ 2*rand(3,3)
 init_ϕ = 0.01
 init_p = [0.1,init_A,init_ϕ]
   
  chain = sample(model, NUTS(0.25), MCMCThreads(),1000, 3,init_theta = init_p)

  advi = ADVI(10, 1000)
q = vi(model, advi);

