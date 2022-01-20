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
using DelimitedFiles

Random.seed!(08)

 function vanderpol(du,u,p,t)
	x1,x2 = u
	θ,ϕ = p
	
	du[1] = x2
	du[2] = θ*(1-x1^2)*x2 - x1 
	
end


function add_noise(du,u,p,t)
    x1,x2 = u
    θ,ϕ = p
    
    du[1] = ϕ#   0	
    du[2] = ϕ
  end

  u0 = [0.1, 0.1]
  tspan = [0.0, 50]
  p = [1, (0.1)]

  slope = 5;
  
  scale = 50;
  
  prob1 = SDEProblem(vanderpol, add_noise, u0, tspan, p)

  sol = solve(prob1,SOSRI(),saveat=0.1)

  plot(sol)

  #plot(sol, vars = (1,2))
  
  ensembleprob = EnsembleProblem(prob1)
  @time data = solve(ensembleprob,SOSRI(),EnsembleThreads(),saveat=0.1,trajectories=1)
  #plot(EnsembleSummary(data))
  #ar=Array(sol)
  ar=Array(data)
  function g_sigmoid(x,slope,scale)
  sig = scale * 1/(1+exp(-slope*(x)))
  return sig;
  end
  


  ysim = g_sigmoid.(ar,slope,scale) #this is the data to be fed to the model
  
  ns = rand(Normal(0,1),2,size(ar,2),size(ar,3)) #this is the measurement noise

  t = [i for i = 0:0.1:50]

  #x = readdlm("van_data_x.txt", '\t', Float64,'\n')
  #y = readdlm("van_data_y.txt", '\t', Float64,'\n')


#length(data)

  Turing.setadbackend(:forwarddiff)

  
  @model function fitvp(data,prob1)
    #σ ~ InverseGamma(8,3)
    σ = 1
    #θ ~ Uniform(0,10)
    #θ ~ Normal(0,2)
    θ ~ Gamma(2,5)
    #θ ~ TruncatedNormal(0,1,0,Inf)
    ϕ ~ Uniform(0,0.2)
    #ϕ ~ Gamma(1.1,0.1)
    x01 ~ Normal(0.1,0.1)
    
    x02 ~ Normal(0.1,0.1)

    p = [θ,ϕ]
    u0 = [x01,x02]
    #u0 = typeof(ϕ).(prob1.u0)
    #u0 = u0

  if θ >=0

    prob = remake(prob1,u0=u0,p=p)
    predicted = solve(prob,SOSRI(),saveat=0.1)#,maxiters=1e7)
    
    if predicted.retcode != :Success
        
        Turing.acclogp!(__varinfo__, -Inf)
    end

    x_pred = Array(predicted)
    y_pred = g_sigmoid.(x_pred,slope,scale)

    
    for j = 1:size(data,3)
    for i = 1:min(length(predicted),size(data,2))#size(data,2)  
    #for i = 1:length(data[:,1])
     # data[i,1] ~ Normal(y_pred[1,i],σ)
     # data[i,2] ~ Normal(y_pred[2,i],σ)

     data[1,i,j] ~ Normal(y_pred[1,i],σ)
     data[2,i,j] ~ Normal(y_pred[2,i],σ)
    end
    end
    
  else
    Turing.acclogp!(__varinfo__, -Inf)
  end

  end

  #model = fitvp(y,prob1)
  
  model = fitvp(ysim+ns,prob1)
  
  # model = fitvp(sol,prob1)

  #chain = sample(model, NUTS(0.25), MCMCThreads(),1000, 1)#,init_theta = [0.1, 0.5, 0.1])
   chain = sample(model, NUTS(0.65),2000)

  plot(chain)
  cr=Array(chain)

  pl2= plot(sol,alpha=2,legend=false)

  
  for k in 1:300
      resol = solve(remake(prob1,p=cr[rand(1:1000),1:2]),SOSRI(),saveat=0.1)
      plot!(pl2,resol, alpha=0.1, color = "#BBBBBB", legend = false)
  end

  pl2
  
  Random.seed!(91)

  advi = ADVI(30, 1000)
  
  q = vi(model, advi);

  samples=rand(q,10000)
  mn=mean(samples,dims=2)
  density(samples[1,:],legend=false,xlims=(0,5))

  pl= plot(sol,alpha=2,legend=false)
  for k in 1:300
    #resol = solve(remake(prob1,u0=samples[3:4,rand(1:10000)],p=samples[1:2,rand(1:10000)]),SOSRI(),saveat=0.1)
    resol = solve(remake(prob1,p=samples[1:2,rand(1:10000)]),SOSRI(),saveat=0.1)
    
    plot!(pl,resol, alpha=0.1, color = "#BBBBBB", legend = false,ylims=(-5,5))
  end
pl
open("samples4.txt","w") do io
	writedlm(io, samples)
end

samples = readdlm("samples4.txt", '\t', Float64,'\n')
