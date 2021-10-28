using Turing
using Distributions
using LinearAlgebra: I, Matrix

@model AR_process(x, p) = begin
	ns, nd = size(x)
	Σ ~ InverseWishart(nd*2, Matrix(1.0I, nd, nd))    # noise covariance matrix
	beta ~ Product(Uniform.(-ones((p+1) * nd), ones((p+1) * nd)))   # linear model parameters
	beta = reshape(beta, (p+1, nd))   # get the parameters in the right matrix shape
    for t in (p+1):ns
		μ = vec(beta[1, :] + sum(beta[2:end, :] .* x[t-p:t-1, :], dims=1)')
        x[t, :] ~ MvNormal(μ, Σ)
    end
end


N = 100
timelags = 2   # = p
dim = 2

#### Produce toy autoregressive process time series ####
# A = randn(timelags, dim)
A =  [0.887951  -0.755387; -0.121936  -1.00258]
Σ = rand(dim, dim)
Σ = (Σ + Σ')/2 + I    # make Hermitian and ensure positive definiteness
ϵ = MvNormal(zeros(dim), Σ)
Y = zeros(N, dim)
Y[1:timelags, :] = rand(timelags, dim)
for i = (timelags+1):N
    Y[i, :] = sum(A.*Y[i-timelags:i-1, :], dims=1) + rand(ϵ)'
end

chain = sample(AR_process(Y, 2), NUTS(0.65), 1000 )
