using LinearAlgebra: I, Matrix
using ForwardDiff: Dual
using FFTW

function dft(x::AbstractArray)
    """discrete fourier transform"""
    N = size(x)[1]
    out = Array{Any}(undef,N)
    for k in 0:N-1
        out[k+1] = sum([x[n+1]*exp(-2*im*π*k*n/N) for n in 0:N-1])
    end
    return out
end

function idft(x::AbstractArray)
    """discrete inverse fourier transform"""
    N = size(x)[1]
    out = Array{Any}(undef,N)
    for n in 0:N-1
        out[n+1] = 1/N*sum([x[k+1]*exp(2*im*π*k*n/N) for k in 0:N-1])
    end
    return out
end


# My own toy model for ADing a CSD computation based on MARs

function mar2csd(mar, freqs, sf)
    Σ = mar["Σ"]
    p = mar["p"]
    A = mar["A"]
    nd = size(Σ, 1)
    w  = 2*pi*freqs/sf
    nf = length(w)
	csd = zeros(eltype(Σ), nf, nd, nd)
	for i = 1:nf
		af_tmp = I
		for k = 1:p
			af_tmp = af_tmp + A[k] * exp(-im * k * w[i])
		end
		iaf_tmp = inv(af_tmp)
		csd[i,:,:] = iaf_tmp * Σ * iaf_tmp'
	end
    csd = 2*csd/sf
    return csd
end

function csd(Σ)
    nd = 2  # number of dimensions
    p = 2  # number of time lags of MAR model
    f = 2.0.^(range(0,stop=5)) # frequencies at which to evaluate CSD
    dt = 1/(2*f[end]) # time step, inverse of sampling frequency
    a = [randn(nd, nd) for i = 1:p]   # MAR model parameters
    mar = Dict([("A", a), ("Σ", Σ), ("p", p)])
    return mar2csd(mar, f, dt^-1)
end

p = Dual(2.0, (1.0, 0.0));
q = Dual(3.0, (0.0, 1.0));
# p, q = 2.3,3.4
A = [p 0.5;
     0.5 q];

csd(A)


# Helmut's toy model to test AD of FFT

using Random
using StatsPlots
using Turing
using LinearAlgebra:I
using FFTW

# this is a test program to see whether ForwardDiff can handle FFT in Turing
# we are creating random data and then calculate the power spectrum using the p-Welch method
x = (rand(128) .- 0.5) .* 2.0
xfft = fft(x)
fp = real(xfft .* conj(xfft))[2:63]
plot(fp)

@model function fit_random(data)
    A ~ Normal(2.0,4.0)
    ϵ ~ Uniform(10,1000)
    s = (rand(128) .- 0.5) .* A
    sfft = fft(s)
    ps = real(sfft .* conj(sfft))[2:63]
    data ~ MvNormal(ps,I*ϵ)
end

rmodel = fit_random(fp)
chain = sample(rmodel, NUTS(),2000)


# DualNumbers has a complex values based Dual. From: https://discourse.julialang.org/t/automatic-differentiation-of-complex-functions-simple-workaround/75775/4
using DualNumbers

f(z) = 1 / (z - (3+4im))
fhand(z) = -1 / (z - (3+4im))^2
z₀ = 2+2im
f(z₀)
f(Dual(z₀,1)).epsilon   #(instead of partials)


# simplest possible example, taken from ForwardDiff test environment https://github.com/JuliaDiff/ForwardDiff.jl/blob/master/test/DerivativeTest.jl
# can't evaluate derivative at complex value, only real works
using ForwardDiff
f(x) = x -> (1+im)*x^2
ForwardDiff.derivative(f(x), 1)
