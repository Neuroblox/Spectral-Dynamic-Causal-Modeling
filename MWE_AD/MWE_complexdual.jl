using LinearAlgebra: I, Matrix
using ForwardDiff: Dual, Partials
using FFTW
using ProfileView

function dft(x::AbstractArray)
    """discrete fourier transform"""
    N = size(x)[1]
    out = Array{ComplexF64}(undef,N)
    for k in 0:N-1
        out[k+1] = sum([x[n+1]*exp(-2*im*π*k*n/N) for n in 0:N-1])
    end
    return out
end

function idft(x::AbstractArray)
    """discrete inverse fourier transform"""
    N = size(x)[1]
    out = Array{eltype(x)}(undef,N)
    for n in 0:N-1
        out[n+1] = 1/N*sum([x[k+1]*exp(2*im*π*k*n/N) for k in 0:N-1])
    end
    return out
end

function idft(x::AbstractArray)
    """discrete inverse fourier transform"""
    N = size(x)[1]
    out = Array{Complex{eltype(x)}}(undef,N)
    for n in 0:N-1
        out[n+1] = 1/N*sum([x[k+1]*exp(2*im*π*k*n/N) for k in 0:N-1])
    end
    return out
end

function FFTW.ifft(x::Array{Complex{Dual{T, P, N}}}) where {T, P, N}
    return ifft(real(x)) + 1im * ifft(imag(x))
end

function FFTW.ifft(x::Array{Dual{T, P, N}}) where {T, P, N}
    v = (tmp->tmp.value).(x)
    iftx = ifft(v)
    iftrp = Array{Partials}(undef, length(x))
    iftip = Array{Partials}(undef, length(x))
    local iftrp_agg, iftip_agg
    for i = 1:N
        dx = (tmp->tmp.partials[i]).(x)
        iftdx = ifft(dx)
        if i == 1
            iftrp_agg = real(iftdx) .* dx
            iftip_agg = imag(iftdx) .* dx
        else
            iftrp_agg = cat(iftrp_agg, real(iftdx) .* dx, dims=2)
            iftip_agg = cat(iftip_agg, imag(iftdx) .* dx, dims=2)
        end
    end
    for i = 1:length(x)
        iftrp[i] = Partials(Tuple(iftrp_agg[i, :]))
        iftip[i] = Partials(Tuple(iftip_agg[i, :]))
    end
    return Complex.(Dual{T, P, N}.(real(iftx), iftrp), Dual{T, P, N}.(imag(iftx), iftip))
end
foo = Ref{Any}()
x = deserialize("deleteme.dat")
x = Dual.(rand(10), Partials.(Tuple.(rand(10))))
@time idft(x);
@time ifft(x);
FFTW.ifft(foo[]) ≈ idft(foo[])

@profilehtml ifft(x)


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
    p = 2   # number of time lags of MAR model
    f = 2.0.^(range(0,stop=5)) # frequencies at which to evaluate CSD
    dt = 1/(2*f[end])          # time step, inverse of sampling frequency
    a = [randn(nd, nd) for i = 1:p]   # MAR model parameters
    mar = Dict([("A", a), ("Σ", Σ), ("p", p)])
    return mar2csd(mar, f, dt^-1)
end

p = Dual(2.0, (1.0, 0.0)) + Dual(0.0, (1.0, 0.0))im;
q = Dual(3.0, (0.0, 1.0)) + Dual(0.0, (0.0, 1.0))im;
# p, q = 2.3,3.4
A = [p 0.5;
     0.5 q];

csd(A)


# Helmut's toy model to test AD of FFT

using Random
using StatsPlots
using Turing
using LinearAlgebra:I, Matrix
using FFTW

# this is a test program to see whether ForwardDiff can handle FFT in Turing
# we are creating random data and then calculate the power spectrum using the p-Welch method
x = (rand(16) .- 0.5) .* 2.0
xfft = dft(x)
# fp = real(xfft .* conj(xfft))[2:63]

@model function fit_random(data)
    A ~ Normal(2.0,4.0)
    s = (rand(16) .- 0.5) .* A
    fts = dft(s)
    data ~ MvNormal(fts, Matrix(1.0I, length(fts), length(fts)))
    # rdata = real(data)
    # idata = imag(data)
    # rdata ~ MvNormal(real(sdft), Matrix(1.0I, length(sdft), length(sdft)))
    # idata ~ MvNormal(imag(sdft), Matrix(1.0I, length(sdft), length(sdft)))
end

model = fit_random(xfft)
chain = sample(model, NUTS(), 2000)


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
