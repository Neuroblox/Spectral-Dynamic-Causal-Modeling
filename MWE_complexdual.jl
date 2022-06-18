using LinearAlgebra: I, Matrix
using ForwardDiff: Dual

function mar2csd(mar, freqs, sf)
    Σ = mar["Σ"]
    p = mar["p"]
    A = mar["A"]
    nd = size(Σ, 1)
    w  = 2*pi*freqs/sf    # isn't it already transformed?? Is the original really in Hz? Also clearly freqs[end] is not the sampling frequency of the signal...
    nf = length(w)
	csd = zeros(eltype(Σ), nf, nd, nd)
	for i = 1:nf
		af_tmp = I
		for k = 1:p
			af_tmp = af_tmp + A[k] * exp(-im * k * w[i])
		end
		iaf_tmp = inv(af_tmp)
		csd[i,:,:] = iaf_tmp * Σ * iaf_tmp'     # is this really the covariance or rather precision?!
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
A = [p 0.5;
     0.5 q];


csd(A)
