using MAT
using ToeplitzMatrices
using Serialization
using FFTW

# TODO: investiagate why SymmetricToeplitz(ccf[1:p, i, j]) is not good to be used but instead need to use Toeplitz(ccf[1:p, i, j], ccf[1:p, j, i])
# as is done in the original MATLAB code (confront comment there). ccf should be a symmetric matrix so there should be no difference between the
# Toeplitz matrices but for the second jacobian (J[2], see for loop for i = 1:nJ in function diff) the computation produces subtle differences between
# the two versions. It would be preferreable to use SymmetricToeplitz over Toeplitz because the latter produces an error when executing iteration 15
# of that same loop above. In what follows we reproduce that error (note that said for loop is in the main code, but not present here. We reproduce the
# scenario when i = 15)

w = vec(matread("MWE_Toeplitz/spectralDCM_demodata_notsparse.mat")["M_nosparse"]["Hz"])
csd = deserialize("MWE_Toeplitz/G_Toeplitztest.dat")
csd_mat = matread("MWE_Toeplitz/G_MATLAB.mat")["y"]

dt  = 1/(2*w[end])
p = 7
dw = w[2] - w[1]
w = w/dw
ns = dt^-1
N = ceil(Int64, ns/2/dw)
gj = findall(x -> x > 0 && x < (N + 1), w)
gi = gj .+ (ceil(Int64, w[1]) - 1)    # TODO: understand what's the logic/purpose of this line!
g = zeros(ComplexF64, N)

# transform cross spectral density to cross-correlation function
ccf = zeros(ComplexF64, N*2+1, size(csd,2), size(csd,3))
for i = 1:size(csd, 2)
    for j = 1:size(csd, 3)
        g[gi] = csd[gj,i,j]
        f = ifft(g)
        f = ifft(vcat([0.0im; g; conj(g[end:-1:1])]))
        ccf[:,i,j] = real.(fftshift(f))*N*dw
    end
end

# MAR coefficients from ccf
N = size(ccf,1)
m = size(ccf,2)
n = (N - 1) ÷ 2
p = min(p, n - 1)
ccf = ccf[(1:n) .+ n,:,:]
A = zeros(m*p, m)
B = zeros(m*p, m*p)

# The following will produce an error "First element of the vectors must be the same" since: ccf[1,1,2] != ccl[1,2,1]! but they are equal until the last digit
# The weird thing: using csd produced by MATLAB avoids this error. So the problem doesn't lie with the fft here but happens already before.
# Note also: csd ≈ csd_mat: true but csd == csd_mat: false.

for i = 1:m
    for j = 1:m
        print(i,j,"\n")
        A[((i-1)*p+1):i*p, j] = ccf[(1:p) .+ 1, i, j]
        B[((i-1)*p+1):i*p, ((j-1)*p+1):j*p] = Toeplitz(ccf[1:p, i, j], ccf[1:p, j, i])    # SymmetricToeplitz(ccf[1:p, i, j])
    end
end

csd ≈ csd_mat
csd == csd_mat
