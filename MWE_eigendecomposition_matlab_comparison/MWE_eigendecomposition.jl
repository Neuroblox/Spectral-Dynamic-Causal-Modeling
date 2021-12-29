using MAT
using LinearAlgebra
# using MKL
showless = x -> @show round.(x, digits=4)

# in what follows compare the eigendecomposition of MATLAB and Julia.
# it appears that the only situation where both agree and produce identical results is when MATLAB's eig is called with no parameters
# and Julia's eigen is called with permute=true and sortby=nothing.
# However, the MATLAB code of SPM12 calls eig with the paramter 'nobalance'. In that case the identical eigenvalue's can be produced by
# calling eigen with permute=false and sortby=nothing.
# Note also that matters get even more complicated when MKL is activated (check LinearAlgebra.BLAS.lbt_get_config() to be sure not to use openblas)
# in that case correspondence is found only when MATLAB's eig is called with no parameter and never when called with 'nobalance'.
M = matread("MWE_eigendecomposition_matlab_comparison/eig-test.mat")["J"]
tmp = matread("MWE_eigendecomposition_matlab_comparison/eigenvals_eigenvecs.mat")
v_matb, s_matb = tmp["v"], tmp["s"]
tmp = matread("MWE_eigendecomposition_matlab_comparison/eigenvals_eigenvecs_nobalance.mat")
v_mat, s_mat = tmp["v"], tmp["s"]
s, v = eigen(M, permute=true, scale=false, sortby=nothing)
s ≈ s_mat
v ≈ v_mat
showless([s_mat s])
showless(v)
showless(v_mat)