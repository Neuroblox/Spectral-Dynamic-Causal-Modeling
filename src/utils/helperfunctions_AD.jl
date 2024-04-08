using LinearAlgebra: Eigen
using ChainRules: _eigen_norm_phase_fwd!


"""
    function LinearAlgebra.eigen(M::Matrix{Dual{T, P, np}}) where {T, P, np}
    
    Dispatch of LinearAlgebra.eigen for dual matrices with complex numbers. Make the eigenvalue decomposition 
    amenable to automatic differentiation. To do so compute the analytical derivative of eigenvalues
    and eigenvectors. 

    Arguments:
    - `M`: matrix of type Dual of which to compute the eigenvalue decomposition. 

    Returns:
    - `Eigen(evals, evecs)`: eigenvalue decomposition returned as type LinearAlgebra.Eigen

"""
function LinearAlgebra.eigen(M::Matrix{Dual{T, P, np}}) where {T, P, np}
    nd = size(M, 1)
    A = (p->p.value).(M)
    F = eigen(A, sortby=nothing, permute=true)
    λ, V = F.values, F.vectors
    local ∂λ_agg, ∂V_agg
    # compute eigenvalue and eigenvector derivatives for all partials
    for i = 1:np
        dA = (p->p.partials[i]).(M)
        tmp = V \ dA
        ∂K = tmp * V   # V^-1 * dA * V
        ∂Kdiag = @view ∂K[diagind(∂K)]
        ∂λ_tmp = eltype(λ) <: Real ? real.(∂Kdiag) : copy(∂Kdiag)   # why do only copy when complex??
        ∂K ./= transpose(λ) .- λ
        fill!(∂Kdiag, 0)
        ∂V_tmp = mul!(tmp, V, ∂K)
        _eigen_norm_phase_fwd!(∂V_tmp, A, V)
        if i == 1
            ∂V_agg = ∂V_tmp
            ∂λ_agg = ∂λ_tmp
        else
            ∂V_agg = cat(∂V_agg, ∂V_tmp, dims=3)
            ∂λ_agg = cat(∂λ_agg, ∂λ_tmp, dims=2)
        end
    end
    ∂V = Array{Partials}(undef, nd, nd)
    ∂λ = Array{Partials}(undef, nd)
    # reassemble the aggregated vectors and values into a Partials type
    for i = 1:nd
        ∂λ[i] = Partials(Tuple(∂λ_agg[i, :]))
        for j = 1:nd
            ∂V[i, j] = Partials(Tuple(∂V_agg[i, j, :]))
        end
    end
    if eltype(V) <: Complex
        evals = map((x,y)->Complex(Dual{T, Float64, length(y)}(real(x), Partials(Tuple(real(y)))), 
                                   Dual{T, Float64, length(y)}(imag(x), Partials(Tuple(imag(y))))), F.values, ∂λ)
        evecs = map((x,y)->Complex(Dual{T, Float64, length(y)}(real(x), Partials(Tuple(real(y)))), 
                                   Dual{T, Float64, length(y)}(imag(x), Partials(Tuple(imag(y))))), F.vectors, ∂V)
    else
        evals = Dual{T, Float64, length(∂λ[1])}.(F.values, ∂λ)
        evecs = Dual{T, Float64, length(∂V[1])}.(F.vectors, ∂V)
    end
    return Eigen(evals, evecs)
end
