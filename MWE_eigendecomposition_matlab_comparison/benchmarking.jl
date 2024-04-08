using DifferentiableEigen
include("../src/utils/helperfunctions_AD.jl")

@benchmark bar1 = LinearAlgebra.eigen(J_tot)
function test(J)
    F = DifferentiableEigen.eigen(J_tot)
    Λ = DifferentiableEigen.arr2Comp(F[1], size(J_tot, 1))
    V = DifferentiableEigen.arr2Comp(F[2], size(J_tot))
    return Eigen(Λ, V)
end
@benchmark bar2 = test(J_tot)
