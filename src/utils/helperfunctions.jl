
"""
    vecparam(param::OrderedDict)

    Function to flatten an ordered dictionary of model parameters and return a simple list of parameter values.

    Arguments:
    - `param`: dictionary of model parameters (may contain numbers and lists of numbers)
"""
function vecparam(param::OrderedDict)
    flatparam = Float64[]
    for v in values(param)
        if (typeof(v) <: Array)
            for vv in v
                push!(flatparam, vv)
            end
        else
            push!(flatparam, v)
        end
    end
    return flatparam
end

"""
    function matlab_norm(A, p)

    Simple helper function to implement the norm of a matrix that is equivalent to the one given in MATLAB for order=1, 2, Inf. 
    This is needed for the reproduction of the exact same results of SPM12.

    Arguments:
    - `A`: matrix
    - `p`: order of norm
"""
function matlab_norm(M, p)
    if p == 1
        return maximum(vec(sum(abs.(M),dims=1)))
    elseif p == Inf
        return maximum(vec(sum(abs.(M),dims=2)))
    elseif p == 2
        print("Not implemented yet!\n")
        return NaN
    end
end

"""
    function spm_logdet(M)

    SPM12 style implementation of the logarithm of the determinant of a matrix.

    Arguments:
    - `M`: matrix
"""
function spm_logdet(M)
    TOL = 1e-16
    s = diag(M)
    if sum(abs.(s)) != sum(abs.(M[:]))
        ~, s, ~ = svd(M)
    end
    return sum(log.(s[(s .> TOL) .& (s .< TOL^-1)]))
end

"""
    function csd_Q(csd)

    Compute correlation matrix to be used as functional connectivity prior.
"""
function csd_Q(csd)
    s = size(csd)
    Qn = length(csd)
    Q = zeros(ComplexF64, Qn, Qn);
    idx = CartesianIndices(csd)
    for Qi  = 1:Qn
        for Qj = 1:Qn
            if idx[Qi][1] == idx[Qj][1]
                Q[Qi,Qj] = csd[idx[Qi][1], idx[Qi][2], idx[Qj][2]]*csd[idx[Qi][1], idx[Qi][3], idx[Qj][3]]
            end
        end
    end
    Q = inv(Q .+ matlab_norm(Q, 1)/32*Matrix(I, size(Q)))   # TODO: MATLAB's and Julia's norm function are different! Reconciliate?
    return Q
end