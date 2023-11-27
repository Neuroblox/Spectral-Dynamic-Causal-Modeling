# Lead field function for LFPs

function leadfield(L, sts, nr, str; name)
    @variables lfp(t)[1:nr]
    vars = vcat(lfp, sts)

    idx = [i for (i, s) in enumerate(string.(sts)) if occursin(str, s)]
    eqs = Equation[]
    for i = 1:nr
        push!(eqs, lfp[i] ~ L[i]*sts[idx[i]])
    end
    return ODESystem(eqs, t, vars, values(L); name=name)
end