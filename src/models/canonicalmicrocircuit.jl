# Canonical micro-circuit model
# two ways to design it: either add function to edges or to blox.

# some thoughts on design:
# - include measurement models into blox. Or at least define which variables will be measured (which would be the input to the measurement model). 
#   Differ that from connector, since that is between things.
# - 

using ModelingToolkit
using DifferentialEquations
using MetaGraphs
using Graphs
using Random
using Plots

abstract type AbstractBlox end # Blox is the abstract type for Blox that are displayed in the GUI
abstract type AbstractComponent end
abstract type SuperBlox <: AbstractBlox end
abstract type AbstractNeuronBlox <: AbstractBlox end

@parameters t
D = Differential(t)

sigmoid(x::Real, r::Real) = one(x) / (one(x) + exp(-r*x)) - 0.5

"""
Jansen-Rit model block for canonical micro circuit, analogous to the implementation in SPM12
"""
mutable struct jansen_rit_spm12 <: AbstractComponent
    τ::Num
    r::Num
    connector::Num
    odesystem::ODESystem
    function jansen_rit_spm12(;name, τ=1.0, r=2.0/3.0)
        params = @parameters τ=τ
        sts    = @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0   # should have a conditional here that decides whether something is an output or not
        eqs    = [D(x) ~ y,                                # TODO: shouldn't -2*x/τ be in this line? However, see Friston2012 and SPM12 implementation.
                  D(y) ~ (-2*x - x/τ + jcn)/τ]
        odesys = ODESystem(eqs, t, sts, params; name=name)
        new(τ, r, sigmoid(odesys.x, r), odesys)
    end
end

mutable struct CanonicalMicroCircuit <: SuperBlox
    τ_ss::Num
    τ_sp::Num
    τ_ii::Num
    τ_dp::Num
    r_ss::Num
    r_sp::Num
    r_ii::Num
    r_dp::Num
    connector::Symbolics.Arr{Num}
    noDetail::Vector{Num}
    detail::Vector{Num}
    bloxinput::Symbolics.Arr{Num}
    odesystem::ODESystem
    function CanonicalMicroCircuit(;name, τ_ss=0.002, τ_sp=0.002, τ_ii=0.016, τ_dp=0.028, r_ss=2.0/3.0, r_sp=2.0/3.0, r_ii=2.0/3.0, r_dp=2.0/3.0)
        @variables jcn(t)[1:4], x(t)[1:4]

        @named ss = jansen_rit_spm12(τ=τ_ss, r=r_ss)  # spiny stellate
        @named sp = jansen_rit_spm12(τ=τ_sp, r=r_sp)  # superficial pyramidal
        @named ii = jansen_rit_spm12(τ=τ_ii, r=r_ii)  # inhibitory interneurons granular layer
        @named dp = jansen_rit_spm12(τ=τ_dp, r=r_dp)  # deep pyramidal

        g = MetaDiGraph()
        add_vertex!(g, Dict(:blox => ss, :name => name, :jcn => jcn[1]))
        add_vertex!(g, Dict(:blox => sp, :name => name, :jcn => jcn[2]))
        add_vertex!(g, Dict(:blox => ii, :name => name, :jcn => jcn[3]))
        add_vertex!(g, Dict(:blox => dp, :name => name, :jcn => jcn[4]))

        add_edge!(g, 1, 1, :weight, -800.0)
        add_edge!(g, 2, 1, :weight, -800.0)
        add_edge!(g, 3, 1, :weight, -800.0)
        add_edge!(g, 1, 2, :weight,  800.0)
        add_edge!(g, 2, 2, :weight, -800.0)
        add_edge!(g, 1, 3, :weight,  800.0)
        add_edge!(g, 3, 3, :weight, -800.0)
        add_edge!(g, 4, 3, :weight,  400.0)
        add_edge!(g, 3, 4, :weight, -400.0)
        add_edge!(g, 4, 4, :weight, -200.0)

        @named odecmc = ODEfromGraph(g)
        eqs = [
            x[1] ~ ss.connector
            x[2] ~ sp.connector
            x[3] ~ ii.connector
            x[4] ~ dp.connector
        ]
        odesys = extend(ODESystem(eqs, t, name=:connected), odecmc, name=name)
        new(τ_ss, τ_sp, τ_ii, τ_dp, r_ss, r_sp, r_ii, r_dp, odesys.x, [odesys.ss.x,odesys.sp.x,odesys.ii.x,odesys.dp.x], [odesys.ss.x,odesys.sp.x,odesys.ii.x,odesys.dp.x], odesys.jcn, odesys)
    end
end

function ODEfromGraph(g::MetaDiGraph ;name)
    eqs = []
    sys = []
    for v in vertices(g)
        b = get_prop(g, v, :blox)
        if isa(b, AbstractBlox) || isa(b, AbstractComponent)
            s = b.odesystem
            push!(sys, s)
            if any(occursin.("jcn(t)", string.(states(s))))
                if isa(b, AbstractNeuronBlox)
                    input = Num(0)
                    for vn in inneighbors(g, v) # vertices that point towards s
                        bn = get_prop(g, vn, :blox)
                        if !isa(bn, AbstractNeuronBlox) # only neurons can be inputs to neurons
                            continue
                        end
                        input += bn.connector * get_prop(g, vn, v, :weight) * (bn.odesystem.E_syn - s.V)
                    end
                    push!(eqs, s.I_syn ~ input)
                    push!(eqs, s.jcn ~ s.I_syn)
                else
                    if s.jcn isa Symbolics.Arr
                        bi = b.bloxinput # bloxinput only exists if s.jcn isa Symbolics.Arr
                        input = [zeros(Num,length(s.jcn))]
                        for vn in inneighbors(g, v) # vertices that point towards s
                            M = get_prop(g, vn, v, :weightmatrix)
                            connector = get_prop(g, vn, :blox).connector
                            if connector isa Symbolics.Arr
                                connector = collect(connector)
                            end
                            push!(input, vec(M*connector))
                        end
                        input = sum(input)
                        for i = 1:length(s.jcn)
                            push!(eqs, bi[i] ~ input[i])
                        end
                    else
                        input = Num(0)
                        for vn in inneighbors(g, v) # vertices that point towards s
                            connector = get_prop(g,vn,:blox).connector
                            if connector isa Symbolics.Arr
                                input += sum(vec(get_prop(g, vn, v, :weightmatrix)*collect(connector)))
                            else
                                input += connector * get_prop(g, vn, v, :weight)
                            end
                        end
                        if haskey(props(g,v),:jcn)
                            input += get_prop(g,v,:jcn)
                        end
                        push!(eqs, s.jcn ~ input)
                    end
                end
            end
        end
    end
    return compose(ODESystem(eqs, t; name=:connected), sys; name=name)
end

# g = MetaDiGraph()
# nr = 2
# add_vertex!(g, Dict(:blox => CanonicalMicroCircuit(;name=:r1))) # V1 (see fig. 4 in Bastos et al. 2015)
# add_vertex!(g, Dict(:blox => CanonicalMicroCircuit(;name=:r2))) # V4 (see fig. 4 in Bastos et al. 2015)

# nl = Int((nr^2-nr)/2)   # number of links unidirectional
# @parameters a_ss[1:nl] = repeat([0.0], nl) # forward connection parameter sp -> ss
# @parameters a_dp[1:nl] = repeat([0.0], nl) # forward connection parameter sp -> dp
# @parameters a_sp[1:nl] = repeat([0.0], nl) # backward connection parameter dp -> sp
# @parameters a_ii[1:nl] = repeat([0.0], nl) # backward connection parameters dp -> ii

# for i in 1:nr
#     for j in (i+1):nr
#         add_edge!(g, i, j, :weightmatrix, 
#                 [0 exp(a_ss[i]) 0 0;
#                 0 0 0 0;
#                 0 0 0 0;
#                 0 exp(a_dp[i])/2 0 0] * 200)

#         add_edge!(g, j, i, :weightmatrix,
#                 [0 0 0 0;
#                 0 0 0 -exp(a_sp[i]);
#                 0 0 0 -exp(a_ii[i])/2;
#                 0 0 0 0] * 200)
#     end
# end

# @named cmc_network = ODEfromGraph(g)
# cmc_network = structural_simplify(cmc_network)




# Random.seed!(1234)
# for i = 1:nr
#     for j = 1:nr
#         if i == j continue end
#         nodes_source = nv(regions[i].lngraph.graph)
#         nodes_sink = nv(regions[j].lngraph.graph)
#         A[i, j] = rand(nodes_source, nodes_sink)
#     end
# end
# @named manyregions = connectcomplexblox(regions, A)
# manyregions = structural_simplify(manyregions)


# prob = ODEProblem(manyregions, zeros(length(manyregions.states)), (0, 1), [])
# sol = solve(prob, AutoVern7(Rodas4()))
# plot(sol)



# @variables t x(t)   # independent and dependent variables
# p = @parameters τ h=1.0 a=1       # parameters
# D = Differential(t) # define an operator for the differentiation w.r.t. time

# # your first ODE, consisting of a single equation, the equality indicated by ~
# @named fol = ODESystem([ D(x)  ~ (h - x)^a/τ])

# using DifferentialEquations: solve

# prob = ODEProblem(fol, [x => 0.0], (0.0,10.0), [p[1] => 3.0, p[2] => 0.1, p[3] => 2])
# idxmap = Dict(p[1] => 1, p[2] => 2, p[3] => 3)
# idxs = Int.(ModelingToolkit.varmap_to_vars([p[1] => 1, p[2] => 2, p[3] => 3], p))
# p_new = prob.p
# p_new[idxs[idxmap[τ]]]
# remake()
# findfirst(p .== τ)
# prob = remake(prob, p=[1,1,1])
# # parameter `τ` can be assigned a value, but constant `h` cannot
# sol = solve(prob)

# using Plots
# plot(sol)

# params = @parameters τ r
# sts    = @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0
# eqs    = [D(x) ~ y - ((2/τ)*x),
#           D(y) ~ -x/(τ*τ) + jcn/τ]
# @named odesys = ODESystem(eqs, t, sts, params; defaults=Dict(τ=>0.0, r=>[0.0, 0.5]))
# odesys = structural_simplify(odesys)
# prob = ODEProblem(odesys, zeros(length(odesys.states)), (0, 1), [])

# remake(prob)
