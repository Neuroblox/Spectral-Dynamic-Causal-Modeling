# Canonical micro-circuit model
# two ways to design it: either add function to edges or to blox.

# some thoughts on design:
# - include measurement models into blox. Or at least define which variables will be measured (which would be the input to the measurement model). 
#   Differ that from connector, since that is between things.
# - 
using ModelingToolkit
using DifferentialEquations
using DifferentialEquations
using MetaGraphs
using Graphs
using Random
using Plots

@parameters t
D = Differential(t)

# define a sigmoid function
sigmoid(x::Real, r) = one(x) / (one(x) + exp(-2.0/3.0*exp(r[2])*x + r[1]))

"""
Jansen-Rit model block for canonical micro circuit, analogous to the implementation in SPM12
"""
mutable struct jansen_rit_spm12
    τ::Num
    r::Num
    connector::Num
    odesystem::ODESystem
    function jansen_rit_spm12(;name, τ_init=0.0, r_init=[0.0, 0.0])
        params = @parameters τ=τ_init r
        sts    = @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0
        eqs    = [D(x) ~ y - ((2/τ)*x),
                  D(y) ~ -x/(τ*τ) + jcn/τ]
        odesys = ODESystem(eqs, t, sts, params; name=name, defaults=Dict(τ=>τ_init, r=>r_init))
        new(τ, r, sigmoid(odesys.x, r_init), odesys)
    end
end

function LinearConnections(;name, sys=sys, adj_matrix=adj_matrix, connector=connector)
    adj = adj_matrix .* connector
    eqs = []
    for region_num in 1:length(sys)
       push!(eqs, sys[region_num].jcn ~ sum(adj[:, region_num]))
    end
    return ODESystem(eqs, name=name, systems=sys)
end

function adjmatrixfromdigraph(g::MetaDiGraph)
    myadj = map(Num, adjacency_matrix(g))
    for edge in edges(g)
        s = src(edge)
        d = dst(edge)
        myadj[s,d] = get_prop(g, edge, :weight)
    end
    return myadj
end

function ODEfromGraph(g::MetaDiGraph ;name)
    blox = [get_prop(g, v, :blox) for v in vertices(g)]
    sys = [s.odesystem for s in blox]
    connector = [s.connector for s in blox]
    adj = adjmatrixfromdigraph(g)
    return LinearConnections(name=name, sys=sys, adj_matrix=adj, connector=connector)
end

mutable struct cmc
    τ::Vector{Num}
    r::Vector{Num}
    odesystem::ODESystem
    lngraph::MetaDiGraph
    function cmc(;name, τ=[0.002, 0.002, 0.016, 0.028], r=[[1.0, 2.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        ss = jansen_rit_spm12(τ_init=τ[1], r_init=r[1], name=Symbol(String(name)*"_ss"))  # spiny stellate
        sp = jansen_rit_spm12(τ_init=τ[2], r_init=r[2], name=Symbol(String(name)*"_sp"))  # superficial pyramidal
        ii = jansen_rit_spm12(τ_init=τ[3], r_init=r[3], name=Symbol(String(name)*"_ii"))  # inhibitory interneurons granular layer
        dp = jansen_rit_spm12(τ_init=τ[4], r_init=r[4], name=Symbol(String(name)*"_dp"))  # deep pyramidal

        g = MetaDiGraph()
        add_vertex!(g, Dict(:blox => ss, :name => name))
        add_vertex!(g, Dict(:blox => sp, :name => name))
        add_vertex!(g, Dict(:blox => ii, :name => name))
        add_vertex!(g, Dict(:blox => dp, :name => name))

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

        odesys = ODEfromGraph(g, name=name)
        new(τ, r, odesys, g)
    end
end

function joinmetagraphs(metagraphs::Vector{T}) where T <: Any
    ngraphs = length(metagraphs)
    
    wholegraph = MetaDiGraph()
    nvertex = 0
    for i = 1:ngraphs
        for j in vertices(metagraphs[i].lngraph)
            add_vertex!(wholegraph, props(metagraphs[i].lngraph, j))
        end
        for e in edges(metagraphs[i].lngraph)
            add_edge!(wholegraph, nvertex+src(e), nvertex+dst(e), props(metagraphs[i].lngraph, e))
        end
        nvertex += nv(metagraphs[i].lngraph)
    end
    return wholegraph
end

function connectcomplexblox(bloxlist, adjacency_matrices ;name)
    nr = length(bloxlist)
    g = joinmetagraphs(bloxlist)
    row = 0
    for i = 1:nr
        nodes_source = nv(bloxlist[i].lngraph)
        col = 0
        for j = 1:nr
            nodes_sink = nv(bloxlist[j].lngraph)
            if i == j
                col += nodes_sink
                continue
            end
            for idx in CartesianIndices(adjacency_matrices[i, j])
                add_edge!(g, row+idx[1], col+idx[2], :weight, adjacency_matrices[i, j][idx])
            end
            col += nodes_sink
        end
        row += nodes_source
    end
    
    return ODEfromGraph(g, name=name)
end

regions = []
nr = 2
for i = 1:nr
    push!(regions, cmc(name=Symbol("r$i")))
end


A = Array{Matrix{Float64}}(undef, nr, nr);
Random.seed!(1234)
for i = 1:nr
    for j = 1:nr
        if i == j continue end
        nodes_source = nv(regions[i].lngraph.graph)
        nodes_sink = nv(regions[j].lngraph.graph)
        A[i, j] = rand(nodes_source, nodes_sink)
    end
end
@named manyregions = connectcomplexblox(regions, A)
manyregions = structural_simplify(manyregions)

prob = ODEProblem(manyregions, zeros(length(manyregions.states)), (0,1), [])
sol = solve(prob, AutoVern7(Rodas4()))
plot(sol)



@variables t x(t)   # independent and dependent variables
p = @parameters τ h=1.0 a=1       # parameters
D = Differential(t) # define an operator for the differentiation w.r.t. time

# your first ODE, consisting of a single equation, the equality indicated by ~
@named fol = ODESystem([ D(x)  ~ (h - x)^a/τ])

using DifferentialEquations: solve

prob = ODEProblem(fol, [x => 0.0], (0.0,10.0), [p[1] => 3.0, p[2] => 0.1, p[3] => 2])
idxmap = Dict(p[1] => 1, p[2] => 2, p[3] => 3)
idxs = Int.(ModelingToolkit.varmap_to_vars([p[1] => 1, p[2] => 2, p[3] => 3], p))
p_new = prob.p
p_new[idxs[idxmap[τ]]]
remake()
findfirst(p .== τ)
prob = remake(prob, p=[1,1,1])
# parameter `τ` can be assigned a value, but constant `h` cannot
sol = solve(prob)

using Plots
plot(sol)

params = @parameters τ r
sts    = @variables x(t)=1.0 y(t)=1.0 jcn(t)=0.0
eqs    = [D(x) ~ y - ((2/τ)*x),
          D(y) ~ -x/(τ*τ) + jcn/τ]
@named odesys = ODESystem(eqs, t, sts, params; defaults=Dict(τ=>0.0, r=>[0.0, 0.5]))
odesys = structural_simplify(odesys)
prob = ODEProblem(odesys, zeros(length(odesys.states)), (0, 1), [])

remake(prob)
