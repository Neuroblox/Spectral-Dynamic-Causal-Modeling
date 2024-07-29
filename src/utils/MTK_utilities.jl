const t = ModelingToolkit.t_nounits
const D = ModelingToolkit.D_nounits

### Blox Connector and Utilities ###

mutable struct BloxConnector
    eqs::Vector{Equation}
    weights::Vector{Num}

    BloxConnector() = new(Equation[], Num[])

    function BloxConnector(bloxs)
        eqs = reduce(vcat, input_equations.(bloxs)) 
        weights = reduce(vcat, weight_parameters.(bloxs))

        new(eqs, weights)
    end
end

function accumulate_equation!(bc::BloxConnector, eq)
    lhs = eq.lhs
    idx = find_eq(bc.eqs, lhs)
    bc.eqs[idx] = bc.eqs[idx].lhs ~ bc.eqs[idx].rhs + eq.rhs
end

get_equations_with_parameter_lhs(bc) = filter(eq -> ModelingToolkit.isparameter(eq.lhs), bc.eqs)

get_equations_with_state_lhs(bc) = filter(eq -> !ModelingToolkit.isparameter(eq.lhs), bc.eqs)


function generate_weight_param(blox_out, blox_in; kwargs...)
    name_out = namespaced_nameof(blox_out)
    name_in = namespaced_nameof(blox_in)

    weight = get_weight(kwargs, name_out, name_in)
    w_name = Symbol("w_$(name_out)_$(name_in)")
    if typeof(weight) == Num   # Symbol
        w = weight
    else
        w = only(@parameters $(w_name)=weight)
    end    

    return w
end


function (bc::BloxConnector)(
    bloxout::CanonicalMicroCircuitBlox,
    bloxin::CanonicalMicroCircuitBlox;
    kwargs...
)
    sysparts_out = get_blox_parts(bloxout)
    sysparts_in = get_blox_parts(bloxin)

    wm = get_weightmatrix(kwargs, namespaced_nameof(bloxin), namespaced_nameof(bloxout))

    idxs = findall(!iszero, wm)
    for idx in idxs
        bc(sysparts_out[idx[2]], sysparts_in[idx[1]]; weight=wm[idx])
    end
end

function (bc::BloxConnector)(
    bloxout::CanonicalMicroCircuitBlox,
    bloxin::ObserverBlox;
    kwargs...
)
    sysparts_out = get_blox_parts(bloxout)

    bc(sysparts_out[2], bloxin; kwargs...)
end

# define a sigmoid function
# @register_symbolic sigmoid(x, r) = one(x) / (one(x) + exp(-r*x))
sigmoid(x, r) = one(x) / (one(x) + exp(-r*x)) - 0.5

function (bc::BloxConnector)(
    bloxout::JansenRitSPM12, 
    bloxin::JansenRitSPM12; 
    kwargs...
)
    sys_out = get_namespaced_sys(bloxout)
    sys_in = get_namespaced_sys(bloxin)

    w = generate_weight_param(bloxout, bloxin; kwargs...)
    push!(bc.weights, w)

    x = ModelingToolkit.namespace_expr(bloxout.output, sys_out)
    r = ModelingToolkit.namespace_expr(bloxout.params[2], sys_out)
    push!(bc.weights, r)

    eq = sys_in.jcn ~ sigmoid(x, r)*w
    
    accumulate_equation!(bc, eq)
end


function (bc::BloxConnector)(
    bloxout::NeuralMassBlox, 
    bloxin::NeuralMassBlox;
    kwargs...
)
    sys_out = get_namespaced_sys(bloxout)
    sys_in = get_namespaced_sys(bloxin)

    w = generate_weight_param(bloxout, bloxin; kwargs...)
    push!(bc.weights, w)

    x = ModelingToolkit.namespace_expr(bloxout.output, sys_out)
    eq = sys_in.jcn ~ x*w
    
    accumulate_equation!(bc, eq)
end

# additional dispatch to connect to hemodynamic observer blox
function (bc::BloxConnector)(
    bloxout::NeuralMassBlox, 
    bloxin::ObserverBlox; 
    weight=1
)

    sys_out = get_namespaced_sys(bloxout)
    sys_in = get_namespaced_sys(bloxin)

    w_name = Symbol("w_$(nameof(sys_out))_$(nameof(sys_in))")
    if typeof(weight) == Num # Symbol
        w = weight
    else
        w = only(@parameters $(w_name)=weight [tunable=false])
    end
    push!(bc.weights, w)
    x = ModelingToolkit.namespace_expr(bloxout.output, sys_out, nameof(sys_out))
    eq = sys_in.jcn ~ x*w

    accumulate_equation!(bc, eq)
end

function (bc::BloxConnector)(
    bloxout::StimulusBlox,
    bloxin::CanonicalMicroCircuitBlox;
    kwargs...
)

    sysparts_in = get_blox_parts(bloxin)

    bc(bloxout, sysparts_in[1]; kwargs...)
end

function (bc::BloxConnector)(
    bloxout::StimulusBlox,
    bloxin::NeuralMassBlox;
    weight=1
)

    sys_out = get_namespaced_sys(bloxout)
    sys_in = get_namespaced_sys(bloxin)

    w_name = Symbol("w_$(nameof(sys_out))_$(nameof(sys_in))")
    if typeof(weight) == Num # Symbol
        w = weight
    else
        w = only(@parameters $(w_name)=weight)
    end    
    push!(bc.weights, w)

    x = ModelingToolkit.namespace_expr(bloxout.output, sys_out, nameof(sys_out))
    eq = sys_in.jcn ~ x*w

    accumulate_equation!(bc, eq)
end

### Namespacing Utilities ###

function get_namespaced_sys(blox)
    sys = get_sys(blox)
    System(
        equations(sys), 
        independent_variable(sys), 
        unknowns(sys), 
        parameters(sys); 
        name = namespaced_nameof(blox)
    ) 
end

get_namespaced_sys(sys::ModelingToolkit.AbstractODESystem) = sys

import ModelingToolkit: nameof
nameof(blox) = (nameof ∘ get_sys)(blox)

namespaceof(blox) = blox.namespace

namespaced_nameof(blox) = namespaced_name(inner_namespaceof(blox), nameof(blox))

"""
    Returns the complete namespace EXCLUDING the outermost (highest) level.
    This is useful for manually preparing equations (e.g. connections, see BloxConnector),
    that will later be composed and will automatically get the outermost namespace.
""" 
function inner_namespaceof(blox)
    parts = split((string ∘ namespaceof)(blox), '₊')
    if length(parts) == 1
        return nothing
    else
        return join(parts[2:end], '₊')
    end
end

namespaced_name(parent_name, name) = Symbol(parent_name, :₊, name)
namespaced_name(::Nothing, name) = Symbol(name)

function find_eq(eqs::AbstractVector{<:Equation}, lhs)
    findfirst(eqs) do eq
        lhs_vars = get_variables(eq.lhs)
        length(lhs_vars) == 1 && isequal(only(lhs_vars), lhs)
    end
end

"""
    Returns the equations for all input variables of a system, 
    assuming they have a form like : `sys.input_variable ~ ...`
    so only the input appears on the LHS.

    Input equations are namespaced by the inner namespace of blox
    and then they are returned. This way during system `compose` downstream,
    the higher-level namespaces will be added to them.

    If blox isa AbstractComponent, it is assumed that it contains a `connector` field,
    which holds a `BloxConnector` object with all relevant connections 
    from lower levels and this level.
"""
function input_equations(blox)
    sys = get_sys(blox)
    inps = ModelingToolkit.inputs(sys)
    sys_eqs = equations(sys)

    eqs = map(inps) do inp
        idx = find_eq(sys_eqs, inp)
        if isnothing(idx)
            ModelingToolkit.namespace_equation(
                inp ~ 0, 
                sys,
                namespaced_name(inner_namespaceof(blox), nameof(blox))
            )
        else
            ModelingToolkit.namespace_equation(
                sys_eqs[idx],
                sys,
                namespaced_name(inner_namespaceof(blox), nameof(blox))
            )
        end
    end

    return eqs
end

input_equations(blox::AbstractComponent) = blox.connector.eqs
input_equations(blox::CompositeBlox) = blox.connector.eqs

weight_parameters(blox) = Num[]
weight_parameters(blox::AbstractComponent) = blox.connector.weights #I think this is the fix?
weight_parameters(blox::CompositeBlox) = blox.connector.weights #I think this is the fix?


get_blox_parts(blox) = blox.parts

function get_weight(kwargs, name_blox1, name_blox2)
    if haskey(kwargs, :weight)
        return kwargs[:weight]
    else
        error("Connection weight from $name_blox1 to $name_blox2 is not specified.")
    end
end

function get_weightmatrix(kwargs, name_blox1, name_blox2)
    if haskey(kwargs, :weightmatrix)
        return kwargs[:weightmatrix]
    else
        error("Connection weight from $name_blox1 to $name_blox2 is not specified.")
    end
end


### State and Parameter Utilities ###

"""
    Helper to merge delays and weights into a single vector
"""
function params(bc::BloxConnector)
    weights = []
    for w in bc.weights
        append!(weights, Symbolics.get_variables(w))
    end
    if isempty(weights)
        return weights
    else
        return reduce(vcat, weights)
    end
end


"""
    function paramscoping(;kwargs...)
    
    Scope arguments that are already a symbolic model parameter thereby keep the correct namespace 
    and make those that are not yet symbolic a symbol.
    Keyword arguments are used, because parameter definition require names, not just values.
"""
function paramscoping(;kwargs...)
    paramlist = []
    for (kw, v) in kwargs
        if v isa Num
            paramlist = vcat(paramlist, ParentScope(v))
        else
            paramlist = vcat(paramlist, @parameters $kw = v [tunable=true])
        end
    end
    return paramlist
end

"""
    function get_dynamic_states(sys)
    
    Function extracts states from the system that are dynamic variables, 
    get also indices of external inputs (u(t)) and measurements (like bold(t))
    Arguments:
    - `sys`: MTK system

    Returns:
    - `sts`  : states of the system that are neither external inputs nor measurements, i.e. these are the dynamic states
    - `idx_u`: indices of states that represent external inputs
    - `idx_m`: indices of states that represent measurements
"""
function get_dynamic_states(sys)
    sts = []
    idx = []
    for (i, s) in enumerate(unknowns(sys))
        if !((getdescription(s) == "ext_input") || (getdescription(s) == "measurement"))
            push!(sts, s)
            push!(idx, i)
        end
    end
    return sts, idx
end

function get_eqidx_tagged_vars(sys, tag)
    idx = Int[]
    vars = []
    eqs = equations(sys)
    for s in unknowns(sys)
        if getdescription(s) == tag
            push!(vars, s)
        end
    end

    for v in vars
        for (i, e) in enumerate(eqs)
            for s in Symbolics.get_variables(e)
                if string(s) == string(v)
                    push!(idx, i)
                end 
            end
        end
    end
    return idx
end

function get_idx_tagged_vars(sys, tag)
    idx = Int[]
    for (i, s) in enumerate(unknowns(sys))
        if (getdescription(s) == tag)
            push!(idx, i)
        end
    end
    return idx
end


"""
    function addnontunableparams(param, model)
    
    Function adds parameters of a model that were not marked as tunable to a list of tunable parameters
    and respects the MTK ordering of parameters.

    Arguments:
    - `paramlist`: parameters of an MTK system that were tagged as tunable
    - `sys`: MTK system

    Returns:
    - `completeparamlist`: complete parameter list of a system, including those that were not tagged as tunable
"""
function addnontunableparams(paramlist, sys)
    completeparamlist = []
    k = 0
    for p in parameters(sys)
        if istunable(p)
            k += 1
            push!(completeparamlist, paramlist[k])
        else
            push!(completeparamlist, Symbolics.getdefaultval(p))
        end
    end
    append!(completeparamlist, paramlist[k+1:end])

    return completeparamlist
end

### Graph Utilities ###

function add_blox!(g::MetaDiGraph,blox)
    add_vertex!(g, :blox, blox)
end

function get_blox(g::MetaDiGraph)
    bs = []
    for v in vertices(g)
        b = get_prop(g, v, :blox)
        push!(bs, b)
    end

    return bs
end

get_sys(g::MetaDiGraph) = get_sys.(get_blox(g))
get_sys(blox) = blox.odesystem
get_sys(sys::ModelingToolkit.AbstractODESystem) = sys


function connector_from_graph(g::MetaDiGraph)
    bloxs = get_blox(g)
    link = BloxConnector(bloxs)
    for v in vertices(g)
        b = get_prop(g, v, :blox)
        for vn in inneighbors(g, v)
            bn = get_prop(g, vn, :blox)
            kwargs = props(g, vn, v)
            link(bn, b; kwargs...)
        end
    end

    return link
end

function system_from_graph(g::MetaDiGraph; name, t_block=missing)
    bc = connector_from_graph(g)
    return system_from_graph(g, bc; name, t_block)
end

# Additional dispatch if extra parameters are passed for edge definitions
function system_from_graph(g::MetaDiGraph, p::Vector{Num}; name, t_block=missing)
    bc = connector_from_graph(g)
    return system_from_graph(g, bc, p; name, t_block)
end

function system_from_graph(g::MetaDiGraph, bc::BloxConnector; name, t_block=missing)
    blox_syss = get_sys(g)

    connection_eqs = get_equations_with_state_lhs(bc)
    return compose(System(connection_eqs, t, [], params(bc); name), blox_syss)
end

# function system_from_graph(g::MetaDiGraph, bc::BloxConnector, p::Vector{Num}; name, t_block=missing)
#     blox_syss = get_sys(g)

#     connection_eqs = get_equations_with_state_lhs(bc)
    
#     cbs = identity.(get_callbacks(g, bc; t_block))

#     return compose(System(connection_eqs, t, [], vcat(params(bc), p); name, discrete_events = cbs), blox_syss)
# end

function system_from_parts(parts::AbstractVector; name)
    return compose(ODESystem(Equation[], t, [], []; name), get_sys.(parts))
end
