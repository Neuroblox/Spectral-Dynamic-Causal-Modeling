"""
    LinearNeuralMass(name, namespace)

Create standard linear neural mass blox with a single internal state.
There are no parameters in this blox.
This is a blox of the sort used for spectral DCM modeling.
The formal definition of this blox is:


```math
\\frac{d}{dx} = \\sum{jcn}
```

where ``jcn``` is any input to the blox.


Arguments:
- name: Options containing specification about deterministic.
- namespace: Additional namespace above name if needed for inheritance.
"""

struct LinearNeuralMass <: NeuralMassBlox
    system
    namespace

    function LinearNeuralMass(;name, namespace=nothing)
        sts = @variables x(t)=0.0 [output=true] jcn(t) [input=true]
        eqs = [D(x) ~ jcn]
        sys = System(eqs, t, name=name)
        new(sys, namespace)
    end
end

# Simple input blox
mutable struct ExternalInput <: StimulusBlox
    namespace
    system

    function ExternalInput(;name, I=0.0, namespace=nothing)
        sts = @variables u(t)=0.0 [output=true, irreducible=true, description="ext_input"]
        eqs = [u ~ I]
        odesys = System(eqs, t, sts, []; name=name)

        new(namespace, odesys)
    end
end

"""
Ornstein-Uhlenbeck process Blox

variables:
    x(t):  value
    jcn:   input 
parameters:
    τ:      relaxation time
	μ:      average value
	σ:      random noise (variance of OU process is τ*σ^2/2)
returns:
    an ODE System (but with brownian parameters)
"""
mutable struct OUBlox <: NeuralMassBlox
    # all parameters are Num as to allow symbolic expressions
    namespace
    stochastic
    system
    function OUBlox(;name, namespace=nothing, μ=0.0, σ=1.0, τ=1.0)
        p = paramscoping(μ=μ, τ=τ, σ=σ)
        μ, τ, σ = p
        sts = @variables x(t)=0.0 [output=true] jcn(t) [input=true]
        @brownian w

        eqs = [D(x) ~ (-x + μ + jcn)/τ + sqrt(2/τ)*σ*w]
        sys = System(eqs, t; name=name)
        new(namespace, true, sys)
    end
end

# Canonical micro-circuit model
"""
Jansen-Rit model block for canonical micro circuit, analogous to the implementation in SPM12
"""
mutable struct JansenRitSPM <: NeuralMassBlox
    params
    output
    jcn
    odesystem
    namespace
    function JansenRitSPM(;name, namespace=nothing, τ=1.0, r=2.0/3.0)
        p = paramscoping(τ=τ, r=r)
        τ, r = p

        sts    = @variables x(t)=0.0 [output=true] y(t)=0.0 jcn(t)=0.0 [input=true]
        eqs    = [D(x) ~ y,                                # TODO: shouldn't -2*x/τ be in this line? However, see Friston2012 and SPM12 implementation.
                  D(y) ~ (-2*y - x/τ + jcn)/τ]

        sys = System(eqs, t, name=name)
        new(p, sts[1], sts[3], sys, namespace)
    end
end

mutable struct CanonicalMicroCircuitBlox <: CompositeBlox
    namespace
    parts
    odesystem
    connector
    function CanonicalMicroCircuitBlox(;name, namespace=nothing, τ_ss=0.002, τ_sp=0.002, τ_ii=0.016, τ_dp=0.028, r_ss=2.0/3.0, r_sp=2.0/3.0, r_ii=2.0/3.0, r_dp=2.0/3.0)
        @named ss = JansenRitSPM12(;namespace=namespaced_name(namespace, name), τ=τ_ss, r=r_ss)  # spiny stellate
        @named sp = JansenRitSPM12(;namespace=namespaced_name(namespace, name), τ=τ_sp, r=r_sp)  # superficial pyramidal
        @named ii = JansenRitSPM12(;namespace=namespaced_name(namespace, name), τ=τ_ii, r=r_ii)  # inhibitory interneurons granular layer
        @named dp = JansenRitSPM12(;namespace=namespaced_name(namespace, name), τ=τ_dp, r=r_dp)  # deep pyramidal

        g = MetaDiGraph()
        sblox_parts = vcat(ss, sp, ii, dp)
        add_blox!.(Ref(g), sblox_parts)
        @parameters w=1.0 [tunable=false]
        add_edge!(g, 1, 1, :weight, -800.0*w)
        add_edge!(g, 2, 1, :weight, -800.0*w)
        add_edge!(g, 3, 1, :weight, -1600.0*w)
        add_edge!(g, 1, 2, :weight,  800.0*w)
        add_edge!(g, 2, 2, :weight, -800.0*w)
        add_edge!(g, 1, 3, :weight,  800.0*w)
        add_edge!(g, 3, 3, :weight, -800.0*w)
        add_edge!(g, 4, 3, :weight,  400.0*w)
        add_edge!(g, 3, 4, :weight, -400.0*w)
        add_edge!(g, 4, 4, :weight, -200.0*w)

        # Construct a BloxConnector object from the graph
        # containing all connection equations from lower levels and this level.
        bc = connector_from_graph(g)
        # If a namespace is not provided, assume that this is the highest level
        # and construct the ODEsystem from the graph.
        # If there is a higher namespace, construct only a subsystem containing the parts of this level
        # and propagate the BloxConnector object `bc` to the higher level 
        # to potentially add more terms to the same connections.
        sys = isnothing(namespace) ? system_from_graph(g, bc; name) : system_from_parts(sblox_parts; name)

        new(namespace, sblox_parts, sys, bc)
    end
end
