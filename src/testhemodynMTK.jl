# questions for Chris:
# https://docs.sciml.ai/ModelingToolkit/stable/basics/Composition/#Inheritance-and-Combine
# shoudn't the initial conditions be set with S I and R?
# - argument why to use different names for symbolic and for initial condition and use Dict instead of directly attributing the value


using DifferentialEquations
using ModelingToolkit
using Plots

@parameters t
D = Differential(t)

function hemodynamicsMTK(lndecay, lntransit; name)
    #= hemodynamic parameters
        H(1) - signal decay                                   d(ds/dt)/ds)
        H(2) - autoregulation                                 d(ds/dt)/df)
        H(3) - transit time                                   (t0)
        H(4) - exponent for Fout(v)                           (alpha)
        H(5) - resting oxygen extraction                      (E0)
    =#
    H = [0.64, 0.32, 2.00, 0.32, 0.4]

    params = @parameters κ τ
    states = @variables s(t) lnf(t) lnν(t) lnq(t) x(t)

    eqs = [
        D(s)   ~ x - H[1]*exp(κ)*s - H[2]*(exp(lnf) - 1),
        D(lnf) ~ s / exp(lnf),
        D(lnν) ~ (exp(lnf) - exp(lnν)^(H[4]^-1)) / (H[3]*exp(τ)*exp(lnν)),
        D(lnq) ~ (exp(lnf)/exp(lnq)*((1 - (1 - H[5])^(exp(lnf)^-1))/H[5]) - exp(lnν)^(H[4]^-1 - 1))/(H[3]*exp(τ))
    ]

    return ODESystem(eqs, t, states, params; name=name, defaults=Dict(κ=>lndecay, τ=>lntransit))
end


function boldsignal(lnϵ; name)
    # NB: Biophysical constants for 1.5T scanners
    # Time to echo
    TE  = 0.04
    # resting venous volume (%)
    V0  = 4
    # slope r0 of intravascular relaxation rate R_iv as a function of oxygen 
    # saturation S:  R_iv = r0*[(1 - S)-(1 - S0)] (Hz)
    r0  = 25
    # frequency offset at the outer surface of magnetized vessels (Hz)
    nu0 = 40.3
    # resting oxygen extraction fraction
    E0  = 0.4
    # -Coefficients in BOLD signal model
    k1  = 4.3*nu0*E0*TE

    params = @parameters ϵ=lnϵ
    vars = @variables bold(t) jcn[1:2](t) q(t) ν(t)

    eqs = [
        bold ~ V0*(k1 - k1*exp(q) + ϵ*r0*E0*TE - ϵ*r0*E0*TE*exp(q)/exp(ν) + 1-ϵ - (1-ϵ)*exp(ν))
    ]

    ODESystem(eqs, t, vars, params; name=name)
end

@named bar = boldsignal(0.0)

calculate_jacobian(bar)

function linearneuralmass(;name)
    states = @variables x(t) jcn(t)
    eqs = D(x) ~ jcn
    return ODESystem(eqs, t, states, []; name=name)
end

function linearconnectionssymbolic(;name, sys=sys, adj_matrix=adj_matrix, connector=connector)
    eqs = []
    nr = length(sys)
    for i in 1:nr
       push!(eqs, sys[i].nmm.jcn ~ sum(adj_matrix[(1+(i-1)*nr):nr*i] .* connector))
    end
    return ODESystem(eqs, name=name, systems=sys)
end

function connectblox(bloxlist, A)
    g = MetaDiGraph()
    for blox in bloxlist
        add_vertex!(g, Dict(:blox => blox))
    end

    nonzeroelems = findall(x -> x != 0, A)
    for idx in nonzeroelems
        add_edge!(g, Tuple(idx)..., :weight, A[idx])
    end
    return g
end

function odefromgraph(g::MetaDiGraph; name)
    bloxlist = [get_prop(g, v, :blox) for v in vertices(g)]
    eqs = []
    buildexp = ""
    for v in vertices(g)
        for vn in inneighbors(g, v)
            buildexp *= "get_prop(g, $vn, $v, :weight) * bloxlist[$vn].nmm.x+"
        end
        # push!(eqs, get_prop(g, dst(edge), :blox).jcn ~ get_prop(g, edge, :weight) * get_prop(g, src(edge), :blox).x)
        @show Meta.parse(buildexp)
        push!(eqs, bloxlist[v].nmm.jcn ~ Meta.parse(buildexp[1:end-1]))
    end
    ODESystem(eqs, systems=bloxlist; name=name)
end


nr = 2
regions = []
connex = Num[]
for ii = 1:nr
    @named nmm = linearneuralmass()
    @named hemo = hemodynamicsMTK(0.0, 0.0)
    eqs = nmm.x ~ hemo.x
    region = ODESystem(eqs, systems=[nmm, hemo], name=Symbol("r$ii"))

    push!(connex, region.nmm.x)
    push!(regions, region)
end

A = [1 0.5;
    0 -1]

@parameters adj[1:length(A)] = vec(A) 
@named model = linearconnectionssymbolic(sys=regions, adj_matrix=adj, connector=connex)
model = structural_simplify(model)

jac = calculate_jacobian(model)

foo = substitute(jac, Dict([r.hemo.τ => 2.0 for r in regions]))

[r.hemo.τ => 2.0 for r in regions]

function connectblox(bloxlist, A)
    g = MetaDiGraph()
    for blox in bloxlist
        add_vertex!(g, Dict(:blox => blox))
    end

    nonzeroelems = findall(x -> x != 0, A)
    for idx in nonzeroelems
        add_edge!(g, Tuple(idx)..., :weight, A[idx])
    end
    return g
end

function odefromgraph(g::MetaDiGraph; name)
    bloxlist = [get_prop(g, v, :blox) for v in vertices(g)]
    eqs = []
    buildexp = ""
    for v in vertices(g)
        # build right hand side of equation sum over all input neighbors
        for vn in inneighbors(g, v)
            buildexp *= "get_prop(g, $vn, $v, :weight) * bloxlist[$vn].x+"
        end
        push!(eqs, bloxlist[v].jcn ~ eval(Meta.parse(buildexp[1:end-1])))
    end
    ODESystem(eqs, systems=bloxlist; name=name)
end


function linearconnections(;name, sys=sys, adj_matrix=adj_matrix, connector=connector)
    adj = adj_matrix .* connector
    eqs = []
    for region_num in 1:length(sys)
       push!(eqs, sys[region_num].nmm.jcn ~ sum(adj[:, region_num]))
    end
    return ODESystem(eqs, name=name, systems=sys)
end


nr = 2 # number of regions
regions = []
connex = Num[]
# create list of linear neural mass models
for ii = 1:nr
    region = linearneuralmass(name=Symbol("r$ii"))
    push!(connex, region.x)
    push!(regions, region)
end
# connect linear neural mass models
A = [1 0.5;
    0 -1]

@parameters adj[1:length(A)] = vec(A)
@named linsymb = linearconnectionssymbolic(sys=regions, adj_matrix=adj, connector=connex)
@named linnormal = linearconnections(sys=regions, adj_matrix=A, connector=connex)
foo = structural_simplify(linsymb)
timespan = (0, 10)
u0 = zeros(10)

prob = ODEProblem(foo, [0.0], timespan, [2.0])
probnew = remake(prob; p=[:a => 1.0])

g = connectblox(regions, A)

function linearneuralmass(;name)
    states = @variables x(t) jcn(t)
    param = @parameters α
    eqs = D(x) ~ α * jcn
    return ODESystem(eqs, t, states, param; name=name)
end


@named testblox = linearneuralmass()
@named foo = ODESystem([testblox.jcn ~ 2.0], systems=[testblox])
foo = structural_simplify(foo)
graph = connectblox(regions, A)

adj = A .* connector
eqs = []
for region_num in 1:length(regions)
    push!(eqs, regions[region_num].nmm₊jcn ~ sum(adj[:, region_num]))
end
@named linsys = ODESystem(eqs, systems=regions)



@named hemosys = hemodynamicsMTK(region, 0, 0)

generate_jacobian(hemosys, dvs = states(hemosys), ps = parameters(hemosys), expression = Val{true})
jac = calculate_jacobian(hemosys)


function hemodynamics!(dy, y, p, t)
    """
    x     - neural activity
    Components of y are:
    y[1] - vascular signal: s
    y[2] - rCBF: ln(f)
    y[3] - venous volume: ln(ν)
    y[4] - deoxyhemoglobin (dHb): ln(q)
    
    decay, transit - free parameters, set to 0 for standard parameters.
    
    This function implements the hymodynamics model (balloon model and neurovascular state eq.) described in: 
    
    Stephan KE, Weiskopf N, Drysdale PM, Robinson PA, Friston KJ (2007)
    Comparing hemodynamic models with DCM. NeuroImage 38: 387-401.
    
    adapted from spm_fx_fmri.m in SPM12 by Karl Friston & Klaas Enno Stephan
    """
    x = p[1]
    decay = p[2]
    transit = p[3]
    #= hemodynamic parameters
        H(1) - signal decay                                   d(ds/dt)/ds)
        H(2) - autoregulation                                 d(ds/dt)/df)
        H(3) - transit time                                   (t0)
        H(4) - exponent for Fout(v)                           (alpha)
        H(5) - resting oxygen extraction                      (E0)
    =#
    H = [0.64, 0.32, 2.00, 0.32, 0.4]

    # exponentiation of hemodynamic state variables
    y[2:4] = exp.(y[2:4])

    # signal decay
    κ = H[1]*exp(decay)

    # transit time
    τ = H[3]*exp(transit)

    # Fout = f(v) - outflow
    fv = y[3]^(H[4]^-1)

    # e = f(f) - oxygen extraction
    ff = (1.0 - (1.0 - H[5])^(y[2]^-1))/H[5]

    # implement differential state equation f = dx/dt (hemodynamic)

    dy[1] = x - κ*y[1] - H[2]*(y[2] - 1)       # Corresponds to eq (9)
    dy[2] = y[1]/y[2]                          # Corresponds to eq (10), note the added logarithm (see doc string)
    dy[3] = (y[2] - fv)/(τ*y[3])               # Corresponds to eq (8), note the added logarithm (see doc string)
    dy[4] = (ff*y[2] - fv*y[4]/y[3])/(τ*y[4])  # Corresponds to eq (8), note the added logarithm (see doc string)

    J = zeros(eltype(y), 4, 4)
    J[1, :] = [-κ,
               -H[2]*y[2],
               0,
               0]

    J[2, :] = [y[2]^-1,
               -y[1]/y[2],
               0,
               0]

    J[3, :] = [0,
               y[2]/(τ*y[3]),
               -y[3]^(H[4]^-1 - 1)/(τ*H[4]) - (y[2] - y[3]^(H[4]^-1))/(τ*y[3]),
               0]

    J[4, :] = [0,
               (y[2] + log(1 - H[5])*(1 - H[5])^(y[2]^-1) - y[2]*(1 - H[5])^(y[2]^-1))/(τ*x[4]*H[5]),
               (y[3]^(H[4]^-1 - 1)*(1 - H[4]^-1))/τ,
               (y[2]/y[4])*((1 - H[5])^(y[2]^-1) - 1)/(τ*H[5])]
end




u0 = [0.0;0.0;0.0;0.0]
tspan = (0.0,10.0)
p = [0.1, 0.0, 0.0]
prob = ODEProblem(hemodynamics!, u0, tspan, p, solver = AutoVern7(Rodas4()))
sol = solve(prob)

plot(sol)



using ModelingToolkit

@parameters t
D = Differential(t)

states = @variables x(t) jcn(t)
param = @parameters α
eqs = D(x) ~ α * jcn
@named r1 = ODESystem(eqs, t, states, param)
@named r2 = ODESystem(eqs, t, states, param)

A = [1 0.5;
    0 -1]

@parameters adj[1:length(A)] = vec(A)

@named foo = symboliclinearconnections(sys=[r1, r2], adj_matrix=adj, connector=[r1.x, r2.x])
calculate_jacobian(structural_simplify(foo))
eqs = [r1.jcn ~ adj[1] * r1.x + adj[2] * r2.x, 
       r2.jcn ~ adj[3] * r1.x + adj[4] * r2.x]

@named linsys = ODESystem(eqs, systems=[r1, r2])
linsys = structural_simplify(linsys)
jac = calculate_jacobian(linsys)

substitute(jac, Dict(adj[1,2] => 2.0, linsys.r1.α => 0.1))

function symboliclinearconnections(;name, sys=sys, adj_matrix=adj_matrix, connector=connector)
    eqs = []
    nr = length(sys)
    for i in 1:nr
       push!(eqs, sys[i].jcn ~ sum(adj_matrix[(1+(i-1)*nr):nr*i] .* connector))
    end
    return ODESystem(eqs, name=name, systems=sys)
end



nr = 2 # number of regions
regions = []
connex = Num[]
# create list of linear neural mass models
for ii = 1:nr
    region = linearneuralmass(name=Symbol("r$ii"))
    push!(connex, region.x)
    push!(regions, region)
end

# connect linear neural mass models
A = [1 0.5;
    0 -1]

@parameters adj[1:length(A)] = vec(A)
@named linsymb = linearconnectionssymbolic(sys=regions, adj_matrix=adj, connector=connex)
model = structural_simplify(linsymb)
jac = calculate_jacobian(model)


using ModelingToolkit

@parameters t
D = Differential(t)

function linearneuralmass(;name)
    states = @variables x(t) jcn(t)
    param = @parameters α
    eqs = D(x) ~ α * jcn
    return ODESystem(eqs, t, states, param; name=name)
end

nr = 2 # number of regions
regions = []
connex = Num[]
# create list of linear neural mass models
for i = 1:nr
    region = linearneuralmass(name=Symbol("r$i"))
    push!(connex, region.x)
    push!(regions, region)
end

# connect two systems with the following adjacency matrix:
A = [1 0.5;
    0 -1]
@parameters adj[1:length(A)] = vec(A)

eqs = []
for i in 1:nr
   push!(eqs, regions[i].jcn ~ sum(adj[(1+(i-1)*nr):nr*i] .* connex))
end
@named linsys = ODESystem(eqs, systems=regions)
linsys = structural_simplify(linsys)
jac = calculate_jacobian(linsys)

# now comes the relevant part:
substitute(jac, Dict(adj[1] => 2.0, regions[1].α => 0.1))