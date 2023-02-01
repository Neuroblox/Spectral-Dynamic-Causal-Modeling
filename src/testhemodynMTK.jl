using DifferentialEquations
using ModelingToolkit
using Plots

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

    # J = zeros(eltype(y), 4, 4)
    # J[1, :] = -κ, -H[2], 0, 0
    # J[2, :] = y[2]^-1, -y[1]/y[2]^2
    # J[d+1:2d,1:d] = diagm( x[:,2].^-1)
    # J[d+1:2d,d+1:2d] = diagm(-x[:,1]./x[:,2])
    # J[2d+1:3d,d+1:2d] = diagm(x[:,2]./(τ.*x[:,3]))
    # J[2d+1:3d,2d+1:3d] = diagm(-x[:,3].^(H[4]^-1 - 1)./(τ*H[4]) - (x[:,3].^-1 .*(x[:,2] - x[:,3].^(H[4]^-1)))./τ)
    # J[3d+1:4d,d+1:2d] = diagm((x[:,2] .+ log(1 - H[5])*(1 - H[5]).^(x[:,2].^-1) .- x[:,2].*(1 - H[5]).^(x[:,2].^-1))./(τ.*x[:,4]*H[5]))
    # J[3d+1:4d,2d+1:3d] = diagm((x[:,3].^(H[4]^-1 - 1)*(H[4] - 1))./(τ*H[4]))
    # J[3d+1:4d,3d+1:4d] = diagm((x[:,2]./x[:,4]).*((1 - H[5]).^(x[:,2].^-1) .- 1)./(τ*H[5]))
    # J[1:d,1:d] = Matrix(-κ*I, d, d)   # TODO: make it work when κ/decay is a vector. Only solution if-clause? diagm doesn't allow scalars, [κ] would work in that case
    # J[1:d,d+1:2d] = diagm(-H[2]*x[:,2])
    # J[d+1:2d,1:d] = diagm( x[:,2].^-1)
    # J[d+1:2d,d+1:2d] = diagm(-x[:,1]./x[:,2])
    # J[2d+1:3d,d+1:2d] = diagm(x[:,2]./(τ.*x[:,3]))
    # J[2d+1:3d,2d+1:3d] = diagm(-x[:,3].^(H[4]^-1 - 1)./(τ*H[4]) - (x[:,3].^-1 .*(x[:,2] - x[:,3].^(H[4]^-1)))./τ)
    # J[3d+1:4d,d+1:2d] = diagm((x[:,2] .+ log(1 - H[5])*(1 - H[5]).^(x[:,2].^-1) .- x[:,2].*(1 - H[5]).^(x[:,2].^-1))./(τ.*x[:,4]*H[5]))
    # J[3d+1:4d,2d+1:3d] = diagm((x[:,3].^(H[4]^-1 - 1)*(H[4] - 1))./(τ*H[4]))
    # J[3d+1:4d,3d+1:4d] = diagm((x[:,2]./x[:,4]).*((1 - H[5]).^(x[:,2].^-1) .- 1)./(τ*H[5]))
end

u0 = [0.0;0.0;0.0;0.0]
tspan = (0.0,10.0)
p = [0.1, 0.0, 0.0]
prob = ODEProblem(hemodynamics!, u0, tspan, p, solver = AutoVern7(Rodas4()))
sol = solve(prob)

plot(sol)