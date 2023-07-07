using LinearAlgebra
using MKL
using FFTW
using ToeplitzMatrices
using MAT
using ExponentialUtilities
using ForwardDiff
using OrderedCollections

### a few packages relevant for speed tests and profiling ###
using Serialization
using StatProfilerHTML


# simple dispatch for vec to deal with 1xN matrices
function Base.vec(x::T) where (T <: Real)
    return x*ones(1)
end

include("../src/hemodynamic_response.jl")     # hemodynamic and BOLD signal model
include("../src/mar.jl")                      # multivariate auto-regressive model functions
include("../src/VariationalBayes_AD.jl")

function wrapperfunction(vars, iter)
    y = vars["data"];
    dt = vars["dt"];
    nd = size(y, 2);
    freqs = vec(vars["Hz"]);
    p = 8;                               # order of MAR, it is hard-coded in SPM12 with this value. We will just use the same for now.
    mar = mar_ml(y, p);                  # compute MAR from time series y and model order p
    y_csd = mar2csd(mar, freqs, dt^-1);  # compute cross spectral densities from MAR parameters at specific frequencies freqs, dt^-1 is sampling rate of data

    ### Define priors and initial conditions ###
    x = vars["x"];                       # initial condition of dynamic variabls
    θΣ = vars["pC"];                     # prior covariance of parameter values 
    # depending on the definition of the priors (note that we take it from the SPM12 code), some dimensions are set to 0 and thus are not changed.
    # Extract these dimensions and remove them from the remaining computation. I find this a bit odd and further thoughts would be necessary to understand
    # to what extend this is legitimate. 
    idx = findall(x -> x != 0, θΣ);
    V = zeros(size(θΣ, 1), length(idx));
    order = sortperm(θΣ[idx], rev=true);
    idx = idx[order];
    for i = 1:length(idx)
        V[idx[i][1], i] = 1.0
    end
    θΣ = V'*θΣ*V;       # reduce dimension by removing columns and rows that are all 0

    Πλ_p = vars["ihC"];                  # prior precision matrix of hyperparameters
    if typeof(Πλ_p) <: Number            # typically Πλ_p is a matrix, however, if only one hyperparameter is used it will turn out to be a scalar -> transform that to matrix
        Πλ_p *= ones(1, 1)
    end

    Q = csd_Q(y_csd);                 # compute prior of Q, the precision (of the data) components. See Friston etal. 2007 Appendix A

    priors = Dict(:μ => OrderedDict{Any, Any}(
                            :A => vars["pE"]["A"],      # prior mean of connectivity matrix
                            :C => ones(Float64, nd),    # C as in equation 3. NB: whatever C is defined to be here, it will be replaced in csd_approx. Another strange thing of SPM12...
                            :lnτ => zeros(Float64, nd), # hemodynamic transit parameter
                            :lnκ => 0.0,                # hemodynamic decay time
                            :lnϵ => 0.0,                # BOLD signal ratio between intra- and extravascular signal
                            :lnα => [0.0, 0.0],         # intrinsic fluctuations, ln(α) as in equation 2 of Friston et al. 2014
                            :lnβ => [0.0, 0.0],         # global observation noise, ln(β) as above
                            :lnγ => zeros(Float64, nd)  # region specific observation noise
                        ),
                    :Σ => Dict(
                    :Πθ_pr => inv(θΣ),           # prior model parameter precision
                    :Πλ_pr => Πλ_p,              # prior metaparameter precision
                    :μλ_pr => vec(vars["hE"]),   # prior metaparameter mean
                    :Q => Q                      # decomposition of model parameter covariance
                    )
                    );


    ### Compute the DCM ###
    results = variationalbayes(x, y_csd, freqs, V, p, priors, iter);
    return results
end

function wrapperfunction_MTK(vars, iter)
    y = vars["data"];
    nd = size(y, 2);
    dt = vars["dt"];
    freqs = vec(vars["Hz"]);
    p = 8;                               # order of MAR, it is hard-coded in SPM12 with this value. We will just use the same for now.
    mar = mar_ml(y, p);                  # compute MAR from time series y and model order p
    y_csd = mar2csd(mar, freqs, dt^-1);  # compute cross spectral densities from MAR parameters at specific frequencies freqs, dt^-1 is sampling rate of data
    ### Define priors and initial conditions ###
    x = vars["x"];                       # initial condition of dynamic variabls
    Adj = vars["pE"]["A"];                 # initial values of connectivity matrix
    θΣ = vars["pC"];                     # prior covariance of parameter values 
    Πλ_p = vars["ihC"];                  # prior precision matrix of hyperparameters
    if typeof(Πλ_p) <: Number            # typically Πλ_p is a matrix, however, if only one hyperparameter is used it will turn out to be a scalar -> transform that to matrix
        Πλ_p *= ones(1,1)
    end
    
    ########## assemble the model ##########
    
    regions = []
    connex = Num[]
    @parameters κ=0.0
    for ii = 1:nd
        @named nmm = linearneuralmass()
        @named hemo = hemodynamicsMTK(;κ=κ, τ=0.0)
        eqs = [nmm.x ~ hemo.x]
        region = ODESystem(eqs, systems=[nmm, hemo], name=Symbol("r$ii"))
    
        push!(connex, region.nmm.x)
        push!(regions, region)
    end
    diagelem = [(i-1)*nd+i for i in 1:nd]
    @parameters A[1:length(Adj)] = vec(Adj)
    
    @named model = linearconnectionssymbolic(sys=regions, adj_matrix=[i in diagelem ? -exp(a)/2 : a for (i, a) in enumerate(A)], connector=connex)
    f = structural_simplify(model)
    jac_f = calculate_jacobian(f)
    jac_f = substitute(jac_f, Dict([p for p in parameters(f) if occursin("κ", string(p))] .=> κ))
    
    @named bold = boldsignal()
    grad_g = calculate_jacobian(bold)[2:3]
    
    # define values of states
    all_s = states(f)
    
    sts = Dict{typeof(all_s[1]), eltype(x)}()
    for i in 1:nd
        for (j, s) in enumerate(all_s[occursin.("r$i", string.(all_s))])
            sts[s] = x[i, j]
        end
    end
    
    bolds = states(bold)
    statesubs = merge.([Dict(bolds[2] => s) for s in all_s if occursin(string(bolds[2]), string(s))],
                       [Dict(bolds[3] => s) for s in all_s if occursin(string(bolds[3]), string(s))])
    
    grad_g_full = Num.(zeros(nd, length(all_s)))
    for (i, s) in enumerate(all_s)
        dim = parse(Int64, string(s)[2])
        if occursin.(string(bolds[2]), string(s))
            grad_g_full[dim, i] = substitute(grad_g[1], statesubs[dim])
        elseif occursin.(string(bolds[3]), string(s))
            grad_g_full[dim, i] = substitute(grad_g[2], statesubs[dim])
        end
    end
    
    
    modelparam = OrderedDict{Any, Any}()
    for par in parameters(f)
        while Symbolics.getdefaultval(par) isa Num
            par = Symbolics.getdefaultval(par)
        end
        modelparam[par] = Symbolics.getdefaultval(par)
    end
    # Noise parameter mean
    modelparam[:lnα] = [0.0, 0.0];           # intrinsic fluctuations, ln(α) as in equation 2 of Friston et al. 2014 
    modelparam[:lnβ] = [0.0, 0.0];           # global observation noise, ln(β) as above
    modelparam[:lnγ] = zeros(Float64, nd);   # region specific observation noise
    modelparam[:C] = ones(Float64, nd);     # C as in equation 3. NB: whatever C is defined to be here, it will be replaced in csd_approx. Another strange thing of SPM12...
    
    for par in parameters(bold)
        modelparam[par] = Symbolics.getdefaultval(par)
    end
    
    idx_A = findall(occursin.("A[", string.(jac_f)))
    pnames = [k for k in keys(modelparam)]
    derivatives = Dict(:∂f => eval(Symbolics.build_function(substitute(jac_f, sts), pnames[1:nd^2+nd+1]...)[1]),
                       :∂g => eval(Symbolics.build_function(substitute(grad_g_full, sts), pnames[end])[1]))
    # derivatives = Dict(:∂f => substitute(jac_f, sts), :∂g => substitute(grad_g_full, sts))
    
    # define prior variances
    paramvariance = copy(modelparam)
    paramvariance[:C] = zeros(Float64, nd);
    paramvariance[:lnγ] = ones(Float64, nd)./64.0;
    paramvariance[:lnα] = ones(Float64, length(modelparam[:lnα]))./64.0; 
    paramvariance[:lnβ] = ones(Float64, length(modelparam[:lnβ]))./64.0;
    for (k, v) in paramvariance
        if occursin("A[", string(k))
            paramvariance[k] = θΣ[1,1]
        elseif occursin("κ", string(k))
            paramvariance[k] = ones(length(v))./256.0;
        elseif occursin("ϵ", string(k))
            paramvariance[k] = 1/256.0;
        elseif occursin("τ", string(k))
            paramvariance[k] = 1/256.0;
        end
    end
    θΣ = diagm(vecparam(paramvariance));
    
    # depending on the definition of the priors (note that we take it from the SPM12 code), some dimensions are set to 0 and thus are not changed.
    # Extract these dimensions and remove them from the remaining computation. I find this a bit odd and further thoughts would be necessary to understand
    # to what extend this is a the most reasonable approach. 
    idx = findall(x -> x != 0, θΣ);
    V = zeros(size(θΣ, 1), length(idx));
    order = sortperm(θΣ[idx], rev=true);
    idx = idx[order];
    for i = 1:length(idx)
        V[idx[i][1], i] = 1.0
    end
    θΣ = V'*θΣ*V;       # reduce dimension by removing columns and rows that are all 0
    
    Q = csd_Q(y_csd);                 # compute prior of Q, the precision (of the data) components. See Friston etal. 2007 Appendix A
    
    priors = Dict(:μ => modelparam,
                  :Σ => Dict(
                             :Πθ_pr => inv(θΣ),           # prior model parameter precision
                             :Πλ_pr => Πλ_p,              # prior metaparameter precision
                             :μλ_pr => vec(vars["hE"]),   # prior metaparameter mean
                             :Q => Q                      # decomposition of model parameter covariance
                            )
                 );
    ### Compute the DCM ###
    results = variationalbayes(idx_A, y_csd, derivatives, freqs, V, p, priors, iter)
    return results
end

# speed comparison between different DCM implementations
for n in 2:3
    vals = matread("speedandaccuracy/matlab0.01_" * string(n) *"regions.mat");
    include("../src/VariationalBayes_spm12.jl")      # this can be switched between _spm12 and _AD version. There is also a separate ADVI version in VariationalBayes_ADVI.jl
    wrapperfunction(vals, 1)
    t_juliaSPM = @elapsed res_spm = wrapperfunction(vals, 128)
    include("../src/VariationalBayes_AD.jl")      # this can be switched between _spm12 and _AD version. There is also a separate ADVI version in VariationalBayes_ADVI.jl
    wrapperfunction(vals, 1)
    t_juliaAD = @elapsed res_AD = wrapperfunction(vals, 128)
    wrapperfunction_MTK(vals, 1)
    t_juliaMTK = @elapsed res_mtk = wrapperfunction_MTK(vals, 128)
    @show t_juliaAD, t_juliaSPM, t_juliaMTK

    matwrite("test" * string(n) * ".mat", Dict(
        "t_mat" => vals["matcomptime"],
        "F_mat" => vals["F"],
        "t_jad" => t_juliaAD,
        "F_jad" => res_AD.F,
        "t_jspm" => t_juliaSPM,
        "F_jspm" => res_spm.F,
        "t_mtk" => t_juliaMTK,
        "F_mtk" => res_mtk.F,
        "iter_spm" => res_spm.iter,
        "iter_ad" => res_AD.iter,
        "iter_mtk" => res_mtk.iter
    ); compress = true)    
end


### Profiling ###
include("../src/VariationalBayes_AD.jl")

n = 3
vars = matread("speedandaccuracy/matlab0.01_" * string(n) *"regions.mat");
y = vars["data"];
dt = vars["dt"];
freqs = vec(vars["Hz"]);
p = 8;                               # order of MAR, it is hard-coded in SPM12 with this value. We will just use the same for now.
mar = mar_ml(y, p);                  # compute MAR from time series y and model order p
y_csd = mar2csd(mar, freqs, dt^-1);  # compute cross spectral densities from MAR parameters at specific frequencies freqs, dt^-1 is sampling rate of data

### Define priors and initial conditions ###
x = vars["x"];                       # initial condition of dynamic variabls
A = vars["pE"]["A"];                 # initial values of connectivity matrix
θΣ = vars["pC"];                     # prior covariance of parameter values 
λμ = vec(vars["hE"]);                # prior mean of hyperparameters
Πλ_p = vars["ihC"];                  # prior precision matrix of hyperparameters
if typeof(Πλ_p) <: Number            # typically Πλ_p is a matrix, however, if only one hyperparameter is used it will turn out to be a scalar -> transform that to matrix
    Πλ_p *= ones(1,1)
end

# depending on the definition of the priors (note that we take it from the SPM12 code), some dimensions are set to 0 and thus are not changed.
# Extract these dimensions and remove them from the remaining computation. I find this a bit odd and further thoughts would be necessary to understand
# to what extend this is legitimate. 
idx = findall(x -> x != 0, θΣ);
V = zeros(size(θΣ, 1), length(idx));
order = sortperm(θΣ[idx], rev=true);
idx = idx[order];
for i = 1:length(idx)
    V[idx[i][1], i] = 1.0
end
θΣ = V'*θΣ*V;       # reduce dimension by removing columns and rows that are all 0
Πθ_p = inv(θΣ);

# define a few more initial values of parameters of the model
dim = size(A, 1);
C = zeros(Float64, dim);          # C as in equation 3. NB: whatever C is defined to be here, it will be replaced in csd_approx. Another little strange thing of SPM12...
lnα = [0.0, 0.0];                 # ln(α) as in equation 2 
lnβ = [0.0, 0.0];                 # ln(β) as in equation 2
lnγ = zeros(Float64, dim);        # region specific observation noise parameter
lnϵ = 0.0;                        # BOLD signal parameter
lndecay = 0.0;                    # hemodynamic parameter
lntransit = zeros(Float64, dim);  # hemodynamic parameters
param = [p; reshape(A, dim^2); C; lntransit; lndecay; lnϵ; lnα[1]; lnβ[1]; lnα[2]; lnβ[2]; lnγ;];
# Strange α and β sorting? yes. This is to be consistent with the SPM12 code while keeping nomenclature consistent with the spectral DCM paper
Q = csd_Q(y_csd);                 # compute prior of Q, the precision (of the data) components. See Friston etal. 2007 Appendix A
priors = [Πθ_p, Πλ_p, λμ, Q];
variationalbayes(x, y_csd, freqs, V, param, priors, 26)

@profilehtml results = variationalbayes(x, y_csd, freqs, V, param, priors, 26)
