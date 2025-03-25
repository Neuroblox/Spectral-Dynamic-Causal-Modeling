"""
fmri.jl 

Models of fMRI signals.

BalloonModel : computes hemodynamic responses and bold signal
"""


"""
    BalloonModel(;name, namespace=nothing, lnκ=0.0, lnτ=0.0)

    Create a balloon model blox which computes the hemodynamic responses to some underlying neuronal activity
    The formal definition of this blox is:
    """
        # ```math
        #     \\frac{ds}{dt} = \\text{jcn} - \\kappa s - \\gamma (u-1)
        #     \\frac{du}{dt} = s
        #     \\frac{d\\nu}{dt} = u - v^{1/\\alpha}
        #     \\frac{dq}{dt} = u E(u, E_0)/E_0 - v^{1/\\alpha} q/v
        # ```
    """
    where ``jcn`` is any input to the blox (represents the neuronal activity).

    Includes also the bold measurement function:
     """
        # ```math
        #     \\lambda(\\nu, q) = V_0 \\left[ k_1 (1-q) + k_2 \\left( 1 - \\frac{q}{v} \\right) + k_3 (1-v)\\right]
        # ```
    """
    
Arguments:
- `name`: Name given to `ODESystem` object within the blox.
- `namespace`: Additional namespace above `name` if needed for inheritance.
- `lnκ`: logarithmic prefactor to signal decay H[1], set to 0 for standard parameter value.
- `lnτ`: logarithmic prefactor to transit time H[3], set to 0 for standard parameter value.
- `lnϵ`: logarithm of ratio of intra- to extra-vascular signal

NB: the prefix ln of the variables u, ν, q as well as the parameters κ, τ denotes their transformation into logarithmic space
to enforce their positivity. This transformation is considered in the derivates of the model equations below. 

Citations:
1. Stephan K E, Weiskopf N, Drysdale P M, Robinson P A, and Friston K J. Comparing Hemodynamic Models with DCM. NeuroImage 38, no. 3 (2007): 387–401. doi: 10.1016/j.neuroimage.2007.07.040
2. Hofmann D, Chesebro A G, Rackauckas C, Mujica-Parodi L R, Friston K J, Edelman A, and Strey H H. Leveraging Julia's Automated Differentiation and Symbolic Computation to Increase Spectral DCM Flexibility and Speed, 2023. doi: 10.1101/2023.10.27.564407
"""
struct BalloonModel <: ObserverBlox
    params
    system
    namespace

    function BalloonModel(;name, namespace=nothing, lnκ=0.0, lnτ=0.0, lnϵ=0.0)
        #= hemodynamic parameters
            H[1] - signal decay                                   d(ds/dt)/ds)
            H[2] - autoregulation                                 d(ds/dt)/df)
            H[3] - transit time                                   (t0)
            H[4] - exponent for Fout(v)                           (alpha)
            H[5] - resting state oxygen extraction                (E0)
        =#
        H = [0.64, 0.32, 2.00, 0.32, 0.4]

        #=
            NB: Biophysical constants for 1.5T scanners
            
            B[1]: TE, time to echo
            B[2]: V0, resting venous volume (%)
            B[3]: r0, slope of intravascular relaxation rate R_iv as a function of oxygen saturation S:  R_iv = r0*[(1 - S)-(1 - S0)] (Hz)
            B[4]: nu0, frequency offset at the outer surface of magnetized vessels (Hz)
            B[5]: E0, resting oxygen extraction fraction
        =#

        B = [0.04, 4, 25, 40.3, 0.4]
        k1 = 4.3*B[1]*B[4]*B[5]                         # coefficients in BOLD signal model

        p = paramscoping(lnκ=lnκ, lnτ=lnτ, lnϵ=lnϵ)     # finally compile all parameters
        lnκ, lnτ, lnϵ = p                               # assign the modified parameters

        sts = @variables s(t)=0.0 lnu(t)=0.0 lnν(t)=0.0 lnq(t)=0.0 bold(t)=0.0 [irreducible=true, output=true, description="measurement"] jcn(t) [input=true]

        eqs = [
            D(s)   ~ jcn - H[1]*exp(lnκ)*s - H[2]*(exp(lnu) - 1),
            D(lnu) ~ s / exp(lnu),
            D(lnν) ~ (exp(lnu) - exp(lnν)^(H[4]^-1)) / (H[3]*exp(lnτ)*exp(lnν)),
            D(lnq) ~ (exp(lnu)/exp(lnq)*((1 - (1 - H[5])^(exp(lnu)^-1))/H[5]) - exp(lnν)^(H[4]^-1 - 1))/(H[3]*exp(lnτ)),
            bold   ~ B[2]*(k1 - k1*exp(lnq) + exp(lnϵ)*B[3]*B[5]*B[1] - exp(lnϵ)*B[3]*B[5]*B[1]*exp(lnq)/exp(lnν) + 1-exp(lnϵ) - (1-exp(lnϵ))*exp(lnν))
        ]
        sys = System(eqs, t; name=name)
        new(p, sys, namespace)
    end
end

# Lead field function for LFPs
struct LeadField <: ObserverBlox
    params
    system
    namespace

    function LeadField(;name, namespace=nothing, L=1.0)
        p = paramscoping(L=L)
        L, = p

        sts = @variables lfp(t)=0.0 [irreducible=true, output=true, description="measurement"] jcn(t)=1.0 [input=true]

        eqs = [
            lfp ~ L * jcn
        ]

        sys = System(eqs, t; name=name)
        new(p, sys, namespace)
    end
end