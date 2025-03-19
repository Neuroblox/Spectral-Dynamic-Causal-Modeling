# Spectral-DCM

Implementation of spectral dynamic mode decomposition (DCM) in Julia.
Code partially translated from the DCM implementation in MATLAB as part of SPM12 (https://www.fil.ion.ucl.ac.uk/spm/software/spm12/)

# Define notational correspondence between SPM and this code:

the following two precision matrices will not be updated by the code,
they belong to the assumed prior distribution p (fixed, but what if it isn't
the ground truth?)
ipC = Πθ_pr   # precision matrix of prior of parameters p(θ)
ihC = Πλ_pr   # precision matrix of prior of hyperparameters p(λ)

Variational distribution parameters:
pE, Ep = μθ_pr, μθ   # prior expectation of parameters (q(θ))
pC, Cp = θΣ, Σθ   # prior covariance of parameters (q(θ))
hE, Eh = μλ_pr, μλ   # prior expectation of hyperparameters (q(λ))
hC, Ch = λΣ, Σλ   # prior covariance of hyperparameters (q(λ))

Σ, iΣ  # data covariance matrix (likelihood), and its inverse (precision of likelihood - use Π only for those precisions that don't change)
Q      # components of iΣ; definition: iΣ = sum(exp(λ)*Q)

