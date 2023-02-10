##############################################################################################
# A consumption-saving problem 
# v4 - Use the lifetime reward method
##############################################################################################

##############################################################################################
# 0. Preliminaries
##############################################################################################

# Activate and set up project 
using DrWatson
@quickactivate "p03_ConsumptionSaving"

# Specifications
const prec = 1.0f0                              # Set floating point precision

# Include the requisite objects
include(srcdir("p03_Packages.jl"))              # Julia packages
include(srcdir("p03_Functions_Lifetime.jl"))    # Project-specific functions 

##############################################################################################
# 1. Set up
##############################################################################################

# Set model parameters
const β = oftype(prec,0.9)        # Subjective discount factor
const γ = oftype(prec,2.0)        # Coefficient of relative risk aversion
const σ_r = oftype(prec,0.001)    # Volatility of return shock
const ρ_r = oftype(prec,0.20)     # Persistence of return shock
const σ_p = oftype(prec,0.0001)   # Volatility of persistent income shock
const ρ_p = oftype(prec,0.999)    # Persistence of persistent income shock
const σ_q = oftype(prec,0.001)    # Volatility of transitory income shock
const ρ_q = oftype(prec,0.9)      # Persistence of transitory income shock
const σ_δ = oftype(prec,0.001)    # Volatility of discount factor shock
const ρ_δ = oftype(prec,0.2)      # Persistence of discount factor shock
const rbar = oftype(prec,1.04)    # Average gross return

# Volatilities in the ergodic distribution
const σ_e_r = σ_r/sqrt(one(prec) - ρ_r^oftype(prec,2.0))        # Return shock
const σ_e_p = σ_p/sqrt(one(prec) - ρ_p^oftype(prec,2.0))        # Persistent income shock
const σ_e_q = σ_q/sqrt(one(prec) - ρ_q^oftype(prec,2.0))        # Transitory income shock
const σ_e_δ = σ_δ/sqrt(one(prec) - ρ_δ^oftype(prec,2.0))        # Discount factor shock

# Some hyperparameters
const wmin = oftype(prec,0.1)                         # Lower bound of cash-on-hand grid
const wmax = oftype(prec,4.0)                         # Upper bound of cash-on-hand grid
const n = 128                                         # Length of simulated training samples in each epoch
const Tmax = 30                                       # Time horizon for finite-horizon approximation of infinite-horizon value

# Set up neural network
numInputs = 5
numUnits = 32
numOutputs = 1
function NeuralNetwork()
    return Chain(
            Dense(numInputs, numUnits,relu; init = Flux.kaiming_uniform()),
            Dense(numUnits, numUnits,relu; init = Flux.kaiming_uniform()),
            Dense(numUnits, numUnits,relu; init = Flux.kaiming_uniform()),
            Dense(numUnits,numOutputs)
            )
end
m    = NeuralNetwork()
opt = Flux.setup(Adam(0.001, (0.9, 0.999)), m)

##############################################################################################
# 2. Solution
##############################################################################################

# Time solution
epochs = 50000
nn_train!(opt,m,1)
@time nn_train!(opt,m,epochs)

# Plot solution
wvec = collect(LinRange(wmin, wmax, 100))
ζvec = hh_policy_fn(m,zeros(typeof(prec),100,1),zeros(typeof(prec),100,1),zeros(typeof(prec),100,1),zeros(typeof(prec),100,1),wvec)
plot(wvec, wvec, linestyle=:dash)
plot!(wvec, wvec.*ζvec, linestyle=:solid)

