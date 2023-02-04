##############################################################################################
# A consumption-saving problem 
# v1 - First attempt to replicate the Python code from QuantEcon
##############################################################################################

##############################################################################################
# 0. Preliminaries
##############################################################################################

# Activate and set up project 
using DrWatson
@quickactivate "p03_ConsumptionSaving"

# Include the requisite objects
include(srcdir("p03_Packages.jl"))          # Julia packages
include(srcdir("p03_Functions.jl"))         # Project-specific functions 

##############################################################################################
# 1. Set up
##############################################################################################

# Set model parameters
Params = @with_kw (
                    # Fundamental model parameters
                    β = 0.9,        # Subjective discount factor
                    γ = 2.0,        # Coefficient of relative risk aversion
                    σ_r = 0.001,    # Volatility of return shock
                    ρ_r = 0.20,     # Persistence of return shock
                    σ_p = 0.0001,   # Volatility of persistent income shock
                    ρ_p = 0.999,    # Persistence of persistent income shock
                    σ_q = 0.001,    # Volatility of transitory income shock
                    ρ_q = 0.9,      # Persistence of transitory income shock
                    σ_δ = 0.001,    # Volatility of discount factor shock
                    ρ_δ = 0.2,      # Persistence of discount factor shock
                    rbar = 1.04,    # Average gross return

                    # Volatilities in the ergodic distribution
                    σ_e_r = σ_r/sqrt(1 - ρ_r^2),        # Return shock
                    σ_e_p = σ_p/sqrt(1 - ρ_p^2),        # Persistent income shock
                    σ_e_q = σ_q/sqrt(1 - ρ_q^2),        # Transitory income shock
                    σ_e_δ = σ_δ/sqrt(1 - ρ_δ^2),        # Discount factor shock

                    # Some hyperparameters
                    wmin = 0.1,                         # Lower bound of cash-on-hand grid
                    wmax = 4.0,                         # Upper bound of cash-on-hand grid
                    n = 128                             # Length of simulated training samples in each epoch
                )
Params = Params()

# Set up neural network
numInputs = 5
numUnits = 32
numOutputs = 2
function NeuralNetwork()
    return Chain(
            Dense(numInputs, numUnits,relu; init = Flux.kaiming_uniform()),
            Dense(numUnits, numUnits,relu),
            Dense(numUnits, numUnits,relu),
            Dense(numUnits,numOutputs)
            )
end
m    = NeuralNetwork()
opt = Flux.setup(Adam(0.001, (0.9, 0.999)), m)

# Time solution
epochs = 50000
nn_train!(opt,m,Params,1)
@time nn_train!(opt,m,Params,epochs)

@unpack n, σ_e_r, σ_e_δ, σ_e_p, σ_e_q, wmin, wmax = Params
wvec = collect(LinRange(wmin, wmax, 100)')
ζvec , hvec = hh_policy_fn(m,wvec*0,wvec*0,wvec*0,wvec*0,wvec,Params)
plot(wvec', wvec', linestyle=:dash)
plot!(wvec', wvec'.*ζvec', linestyle=:solid)
