##############################################################################################
# An MIT shock in a model of Aiyagari
##############################################################################################

##############################################################################################
# 1. Preliminaries and setup
##############################################################################################

# Activate and set up project 
using DrWatson
@quickactivate "p07_Aiyagari"

# Specifications
const prec = 1.0

# Package inclusions
include(srcdir("p07_Packages.jl"))

# Set structural model parameters
const β = oftype(prec,0.98)     # Discount factor
const γ = oftype(prec,2.0)      # Inverse elasticity of intertemporal substitution 
const α = oftype(prec,0.40)     # Capital elasticity 
const δ = oftype(prec,0.02)     # Depreciation rate     
const σ = oftype(prec,0.05)     # Volatility of TFP shock 
const ρ = oftype(prec,0.95)     # Persistence of TFP shock

# Parameters of the idiosyncratic employment process
const n_empl = 2                                                                        # Number of idiosyncratic employment states
const trans_empl = oftype.(prec,[[1.0 - 0.6 ; 0.6] [0.2 ; 1.0 - 0.2]])                  # Employment state transition matrix. Element (i,j) = P(moving from j to i)
const StatDist_empl = eigen(trans_empl).vectors[:,(eigen(trans_empl).values .== 1.0)]./sum(eigen(trans_empl).vectors[:,(eigen(trans_empl).values .== 1.0)])
const grid_empl = oftype.(prec,[1.0 ; 2.5]) ./ dot(oftype.(prec,[1.0 ; 2.5]) , StatDist_empl)   # Vector of employment levels
const Lbar = dot(grid_empl,StatDist_empl)                                               # Aggregate employment 

# Parameters of idiosyncratic wealth process
const a_min = oftype(prec,0.0)
const a_max = oftype(prec,200.0)
const n_wealth = 201
const grid_wealth = collect(range(a_min^0.25, a_max^0.25, length = n_wealth)).^oftype(prec,4.0)

# Resized versions of the grid to represent idiosyncratic state space
const mesh_wealth = transpose(repeat(grid_wealth, outer = [1,n_empl]))
const mesh_empl = repeat(grid_empl, outer = [1,n_wealth])

# Algorithm parameters for conventional solution approach
const tol = 1.0e-6
const max_iter = 10000
const tuner_K = oftype(prec,0.25)

# Algorithm parameters for machine learning approach
# To be implemented

# Function inclusions
include(srcdir("p07_Functions.jl"))

# Create the MIT shock 
const Tmax = 300
Z_path = zeros(typeof(prec),Tmax)
Z_path[1] = σ                                                   # One standard deviation shock
for tt in 2:Tmax
    Z_path[tt] = ρ*Z_path[tt-1]                                 # Evolves with persistence ρ ...
    if abs(Z_path[tt]) < 1.0e-4                                 # ... but when it becomes small, just set it to zero
        Z_path[tt] = 0.0
    end
end
Z_path = exp.(Z_path)                                           # From log to level
const Z_length = searchsortedfirst(-Z_path,-1.0) - 1            # Location of last entry before Z returns to Z_star = 1.0

##############################################################################################
# 2. Solution by conventional means
#    Adapted from code of Julien Pascal
#    https://notes.quantecon.org/submission/5f5f811909e80c001bbd6eaf
##############################################################################################

# Obtain steady state capital and the stationary distribution
K_star = GetSteadyStateCapital(46.0,46.1)
g_star, c_star, g_low_star, g_high_star, success_flag = solve_EGM(x->log(x), x->log(x), RealReturn(K_star), Wage(K_star))
T_star = GetTransitionMatrix(g_star)
μ_star = GetStationaryDist(T_star)

plot(grid_wealth,μ_star[1:n_wealth]./sum(μ_star[1:n_wealth]), label = "Low employment state", title="Wealth distribution by employment level")
plot!(grid_wealth,μ_star[n_wealth+1:n_wealth+n_wealth]./sum(μ_star[n_wealth+1:n_wealth+n_wealth]), label = "High employment state")
savefig(plotsdir("StationaryWealthDistribution"))

# MIT shock
K_path = fill(K_star,Tmax)
K_path, convergence_flag = GetMITShockSolution(K_path,Z_path,g_low_star,g_high_star,μ_star,K_star)
R_path = RealReturn(K_path,Z_path)
W_path = Wage(K_path,Z_path)
Y_path = Z_path.*K_path.^α.*Lbar.^(oftype(prec,1.0) .- α)
C_path = similar(Y_path)
C_path[1:Tmax-1] = Y_path[1:Tmax-1] .+ (oftype(prec,1.0) .- δ).*K_path[1:Tmax-1] .- K_path[2:Tmax]
C_path[Tmax] = K_star^α*Lbar^(oftype(prec,1.0) - α) - δ*K_star

plot(layout = (2,2))
plot!(Y_path, sp=1, title="Output", legend=false)
plot!(C_path, sp=2, title="Consumption", legend=false)
plot!(R_path, sp=3, title="Interest rate", legend=false)
plot!(W_path, sp=4, title="Wage", legend=false)
savefig(plotsdir("MIT_Conventional"))

##############################################################################################
# 3. Solution by machine learning
##############################################################################################

# To be implemented
