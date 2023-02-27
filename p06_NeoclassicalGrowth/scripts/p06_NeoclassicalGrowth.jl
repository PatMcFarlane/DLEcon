##############################################################################################
# Exploring solution methods for the neoclassical growth model
##############################################################################################

##############################################################################################
# 1. Preliminaries and setup
##############################################################################################

# Activate and set up project 
using DrWatson
@quickactivate "p06_NeoclassicalGrowth"

# Specifications
UseSavedNeuralNets = true      # If false, neural nets are re-trained. 

# Set model parameters
const α = 0.33                 # Capital elasticity
const β = 0.95                 # Discount factor
const δ = 0.05                 # Depreciation rate
const γ = 1.0                  # Inverse elasticity of intertemporal substitution
const σ = 0.05                 # Volatility of productivity shock

# Model steady state
const Kss = ((1.0/β - (1.0 - δ))/α)^(1.0/(α-1.0))      # Capital stock
const Css = Kss^α - δ*Kss                              # Consumption

# Set some hyperparameters
const tol = 1.0e-6                          # Tolerance for standard solution
const Tmax = 100                            # Time horizon to approximate ∞
const Ntrain = 5000                         # Number of training sequences (for Approach A)
const Ntest = 5000                          # Number of test sequences (for Approach A)
const Ndraws = 100                          # Number of shock draws per epoch (for Approach B)
const wgt_euler = 1.0                       # Weight on Euler equation errors (for Approach B)
const wgt_lom = 1.0                         # Weight on capital law of motion errors (for Approach B)

# Include the requisite objects
include(srcdir("p06_Packages.jl"))                      # Julia packages
include(srcdir("p06_Functions.jl"))                     # Project functions

##############################################################################################
# 2. Sequence space solution by neural network
##############################################################################################

##############################################################################################
#    Approach A: Supervised learning
#    Try three versions of this:
#       A1. Approximate g: S -> SxS 
#       A2. Approximate g_c: S -> S 
#       A3. Approximate g_k: S -> S
##############################################################################################

# Create the training and test data
if ~UseSavedNeuralNets
    ShocksTrain = MakeShockSequences(Tmax,Ntrain)           # Shock sequences for training
    Ctrain,Ktrain = Shooting(ShocksTrain)
end
ShocksTest = MakeShockSequences(Tmax,Ntest)             # Shock sequences for testing
Ctest,Ktest = Shooting(ShocksTest)

##############################
# A1. Approximate g: S -> SxS 
##############################

if ~UseSavedNeuralNets 

    # Set up neural network
    numInputs = Tmax                # Input = a shock sequence of length Tmax. (If n shocks, need nxTmax inputs.)
    numUnits = 64
    numOutputs = 2*Tmax             # Output two sequences: g_c = consumption; g_k = capital stock
    function NeuralNetwork()
        return Chain(
                Dense(numInputs, numUnits,relu; init = Flux.kaiming_uniform()),
                Dense(numUnits, numUnits,relu),
                Dense(numUnits, numUnits,relu),
                Dense(numUnits,numOutputs,exp) 
                )
    end
    DataTrain = [(ShocksTrain,vcat(Ctrain,Ktrain))]
    m_a1 = NeuralNetwork()
    opt = Flux.setup(Adam(0.001, (0.9, 0.999)), m_a1)
    loss(m,x,y) = sum(Flux.Losses.mse(m(x), y))

    # Train the network
    epochs = 100000
    Flux.train!(loss,m_a1,DataTrain,opt)
    @time for i in 1:epochs
        Flux.train!(loss,m_a1,DataTrain,opt)
    end

    # Save model solution
    @save datadir("res_a1.jld2") m_a1 ShocksTrain 

end
if UseSavedNeuralNets
    @load datadir("res_a1.jld2") m_a1 ShocksTrain 
end

# Get predictions on training sample
if UseSavedNeuralNets
    Ctrain,Ktrain = Shooting(ShocksTrain)   # Do this here since it wasn't done earlier
end
Ctrain_a1 = m_a1(ShocksTrain)[1:Tmax,:]
Ktrain_a1 = m_a1(ShocksTrain)[Tmax+1:Tmax+Tmax,:]

# Get predictions on test sample
Ctest_a1 = m_a1(ShocksTest)[1:Tmax,:]
Ktest_a1 = m_a1(ShocksTest)[Tmax+1:Tmax+Tmax,:]

# Plot some responses
plot(Ktest[:,1:16], layout = (4,4), legend = false )
plot!(Ktest_a1[:,1:16], layout = (4,4), legend = false)
savefig(plotsdir("fig_K_a1"))

plot(Ctest[:,1:16], layout = (4,4), legend = false )
plot!(Ctest_a1[:,1:16], layout = (4,4), legend = false)
savefig(plotsdir("fig_C_a1"))

# Summarize errors 
err_K_a1 = 100.0*(Ktest_a1 .- Ktest)./Ktest
err_C_a1 = 100.0*(Ctest_a1 .- Ctest)./Ctest
histogram(mean(err_K_a1,dims = 1)', title = "Model A1: Mean percent error in K", legend = false)
savefig(plotsdir("fig_err_K_a1_mean"))
histogram(mean(err_C_a1,dims = 1)', title = "Model A1: Mean percent error in C", legend = false)
savefig(plotsdir("fig_err_C_a1_mean"))
histogram(GetMaxError(err_K_a1)', title = "Model A1: Largest percent error in K", legend = false)
savefig(plotsdir("fig_err_K_a1_max"))
histogram(GetMaxError(err_C_a1)', title = "Model A1: Largest percent error in C", legend = false)
savefig(plotsdir("fig_err_C_a1_max"))

##############################
# A2. Approximate g_c: S -> SxS 
##############################

if ~UseSavedNeuralNets

    # Set up neural network
    numInputs = Tmax                # Input = a shock sequence of length Tmax. (If n shocks, need nxTmax inputs.)
    numUnits = 64
    numOutputs = Tmax             # Output two sequences: g_c = consumption; g_k = capital stock
    function NeuralNetwork()
        return Chain(
                Dense(numInputs, numUnits,relu; init = Flux.kaiming_uniform()),
                Dense(numUnits, numUnits,relu),
                Dense(numUnits, numUnits,relu),
                Dense(numUnits,numOutputs,exp)
                )
    end
    DataTrain = [(ShocksTrain,Ctrain)]
    m_a2 = NeuralNetwork()
    opt = Flux.setup(Adam(0.001, (0.9, 0.999)), m_a2)
    loss(m,x,y) = sum(Flux.Losses.mse(m(x), y))

    # Train the network
    epochs = 100000
    Flux.train!(loss,m_a2,DataTrain,opt)
    @time for i in 1:epochs
        Flux.train!(loss,m_a2,DataTrain,opt)
    end

    # Save model solution
    @save datadir("res_a2.jld2") m_a2 ShocksTrain 

end
if UseSavedNeuralNets
    @load datadir("res_a2.jld2") m_a2 ShocksTrain 
end

# Get predictions on training sample
Ctrain_a2 = m_a2(ShocksTrain)
Ktrain_a2 = GetK(Ctrain_a2,ShocksTrain)
Ktrain_a0 = GetK(Ctrain,ShocksTrain)

# Get predictions on test sample
Ctest_a2 = m_a2(ShocksTest)
Ktest_a2 = GetK(Ctest_a2,ShocksTest)

# Plot some responses
plot(Ktest[:,1:16], layout = (4,4), legend = false )
plot!(Ktest_a2[:,1:16], layout = (4,4), legend = false)
savefig(plotsdir("fig_K_a2"))

plot(Ctest[:,1:16], layout = (4,4), legend = false )
plot!(Ctest_a2[:,1:16], layout = (4,4), legend = false)
savefig(plotsdir("fig_C_a2"))

# Summarize errors 
err_K_a2 = 100.0*(Ktest_a2 .- Ktest)./Ktest
err_C_a2 = 100.0*(Ctest_a2 .- Ctest)./Ctest
histogram(mean(err_K_a2,dims = 1)', title = "Model A2: Mean percent error in K", legend = false)
savefig(plotsdir("fig_err_K_a2_mean"))
histogram(mean(err_C_a2,dims = 1)', title = "Model A2: Mean percent error in C", legend = false)
savefig(plotsdir("fig_err_C_a2_mean"))
histogram(GetMaxError(err_K_a2)', title = "Model A2: Largest percent error in K", legend = false)
savefig(plotsdir("fig_err_K_a2_max"))
histogram(GetMaxError(err_C_a2)', title = "Model A2: Largest percent error in C", legend = false)
savefig(plotsdir("fig_err_C_a2_max"))

##############################
# A3 Approximate g_k: S -> SxS 
##############################

if ~UseSavedNeuralNets

    # Set up neural network
    numInputs = Tmax                # Input = a shock sequence of length Tmax. (If n shocks, need nxTmax inputs.)
    numUnits = 64
    numOutputs = Tmax               # Output two sequences: g_c = consumption; g_k = capital stock
    function NeuralNetwork()
        return Chain(
                Dense(numInputs, numUnits,relu; init = Flux.kaiming_uniform()),
                Dense(numUnits, numUnits,relu),
                Dense(numUnits, numUnits,relu),
                Dense(numUnits,numOutputs,exp)
                )
    end
    DataTrain = [(ShocksTrain,Ktrain)]
    m_a3 = NeuralNetwork()
    opt = Flux.setup(Adam(0.001, (0.9, 0.999)), m_a3)
    loss(m,x,y) = sum(Flux.Losses.mse(m(x), y))

    # Train the network
    epochs = 100000
    Flux.train!(loss,m_a3,DataTrain,opt)
    @time for i in 1:epochs
        Flux.train!(loss,m_a3,DataTrain,opt)
    end

    # Save model solution
    @save datadir("res_a3.jld2") m_a3 ShocksTrain 

end
if UseSavedNeuralNets
    @load datadir("res_a3.jld2") m_a3 ShocksTrain 
end

# Get predictions on training sample
Ktrain_a3 = m_a3(ShocksTrain)
Ctrain_a3 = GetC(Ktrain_a3,ShocksTrain)
Ctrain_a0 = GetC(Ktrain,ShocksTrain)

# Get predictions on test sample
Ktest_a3 = m_a3(ShocksTest)
Ctest_a3 = GetC(Ktest_a3,ShocksTest)

# Plot some responses
plot(Ktest[:,1:16], layout = (4,4), legend = false )
plot!(Ktest_a3[:,1:16], layout = (4,4), legend = false)
savefig(plotsdir("fig_K_a3"))

plot(Ctest[:,1:16], layout = (4,4), legend = false )
plot!(Ctest_a3[:,1:16], layout = (4,4), legend = false)
savefig(plotsdir("fig_C_a3"))

# Summarize errors 
err_K_a3 = 100.0*(Ktest_a3 .- Ktest)./Ktest
err_C_a3 = 100.0*(Ctest_a3 .- Ctest)./Ctest
histogram(mean(err_K_a3,dims = 1)', title = "Model A3: Mean percent error in K", legend = false)
savefig(plotsdir("fig_err_K_a3_mean"))
histogram(mean(err_C_a3,dims = 1)', title = "Model A3: Mean percent error in C", legend = false)
savefig(plotsdir("fig_err_C_a3_mean"))
histogram(GetMaxError(err_K_a3)', title = "Model A3: Largest percent error in K", legend = false)
savefig(plotsdir("fig_err_K_a3_max"))
histogram(GetMaxError(err_C_a3)', title = "Model A3: Largest percent error in C", legend = false)
savefig(plotsdir("fig_err_C_a3_max"))

##############################################################################################
#    Approach B: Euler equation method
#    Try two versions of this:
#       B1. Approximate g: S -> SxS using entire sequence of Euler errors at each draw
#       B2. Approximate g: S -> SxS using Euler errors in random adjacent periods per draw
#    In training, I impose the known initial condition K[1] and the known terminal condition
#    C[Tmax]. The neural network approximates the remaining 2*Tmax - 2 elements of the C and 
#    K sequences.
##############################################################################################

#################################################################################
# B1. Approximate g: S -> SxS using entire sequence of Euler errors at each draw
#################################################################################

if ~UseSavedNeuralNets

    # Set up neural network
    numInputs = Tmax                # Input = a shock sequence of length Tmax. (If n shocks, need nxTmax inputs.)
    numUnits = 64
    numOutputs = 2*Tmax - 2         # Output two sequences: g_c = consumption; g_k = capital stock. But we will impose boundary conditions, which reduces T dimension by one for each sequence.
    function NeuralNetwork()
        return Chain(
                Dense(numInputs, numUnits,relu; init = Flux.kaiming_uniform()),
                Dense(numUnits, numUnits,relu),
                Dense(numUnits, numUnits,relu),
                Dense(numUnits,numOutputs,exp)
                )
    end
    m_b1    = NeuralNetwork()
    opt = Flux.setup(Adam(0.001, (0.9, 0.999)), m_b1)

    # Time solution
    epochs = 300000
    nn_train!(opt,m_b1,loss_b1,1)
    @time nn_train!(opt,m_b1,loss_b1,epochs)

    # Save model solution
    @save datadir("res_b1.jld2") m_b1 

end
if UseSavedNeuralNets
    @load datadir("res_b1.jld2") m_b1 
end

# Get predictions on test sample
Ctest_b1 = m_b1(ShocksTest)[1:Tmax-1,:]
Ctest_b1 = [Ctest_b1 ; fill(Css,1,Ntest)]               # Add in terminal condition
Ktest_b1 = m_b1(ShocksTest)[Tmax-1+1:Tmax-1+Tmax-1,:]
Ktest_b1 = [fill(Kss,1,Ntest) ; Ktest_b1]               # Add in initial condition 

# Plot some responses
plot(Ktest[:,1:16], layout = (4,4), legend = false )
plot!(Ktest_b1[:,1:16], layout = (4,4), legend = false)
savefig(plotsdir("fig_K_b1"))

plot(Ctest[:,1:16], layout = (4,4), legend = false )
plot!(Ctest_b1[:,1:16], layout = (4,4), legend = false)
savefig(plotsdir("fig_C_b1"))

# Summarize errors 
err_K_b1 = 100.0*(Ktest_b1 .- Ktest)./Ktest
err_C_b1 = 100.0*(Ctest_b1 .- Ctest)./Ctest
histogram(mean(err_K_b1,dims = 1)', title = "Model B1: Mean percent error in K", legend = false)
savefig(plotsdir("fig_err_K_b1_mean"))
histogram(mean(err_C_b1,dims = 1)', title = "Model B1: Mean percent error in C", legend = false)
savefig(plotsdir("fig_err_C_b1_mean"))
histogram(GetMaxError(err_K_b1)', title = "Model B1: Largest percent error in K", legend = false)
savefig(plotsdir("fig_err_K_b1_max"))
histogram(GetMaxError(err_C_b1)', title = "Model B1: Largest percent error in C", legend = false)
savefig(plotsdir("fig_err_C_b1_max"))

#################################################################################
# B2. Approximate g: S -> SxS using Euler errors in random adjacent periods
#################################################################################

# Not yet implemented

