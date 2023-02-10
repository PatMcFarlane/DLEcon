using DrWatson
@quickactivate "DLECON"

using Flux
using Plots
using LaTeXStrings
using Statistics

#Model Parameters
#----------------------

#variable to determine floating point precision
const prec = 1e0

#Behavioural parameters
const β = oftype(prec, 0.9)
const γ = oftype(prec, 2.0)

#Shock variance and persistence
const σ_r = oftype(prec, 0.001)
const ρ_r = oftype(prec, 0.2)
const σ_p = oftype(prec, 0.0001)
const ρ_p = oftype(prec, 0.999)
const σ_q = oftype(prec, 0.001)
const ρ_q = oftype(prec, 0.9)
const σ_δ = oftype(prec, 0.001)
const ρ_δ = oftype(prec, 0.2)

#Steady-state interest rate
const r̄ = oftype(prec, 1.04)

#Standard deviations for the ergodic distriubtion of exogenous state variables:
const σ_e_r = σ_r/(1-ρ_r^2)^oftype(prec, 0.5)
const σ_e_p = σ_p/(1-ρ_p^2)^oftype(prec, 0.5)
const σ_e_q = σ_q/(1-ρ_q^2)^oftype(prec, 0.5)
const σ_e_δ = σ_δ/(1-ρ_δ^2)^oftype(prec, 0.5)

# bounds for endogenous state variable
const wmin = oftype(prec, 0.1)
const wmax = oftype(prec, 4.0)

#Fischer-Burmeister Function
#---------------------------------
min_FB(a,b) = a .+ b .- sqrt.(a.^oftype(prec,2.0) .+ b.^oftype(prec,2.0))

#Parameterize decision function via a neural network
#------------------------------------------------------------
perceptron = Chain(Dense(5 => 32, relu),
          Dense(32 => 32, relu),
          Dense(32 => 32, relu),
          Dense(32 => 2))

#Define Decision rule Function
#------------------------------------------------------------
function dr(f, r, δ, q, p, w)

    #normalize exogenous state variables by their 2 standard deviations so they are typically between -1 and 1
    rtrans = r./σ_e_r./oftype(prec, 2.0)
    δtrans = δ./σ_e_δ./oftype(prec, 2.0)
    qtrans = q./σ_e_q./oftype(prec, 2.0)
    ptrans = p./σ_e_p./oftype(prec, 2.0)

    #normalize income to be between -1 and 1
    wtrans = (w .- wmin)./(wmax .- wmin).*oftype(prec, 2.0) .- oftype(prec, 1.0)

    #prepare input to the perceptron
    s = [rtrans δtrans qtrans ptrans wtrans]'

    x = f(s)

    ζ = σ.(x[1,:])

    h = exp.(x[2,:])

    return ζ, h

end

#Plot the initial value of the decision rule
#---------------------------------------------------
wvec = range(wmin, stop = wmax, length = 100)

ζvec, hvec = dr(perceptron, wvec*0.0, wvec*0.0, wvec*0.0, wvec*0.0, wvec)

plot(wvec, wvec, ls=:dash)
plot!(wvec, wvec.*ζvec)
xlabel!(L"w_t")
ylabel!(L"c_t")
title!("Initial Guess")

#Define Euler Residuals Function
#-------------------------------------------------------
function Residuals(f, ε_r, ε_δ, ε_q, ε_p, r, δ, q, p, w)

    #arguments correspond to the values of the states today
    ζ, h = dr(f, r, δ, q, p, w)
    c = ζ.*w

    #transitions of the exogenous processes
    rP = r.*ρ_r .+ ε_r
    δP = δ.*ρ_δ .+ ε_δ
    pP = p.*ρ_p .+ ε_p
    qP = q.*ρ_q .+ ε_q
    
    #transition of engoenous states
    wP = exp.(pP).*exp.(qP) .+ (w .- c).*r̄.*exp.(rP)

    ζP, hP = dr(f, rP, δP, qP, pP, wP)
    cP = ζP.*wP

    R1 = β.*exp.(δP .- δ).*(cP./c).^(-γ).*r̄.*exp.(rP) .- h
    R2 = min_FB(oftype(prec,1.0) .- h, oftype(prec,1.0) .- ζ)

    return R1, R2

end

#Define all-in-one expectation operator function
#-------------------------------------------------------
function Ξ(f, n)
    
    r = randn(typeof(prec),n).*σ_e_r
    δ = randn(typeof(prec),n).*σ_e_δ
    p = randn(typeof(prec),n).*σ_e_p
    q = randn(typeof(prec),n).*σ_e_q
    w = rand(typeof(prec),n).*(wmax .- wmin) .+ wmin

    #random draws for first realization
    ε1_r = randn(typeof(prec),n).*σ_r
    ε1_δ = randn(typeof(prec),n).*σ_δ
    ε1_p = randn(typeof(prec),n).*σ_p
    ε1_q = randn(typeof(prec),n).*σ_q

    #random draws for 2nd realization
    ε2_r = randn(typeof(prec),n).*σ_r
    ε2_δ = randn(typeof(prec),n).*σ_δ
    ε2_p = randn(typeof(prec),n).*σ_p
    ε2_q = randn(typeof(prec),n).*σ_q

    #residuals for n random grid points under 2 realization of shocks
    R1_ε1, R2_ε1 = Residuals(f, ε1_r, ε1_δ, ε1_q, ε1_p, r, δ, q, p, w)
    R1_ε2, R2_ε2 = Residuals(f, ε2_r, ε2_δ, ε2_q, ε2_p, r, δ, q, p, w)

    #Construct all-in-one expectation operator
    return sum(R1_ε1.*R1_ε2 .+ R2_ε1.*R2_ε2)/n

end

#Training the Neural network
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#Set flux optimizer to Adam
opt = Flux.setup(Adam(oftype(prec,0.001),(oftype(prec,0.9),oftype(prec,0.999)),oftype(prec,1.0e-7)),perceptron)

#Training step
function training_step!(perceptron,n, opt)
    grads = Flux.gradient(m -> Ξ(m, n), perceptron)
    Flux.update!(opt, perceptron, grads[1])    
end


#Training loop
function train_iter!(perceptron,K,n,opt)
    vals = typeof(prec)[]

    for i in 1:K
        push!(vals, Ξ(perceptron, n))
        training_step!(perceptron, n, opt)
    end

    return vals
end

training_step!(perceptron, 128, opt)

#warmup (to precompile before timing)
train_iter!(perceptron,1,128,opt)

#Time for 50K iterations
@time vals = train_iter!(perceptron, 50000 - 1, 128, opt);

plot(sqrt.(abs.(vals)), xscale = :log10, yscale = :log10)


ζvec, hvec = dr(perceptron, wvec*0.0, wvec*0.0, wvec*0.0, wvec*0.0, wvec)

plot(wvec, wvec, ls=:dash)
plot!(wvec, wvec.*ζvec)
xlabel!(L"w_t")
ylabel!(L"c_t")
title!("Final Solution")