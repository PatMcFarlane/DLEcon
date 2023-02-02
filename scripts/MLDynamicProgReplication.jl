using DrWatson
@quickactivate "DLECON"

using Parameters
using Flux
using Plots
using LaTeXStrings
using Statistics

#Model Parameters
ModParas = @with_kw (
    β = 0.9,
    γ = 2.0,
    σ_r = 0.001,
    ρ_r = 0.2,
    σ_p = 0.0001,
    ρ_p = 0.999,
    σ_q = 0.001,
    ρ_q = 0.9,
    σ_δ = 0.001,
    ρ_δ = 0.2,
    r̄ = 1.04,

    #Standard deviations for the ergodic distriubtion of exogenous state variables:
    σ_e_r = σ_r/(1-ρ_r^2)^0.5,
    σ_e_p = σ_p/(1-ρ_p^2)^0.5,
    σ_e_q = σ_q/(1-ρ_q^2)^0.5,
    σ_e_δ = σ_δ/(1-ρ_δ^2)^0.5,

    # bounds for endogenous state variable
    wmin = 0.1,
    wmax = 4.0
    )

P = ModParas()



#Fischer-Burmeister Function
min_FB(a,b) = a .+ b .- sqrt.(a.^2.0 .+ b.^2.0)

#Parameterize decision function via a neural network
perceptron = Chain(Dense(5 => 32, relu),
          Dense(32 => 32, relu),
          Dense(32 => 32, relu),
          Dense(32 => 2))

#Define Decision rule Function
function dr(f, r, δ, q, p, w, P)

    @unpack σ_e_r, σ_e_δ, σ_e_q, σ_e_p, wmax, wmin = P

    #normalize exogenous state variables by their 2 standard deviations so they are typically between -1 and 1
    rtrans = r./σ_e_r./2.0
    δtrans = δ./σ_e_δ./2.0
    qtrans = q./σ_e_q./2.0
    ptrans = p./σ_e_p./2.0

    #normalize income to be between -1 and 1
    wtrans = (w .- wmin)./(wmax .- wmin).*2.0 .- 1.0

    #prepare input to the perceptron
    s = [rtrans δtrans qtrans ptrans wtrans]'

    x = f(s)

    ζ = σ.(x[1,:])

    h = exp.(x[2,:])

    return ζ, h

end

#Plot the initial value of the decision rule
wvec = range(P.wmin, stop = P.wmax, length = 100)

ζvec, hvec = dr(perceptron, wvec*0.0, wvec*0.0, wvec*0.0, wvec*0.0, wvec, P)

plot(wvec, wvec, ls=:dash)
plot!(wvec, wvec.*ζvec)
xlabel!(L"w_t")
ylabel!(L"c_t")
title!("Initial Guess")

#Define Euler Residuals Function
function Residuals(f, ε_r, ε_δ, ε_q, ε_p, r, δ, q, p, w, P)

    @unpack ρ_r, ρ_δ, ρ_p, ρ_q, r̄, β, γ = P

    #arguments correspond to the values of the states today
    ζ, h = dr(f, r, δ, q, p, w, P)
    c = ζ.*w

    #transitions of the exogenous processes
    rP = r.*ρ_r .+ ε_r
    δP = δ.*ρ_δ .+ ε_δ
    pP = p.*ρ_p .+ ε_p
    qP = q.*ρ_q .+ ε_q
    
    #transition of engoenous states
    wP = exp.(pP).*exp.(qP) .+ (w .- c).*r̄.*exp.(rP)

    ζP, hP = dr(f, rP, δP, qP, pP, wP, P)
    cP = ζP.*wP

    R1 = β.*exp.(δP .- δ).*(cP./c).^(-γ).*r̄.*exp.(rP) .- h
    R2 = min_FB(1.0 .- h, 1.0 .- ζ)

    return R1, R2

end

#Define all-in-one expectation operator function

function Ξ(f, n, P)
    
    @unpack  σ_e_r,  σ_e_p,  σ_e_q,  σ_e_δ, wmin, wmax  = P

    r = randn(n).*σ_e_r
    δ = randn(n).*σ_e_δ
    p = randn(n).*σ_e_p
    q = randn(n).*σ_e_q
    w = rand(n).*(wmax .- wmin) .+ wmin

    #random draws for first realization
    ε1_r = randn(n).*σ_e_r
    ε1_δ = randn(n).*σ_e_δ
    ε1_p = randn(n).*σ_e_p
    ε1_q = randn(n).*σ_e_q

    #random draws for 2nd realization
    ε2_r = randn(n).*σ_e_r
    ε2_δ = randn(n).*σ_e_δ
    ε2_p = randn(n).*σ_e_p
    ε2_q = randn(n).*σ_e_q

    #residuals for n random grid points under 2 realization of shocks
    R1_ε1, R2_ε1 = Residuals(f, ε1_r, ε1_δ, ε1_q, ε1_p, r, δ, q, p, w, P)
    R1_ε2, R2_ε2 = Residuals(f, ε2_r, ε2_δ, ε2_q, ε2_p, r, δ, q, p, w, P)

    #Construct all-in-one expectation operator
    return sum(R1_ε1.*R1_ε2 .+ R2_ε1.*R2_ε2)/n

end

#Training the Neural network
opt = Flux.setup(Adam(0.001,(0.9,0.999),1.0e-7),perceptron)

#Training step
function training_step!(perceptron,n, P, opt)
    grads = Flux.gradient(m -> Ξ(m, n, P), perceptron)
    Flux.update!(opt, perceptron, grads[1])    
end


#Training loop
function train_iter!(perceptron,K,n,P,opt)
    vals = Float32[]

    for i in 1:K
        push!(vals, Ξ(perceptron, n, P))
        training_step!(perceptron, n, P, opt)
    end

    return vals
end

training_step!(perceptron, 128, P, opt)

#warmup 
train_iter!(perceptron,1,128,P,opt)

@time vals = train_iter!(perceptron, 50000 - 1, 128, P, opt);

plot(sqrt.((vals)), xscale = :log10, yscale = :log10)


plot(wvec, wvec, ls=:dash)
plot!(wvec, wvec.*ζvec)
xlabel!(L"w_t")
ylabel!(L"c_t")
title!("Final Solution")