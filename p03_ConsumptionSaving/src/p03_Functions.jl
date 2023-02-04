
##############################################################################################
# Functions
##############################################################################################

# Fischer-Burmeister
function Ψ_fb(a::Matrix{Float64},b::Matrix{Float64}) 
    return a .+ b .- sqrt.(a.^2 .+ b.^2)
end

# Consumer policy function
function hh_policy_fn(m::Chain,r::Matrix{Float64},δ::Matrix{Float64},p::Matrix{Float64},q::Matrix{Float64},w::Matrix{Float64},Params::NamedTuple)

    @unpack σ_e_r, σ_e_p, σ_e_q, σ_e_δ, wmin, wmax = Params 
    
    # Normalize inputs
    r_ = r/σ_e_r/2
    δ_ = δ/σ_e_δ/2
    p_ = p/σ_e_p/2
    q_ = q/σ_e_q/2
    w_ = (w .- wmin)./(wmax .- wmin).*2.0 - 1.0

    # Evaluate by neural network
    x = [r_ ; δ_ ; p_ ; q_ ; w_]
    y = m(x)

    ζ = Flux.sigmoid.(y[1,:])'
    h = exp.(y[2,:])'

    return ζ , h

end

# Realized optimality condition residuals
function hh_residuals(m::Chain,e_r::Matrix{Float64},e_δ::Matrix{Float64},e_p::Matrix{Float64},e_q::Matrix{Float64},r::Matrix{Float64},δ::Matrix{Float64},p::Matrix{Float64},q::Matrix{Float64},w::Matrix{Float64},Params::NamedTuple)

    @unpack ρ_r, σ_r, ρ_δ, σ_δ, ρ_p, σ_p, ρ_q, σ_q, β, γ, rbar = Params

    # Today's choices
    ζ , h = hh_policy_fn(m,r,δ,p,q,w,Params)
    c = ζ.*w

    # Exogenous state transitions conditional on specified shocks
    rnext = ρ_r*r .+ e_r
    δnext = ρ_δ*δ .+ e_δ
    pnext = ρ_p*p .+ e_p
    qnext = ρ_q*q .+ e_q

    # Endogenous state transition
    wnext = exp.(pnext).*exp.(qnext) .+ (w - c).*rbar.*exp.(rnext)

    # Tomorrow's choices
    ζnext , hnext = hh_policy_fn(m,rnext,δnext,pnext,qnext,wnext,Params)
    cnext = ζnext.*wnext

    # Residuals
    R1 = β*exp.(δnext .- δ).*(cnext./c).^(-γ).*rbar.*exp.(rnext) .- h
    R2 = Ψ_fb(1.0 .- h, 1.0 .- ζ)

    return R1, R2

end

# Objective function
function loss(m::Chain,r::Matrix{Float64},δ::Matrix{Float64},p::Matrix{Float64},q::Matrix{Float64},w::Matrix{Float64},Params::NamedTuple)

    @unpack σ_r, σ_δ, σ_p, σ_q = Params
    n_ = size(r)

    # Draw shocks
    e1_r = rand(Normal(0, σ_r), n_)
    e1_δ = rand(Normal(0, σ_δ), n_)
    e1_p = rand(Normal(0, σ_p), n_)
    e1_q = rand(Normal(0, σ_q), n_)
    e2_r = rand(Normal(0, σ_r), n_)
    e2_δ = rand(Normal(0, σ_δ), n_)
    e2_p = rand(Normal(0, σ_p), n_)
    e2_q = rand(Normal(0, σ_q), n_)

    # Get residuals
    R1_e1, R2_e1 = hh_residuals(m,e1_r,e1_δ,e1_p,e1_q,r,δ,p,q,w,Params)
    R1_e2, R2_e2 = hh_residuals(m,e2_r,e2_δ,e2_p,e2_q,r,δ,p,q,w,Params)

    # All-in-one expectation
    Rsq = R1_e1.*R1_e2 .+ R2_e1.*R2_e2

    # Return average over draws 
    return mean(Rsq)

end

# Train the neural network
function nn_train!(opt::NamedTuple, m::Chain,Params::NamedTuple,epochs::Int64)

    @unpack n, σ_e_r, σ_e_δ, σ_e_p, σ_e_q, wmin, wmax = Params
    for i in one(epochs):epochs
    
        r = rand(Normal(0, σ_e_r), 1, n)
        δ = rand(Normal(0, σ_e_δ), 1, n)
        p = rand(Normal(0, σ_e_p), 1, n)
        q = rand(Normal(0, σ_e_q), 1, n)
        w = rand(Uniform(wmin, wmax), 1, n)
    
        grads = Flux.gradient(m) do f
            loss(f,r,δ,p,q,w,Params)
        end
    
        Flux.update!(opt, m, grads[1])
        
    end

    #return opt, m
    

end
