

##############################################################################################
# Functions
##############################################################################################

# Fischer-Burmeister
function Ψ_fb(a,b) 
    return a .+ b .- sqrt.(a.^oftype(prec,2.0) .+ b.^oftype(prec,2.0))
end

# Consumer policy function
function hh_policy_fn(m,r,δ,p,q,w)

    #@unpack σ_e_r, σ_e_p, σ_e_q, σ_e_δ, wmin, wmax = Params 
    
    # Normalize inputs
    r_ = r./σ_e_r./oftype(prec,2.0)
    δ_ = δ./σ_e_δ./oftype(prec,2.0)
    p_ = p./σ_e_p./oftype(prec,2.0)
    q_ = q./σ_e_q./oftype(prec,2.0)
    w_ = (w .- wmin)./(wmax .- wmin).*oftype(prec,2.0) .- oftype(prec,1.0)

    # Evaluate by neural network
    x = [r_ ; δ_ ; p_ ; q_ ; w_]
    y = m(x)

    ζ = Flux.sigmoid.(y[1,:])'
    h = exp.(y[2,:])'

    return ζ , h

end

# Realized optimality condition residuals
function hh_residuals(m,e_r,e_δ,e_p,e_q,r,δ,p,q,w)

    #@unpack ρ_r, σ_r, ρ_δ, σ_δ, ρ_p, σ_p, ρ_q, σ_q, β, γ, rbar = Params

    # Today's choices
    ζ , h = hh_policy_fn(m,r,δ,p,q,w)
    c = ζ.*w

    # Exogenous state transitions conditional on specified shocks
    rnext = ρ_r.*r .+ e_r
    δnext = ρ_δ.*δ .+ e_δ
    pnext = ρ_p.*p .+ e_p
    qnext = ρ_q.*q .+ e_q

    # Endogenous state transition
    wnext = exp.(pnext).*exp.(qnext) .+ (w .- c).*rbar.*exp.(rnext)

    # Tomorrow's choices
    ζnext , hnext = hh_policy_fn(m,rnext,δnext,pnext,qnext,wnext)
    cnext = ζnext.*wnext

    # Residuals
    R1 = β*exp.(δnext .- δ).*(cnext./c).^(-γ).*rbar.*exp.(rnext) .- h
    R2 = Ψ_fb(oftype(prec,1.0) .- h, oftype(prec,1.0) .- ζ)

    return R1, R2

end

# Objective function
# - Method from v1, which takes r,δ,p,q,w as arguments
function loss(m,r,δ,p,q,w)

    #@unpack σ_r, σ_δ, σ_p, σ_q = Params
    n_ = size(r)

    # Draw shocks
    e1_r = randn(typeof(prec), n_).*σ_r
    e1_δ = randn(typeof(prec), n_).*σ_δ
    e1_p = randn(typeof(prec), n_).*σ_p
    e1_q = randn(typeof(prec), n_).*σ_q
    e2_r = randn(typeof(prec), n_).*σ_r
    e2_δ = randn(typeof(prec), n_).*σ_δ
    e2_p = randn(typeof(prec), n_).*σ_p
    e2_q = randn(typeof(prec), n_).*σ_q

    # Get residuals
    R1_e1, R2_e1 = hh_residuals(m,e1_r,e1_δ,e1_p,e1_q,r,δ,p,q,w)
    R1_e2, R2_e2 = hh_residuals(m,e2_r,e2_δ,e2_p,e2_q,r,δ,p,q,w)

    # All-in-one expectation
    Rsq = R1_e1.*R1_e2 .+ R2_e1.*R2_e2

    # Return average over draws 
    return mean(Rsq)

end

# - Method that builds r,δ,p,q,w internally
function loss(m)

    #@unpack σ_r, σ_δ, σ_p, σ_q = Params

    r = randn(typeof(prec), 1, n).*σ_e_r
    δ = randn(typeof(prec), 1, n).*σ_e_δ
    p = randn(typeof(prec), 1, n).*σ_e_p
    q = randn(typeof(prec), 1, n).*σ_e_q
    w = rand(typeof(prec), 1, n).*(wmax .- wmin) .+ wmin
    
    # Draw shocks
    e1_r = randn(typeof(prec), 1, n).*σ_r
    e1_δ = randn(typeof(prec), 1, n).*σ_δ
    e1_p = randn(typeof(prec), 1, n).*σ_p
    e1_q = randn(typeof(prec), 1, n).*σ_q
    e2_r = randn(typeof(prec), 1, n).*σ_r
    e2_δ = randn(typeof(prec), 1, n).*σ_δ
    e2_p = randn(typeof(prec), 1, n).*σ_p
    e2_q = randn(typeof(prec), 1, n).*σ_q

    # Get residuals
    R1_e1, R2_e1 = hh_residuals(m,e1_r,e1_δ,e1_p,e1_q,r,δ,p,q,w)
    R1_e2, R2_e2 = hh_residuals(m,e2_r,e2_δ,e2_p,e2_q,r,δ,p,q,w)

    # All-in-one expectation
    Rsq = R1_e1.*R1_e2 .+ R2_e1.*R2_e2

    # Return average over draws 
    return mean(Rsq)

end

# Train the neural network
function nn_train!(opt,m,epochs)
#function nn_train!(opt,m,epochs,r::Matrix{typeof(prec)}=zeros(typeof(prec),1,n),δ::Matrix{typeof(prec)}=zeros(typeof(prec),1,n),p::Matrix{typeof(prec)}=zeros(typeof(prec),1,n),q::Matrix{typeof(prec)}=zeros(typeof(prec),1,n),w::Matrix{typeof(prec)}=zeros(typeof(prec),1,n))

    #@unpack n, σ_e_r, σ_e_δ, σ_e_p, σ_e_q, wmin, wmax = Params
    for i in one(epochs):epochs
    
        #=
        r = randn(typeof(prec), 1, n).*σ_e_r
        δ = randn(typeof(prec), 1, n).*σ_e_δ
        p = randn(typeof(prec), 1, n).*σ_e_p
        q = randn(typeof(prec), 1, n).*σ_e_q
        w = rand(typeof(prec), 1, n).*(wmax .- wmin) .+ wmin
        =#

        grads = Flux.gradient(m) do f
            #loss(f,r,δ,p,q,w)
            loss(f)
        end
        Flux.update!(opt, m, grads[1])
        
    end

    #return opt, m
    
end
