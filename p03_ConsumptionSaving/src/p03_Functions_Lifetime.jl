
##############################################################################################
# Functions
##############################################################################################

# Period utility function
function u_c(c)
    return (c.^(oftype(prec,1.0) .- γ) .- oftype(prec,1.0))./(oftype(prec,1.0) .- γ)
end

# Train the neural network
function nn_train!(opt,m,epochs)

    for i in one(epochs):epochs
        
        grads = Flux.gradient(m) do f
            loss(f)
        end
        Flux.update!(opt, m, grads[1])
        
    end

end

# Consumer policy function
function hh_policy_fn(m,r,δ,p,q,w)

    # Normalize inputs
    r_ = r./σ_e_r./oftype(prec,2.0)
    δ_ = δ./σ_e_δ./oftype(prec,2.0)
    p_ = p./σ_e_p./oftype(prec,2.0)
    q_ = q./σ_e_q./oftype(prec,2.0)
    w_ = (w .- wmin)./(wmax .- wmin).*oftype(prec,2.0) .- oftype(prec,1.0)

    # Evaluate by neural network
    x = [r_ δ_ p_ q_ w_]'
    y = m(x)

    ζ = Flux.sigmoid.(y[1,:])
    
    return ζ

end

# Objective function
function loss(m)

    # Draw initial state
    r = randn(typeof(prec), n, 1).*σ_e_r
    δ = randn(typeof(prec), n, 1).*σ_e_δ
    p = randn(typeof(prec), n, 1).*σ_e_p
    q = randn(typeof(prec), n, 1).*σ_e_q
    w = rand(typeof(prec), n, 1).*(wmax .- wmin) .+ wmin
    
    # Initial consumption and period utility
    c = hh_policy_fn(m,r,δ,p,q,w).*w
    discount_ = ones(typeof(prec),n,1)
    V0 = u_c(c)

    # Obtain lifetime utility under one draw of a shock sequence
    for t in one(Tmax):Tmax

        # Draw shocks
        e_r = randn(typeof(prec), n, 1).*σ_r
        e_δ = randn(typeof(prec), n, 1).*σ_δ
        e_p = randn(typeof(prec), n, 1).*σ_p
        e_q = randn(typeof(prec), n, 1).*σ_q

        # Update state
        r = ρ_r.*r .+ e_r
        δnext = ρ_δ.*δ .+ e_δ
        p = ρ_p.*p .+ e_p
        q = ρ_q.*q .+ e_q
        w = exp.(p).*exp.(q) .+ (w .- c).*rbar.*exp.(r)

        # Update consumption and utility values
        c = hh_policy_fn(m,r,δnext,p,q,w).*w
        discount_ = discount_.*β.*exp.(δnext .- δ)
        V0 = V0 .+ discount_.*u_c(c)
        δ = δnext

    end

    # Return average over draws (multiplied by -1 to convert to minimization problem)
    return -mean(V0)

end

