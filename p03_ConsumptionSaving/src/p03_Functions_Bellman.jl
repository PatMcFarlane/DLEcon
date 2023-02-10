

##############################################################################################
# Functions
##############################################################################################

# Fischer-Burmeister
function Ψ_fb(a,b) 
    return a .+ b .- sqrt.(a.^oftype(prec,2.0) .+ b.^oftype(prec,2.0))
end

# Derivative of value function
function dv_dw(model,x,dtype)

    sqrt_eps = sqrt(eps(typeof(prec)))
    #sqrt_eps = oftype(prec,1.0e-2)
    
    n_ = size(x)[2]

    if dtype == "backward"
        dv = (model(x) .- model(x .- [zeros(typeof(prec),4,n_) ; sqrt_eps.*ones(typeof(prec),1,n_)]))./(sqrt_eps)
    end
    if dtype == "midpoint"
        dv = (model(x .+ [zeros(typeof(prec),4,n_) ; sqrt_eps.*ones(typeof(prec),1,n_)]) .- model(x .- [zeros(typeof(prec),4,n_) ; sqrt_eps.*ones(typeof(prec),1,n_)]))./(oftype(prec,2.0).*sqrt_eps)
    end
    if dtype == "forward"
        dv = (model(x .+ [zeros(typeof(prec),4,n_) ; sqrt_eps.*ones(typeof(prec),1,n_)]) .- model(x))./(sqrt_eps)
    end
    return dv[3,:]

end

# Period utility
function u_c(c)
    return (c.^(oftype(prec,1.0) .- γ) .- oftype(prec,1.0))./(oftype(prec,1.0) .- γ)
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
    x = [r_ ; δ_ ; p_ ; q_ ; w_]
    y = m(x)

    ζ = Flux.sigmoid.(y[1,:])'
    h = exp.(y[2,:])'
    v = y[3,:]'

    return ζ, h, v

end

function hh_policy_fn(m,r,δ,p,q,w,dtype)

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
    v = y[3,:]'

    dv = dv_dw(m,x,dtype)'
    
    return ζ , h, v, dv

end

# Realized optimality condition residuals
function hh_residuals(m,e_r,e_δ,e_p,e_q,r,δ,p,q,w)

    # Today's choices
    #ζ, h, v = hh_policy_fn(m,r,δ,p,q,w)
    ζ, h, v, dv = hh_policy_fn(m,r,δ,p,q,w,dtype)
    c = ζ.*w

    # Exogenous state transitions conditional on specified shocks
    rnext = ρ_r.*r .+ e_r
    δnext = ρ_δ.*δ .+ e_δ
    pnext = ρ_p.*p .+ e_p
    qnext = ρ_q.*q .+ e_q

    # Endogenous state transition
    wnext = exp.(pnext).*exp.(qnext) .+ (w .- c).*rbar.*exp.(rnext)

    # Tomorrow's choices
    ζnext, hnext, vnext, dvnext = hh_policy_fn(m,rnext,δnext,pnext,qnext,wnext,dtype)
    cnext = ζnext.*wnext

    # Residuals
    R1 = v .- u_c(c) .- β.*exp.(δnext .- δ).*vnext                      # Bellman equation error
    R2 = Ψ_fb(oftype(prec,1.0) .- h, oftype(prec,1.0) .- ζ)             # Kuhn-Tucker condition error
    R3 = β.*rbar.*exp.(δnext .- δ .+ rnext).*(c.^γ).*dvnext .- h        # Definition of normalized KT multiplpier

    #R4 = (oftype(prec,1.0) .- h).*(dvnext .- cnext.^(-γ))                              # Envelope condition error for next period if constraint is slack (i.e., h = 1)
    #R4 = (oftype(prec,1.0) .- h).*(dv .- c.^(-γ))    

    return R1, R2, R3
    #return R1, R2, R3, R4

end

# Objective function
function loss(m)

    
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
    R1_e1, R2_e1, R3_e1 = hh_residuals(m,e1_r,e1_δ,e1_p,e1_q,r,δ,p,q,w)
    R1_e2, R2_e2, R3_e2 = hh_residuals(m,e2_r,e2_δ,e2_p,e2_q,r,δ,p,q,w)
    #R1_e1, R2_e1, R3_e1, R4_e1 = hh_residuals(m,e1_r,e1_δ,e1_p,e1_q,r,δ,p,q,w)
    #R1_e2, R2_e2, R3_e2, R4_e2 = hh_residuals(m,e2_r,e2_δ,e2_p,e2_q,r,δ,p,q,w)

    # All-in-one expectation
    Rsq = wgt_R1.*R1_e1.*R1_e2 .+ wgt_R2.*R2_e1.*R2_e2 .+ wgt_R3.*R3_e1.*R3_e2
    #Rsq = wgt_R1.*R1_e1.*R1_e2 .+ wgt_R2.*R2_e1.*R2_e2 .+ wgt_R3.*R3_e1.*R3_e2 .+ R4_e1.*R4_e2

    # Return average over draws 
    return mean(Rsq)

end

function get_equation_squared_errors(m)

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
    R1_e1, R2_e1, R3_e1 = hh_residuals(m,e1_r,e1_δ,e1_p,e1_q,r,δ,p,q,w)
    R1_e2, R2_e2, R3_e2 = hh_residuals(m,e2_r,e2_δ,e2_p,e2_q,r,δ,p,q,w)

    # Mean squared equation errors
    Rsq1 = mean(R1_e1.*R1_e2)
    Rsq2 = mean(R2_e1.*R2_e2) 
    Rsq3 = mean(R3_e1.*R3_e2)

    # Return average over draws 
    return Rsq1, Rsq2, Rsq3

end


# Train the neural network
function nn_train!(opt,m,epochs)

    #vals1 = typeof(prec)[]
    #vals2 = typeof(prec)[]
    #vals3 = typeof(prec)[]
    
    for i in one(epochs):epochs
        #Rsq1, Rsq2, Rsq3 = get_equation_squared_errors(m)
        #push!(vals1,Rsq1)
        #push!(vals2,Rsq2)
        #push!(vals3,Rsq3)
        grads = Flux.gradient(m) do f
            loss(f)
        end
        Flux.update!(opt, m, grads[1])
        
    end

    #return vals1, vals2, vals3

end

