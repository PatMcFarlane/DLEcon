##############################################################################################
# Functions
##############################################################################################

# Utility
function u(c)
    return (c.^(oftype(prec,1.0) .- γ) .- oftype(prec,1.0))./(oftype(prec,1.0) .- γ)
end

# Marginal Utility
function u_c(c)
    return c.^(-γ)
end

# Inverse marginal Utility
function inv_u_c(x)
    return x.^(-oftype(prec,1.0)./γ)
end

# Wage
function Wage(K,Z = oftype(prec,1.0))
    return (oftype(prec,1.0) .- α).*Z.*(K./Lbar).^α
end

# Real return
function RealReturn(K,Z = oftype(prec,1.0))
    return α.*Z.*(K./Lbar).^(α .- oftype(prec,1.0)) .+ oftype(prec,1.0) .- δ
end

# Find nearest point in a grid
function searchsortednearest(a,x)

    idx = searchsortedfirst(a,x)
    if (idx == 1) 
        return idx
    end
    if (idx > length(a))
        return length(a)
    end
    if (a[idx] == x)
        return idx
    end
    if (abs(a[idx]-x) < abs(a[idx-1]-x))
        return idx
    else
        return idx - 1
    end

end

# Obtain steady state capital stock 
function GetSteadyStateCapital(Klow0,Khigh0,Z = oftype(prec,1.0))

    # Initialize
    K_low = Klow0
    K_high = Khigh0
    K = (K_high + K_low)./oftype(prec,2.0)
    disc_ = 1.0e10
    while disc_ > tol

        err_ = SteadyStateCapitalError(K,Z)
        disc_ = abs(err_)
        if disc_ < tol
            println("Converged! Steady state capital = $(K)")
        end
        if disc_ >= tol
            #println("err_ = $(err_)")
            if err_ > 0.0
                K_low = K 
                K = (K_high + K_low)./oftype(prec,2.0)
            end
            if err_ < 0.0
                K_high = K
                K = (K_high + K_low)./oftype(prec,2.0)
            end
        end

    end
    return K

end

# Objective function for steady-state aggregate capital 
function SteadyStateCapitalError(K,Z = oftype(prec,1.0))

    # Obtain implied steady state wage and rental rate 
    W = Wage(K,Z)
    R = RealReturn(K,Z)

    # Solve by EGM
    gg, c_star, g_low, g_high, success_flag = solve_EGM(x->log(x), x->log(x), R, W)
    Tmat = GetTransitionMatrix(gg)
    dd = GetStationaryDist(Tmat)
    K_new = dot(dd , [grid_wealth ; grid_wealth])
    return K_new .- K

end

function GetStationaryDist(T)

    #=
    lam, v = eigen(T)
    v = real(v[:,(real(lam) .== 1.0)])
    v = v ./ sum(v)
    return v
    =#

    F = eigen(T)
    eig_real = real(F.values)
    i = searchsortednearest(eig_real, 1.0)
    vec_real = real(F.vectors[:,i])
    return vec_real/sum(vec_real)

end

function GetTransitionMatrix(gg)

    # Allocate space
    T = zeros(n_empl*n_wealth,n_empl*n_wealth)

    # Construct
    for s in 1:n_wealth
        for l in 1:n_empl

            # Locate ind such that mesh_wealth[l,s] ⋵ [gg[l,ind-1],gg[l,ind]]
            ind = searchsortedfirst(gg[l,:] , mesh_wealth[l,s])
            ind = min( max(ind,2) , n_wealth )

            # Calculate weights for Young (2010) method
            pp = (mesh_wealth[l,s] .- gg[l,ind-1])./(gg[l,ind] .- gg[l,ind-1])
            pp = min( max(pp,0.0) , 1.0)
            sl = (l-1)*n_wealth

            # Input transition probabilities
            for k in 1:n_empl
                sk = (k-1)*n_wealth
                T[sk+ind,sl+s] = pp*trans_empl[k,l]
                T[sk+ind-1,sl+s] = (1.0 .- pp)*trans_empl[k,l]
            end
        end
    end
    
    return T

end

function solve_EGM(g_low0, g_high0, R, W)

    # Initialization
    a_old = similar(mesh_wealth)
    a_new = similar(mesh_wealth)
    c_new = similar(mesh_wealth)
    g_low_old = g_low0
    g_high_old = g_high0
    g_low_new(x) = log(x)
    g_high_new(x) = log(x)
    success_flag = 0

    for i in 1:max_iter

        # Solve back the Euler equation 
        a_new, c_new, g_low_new, g_high_new = euler_back(g_low_old,g_high_old,R,W,R,W)

        # Assess convergence and update or return
        if maximum(abs.(a_new .- a_old)) < tol

            #println("Converged at iteration $(i)")
            success_flag = 1
            break

        else

            a_old = copy(a_new)
            g_low_old = g_low_new
            g_high_old = g_high_new
            
        end

    end

    return a_new, c_new, g_low_new, g_high_new, success_flag

end

function euler_back(g_low, g_high, R, W, Rp, Wp)

    # Compute next-period consumption using the current guess of the saving policy
    c_prime = get_c(g_low,g_high,Rp,Wp)

    # Right-hand side of Euler equation
    rhs = β.*Rp.*(transpose(trans_empl)*u_c(c_prime))

    # Invert marginal utility to get today's consumption
    c = inv_u_c(rhs)

    # Get beginning-of-first-period assets from budget
    a = (mesh_wealth .+ c .- W.*mesh_empl)./R

    # Permutation vectors that sort assets vector in increasing order
    p1 = sortperm(a[1,:])
    p2 = sortperm(a[2,:])

    # Update saving policy function by linear interpolation using a and the a' grid 
    Kg_low = LinearInterpolation(a[1,p1], mesh_wealth[1,p1], extrapolation_bc=Line())
    Kg_high = LinearInterpolation(a[2,p2], mesh_wealth[2,p2], extrapolation_bc=Line())
    
    Kg_low_f(x) = Kg_low(x)
    Kg_high_f(x) = Kg_high(x)
    
    return a, c, Kg_low_f, Kg_high_f
    
end

function get_c(g_low,g_high,R,W)

    c_prime = similar(mesh_wealth)
    c_prime[1,:] = R.*mesh_wealth[1,:] .+ W.*mesh_empl[1,:] .- max.(a_min , g_low.(mesh_wealth[1,:]))
    c_prime[2,:] = R.*mesh_wealth[2,:] .+ W.*mesh_empl[2,:] .- max.(a_min , g_high.(mesh_wealth[2,:]))
    
    return c_prime    

end

# Solve MIT shock 
function GetMITShockSolution(K_path_0,Z_path,g_low_star,g_high_star,μ_star,K_star)

    # Initialization 
    diff_ = 1.0e10
    diff_old = 1.0e10
    convergence_flag = 0
    K_path = K_path_0
    iter_ = 1
    while diff_ > tol

        println("Iteration $(iter_)")

        # Solve backward for saving policy rules given current guess of aggregate path 
        a_path, g_low_path, g_high_path = backward_update(g_low_star,g_high_star,K_path,Z_path)

        # Solve forward for state variable paths given current guess of policy rules
        μ_path_forward, K_path_forward = forward_update(a_path, Z_path, K_star, μ_star)

        # Assess convergence and update 
        diff_ = maximum(abs.(K_path_forward .- K_path))
        if diff_ < tol
            convergence_flag = 1
        end
        if diff_ > tol
            K_path = tuner_K.*K_path_forward .+ (oftype(prec,1.0) .- tuner_K).*K_path
            iter_ = iter_+1
        end

    end

    return K_path, convergence_flag

end

# Backward update step for MIT shock 
function backward_update(g_low_star,g_high_star,K_path,Z_path)

    nT = length(Z_path)

    # Initialization 
    g_low_path = Array{Function}(undef,nT)
    g_high_path = Array{Function}(undef,nT)
    a_path = zeros(n_empl,n_wealth,nT)
    R_path = zeros(nT)
    W_path = zeros(nT)

    # Terminal conditions
    g_low_path[nT] = g_low_star
    g_high_path[nT] = g_high_star
    
    # Get price sequences
    for tt in 0:nT-1
        R_path[nT-tt] = RealReturn(K_path[nT-tt],Z_path[nT-tt]) 
        W_path[nT-tt] = Wage(K_path[nT-tt],Z_path[nT-tt])
    end

    # Solve backward for today's policy given tomorrow's 
    for tt in 1:nT-1
        a_path[:,:,nT-tt], c_new, g_low_path[nT-tt], g_high_path[nT-tt] = euler_back(g_low_path[nT-tt+1],g_high_path[nT-tt+1],R_path[nT-tt],W_path[nT-tt],R_path[nT-tt+1],W_path[nT-tt+1])
    end
    
    return a_path, g_low_path, g_high_path

end

# Forward update step for MIT shock 
function forward_update(a_path, Z_path, K_star, μ_star)

    # Initialize
    nT = length(Z_path)
    K_path_forward = zeros(typeof(prec),nT)
    μ_path_forward = zeros(size(μ_star,1),nT)

    # Initial conditions
    K_path_forward[1] = K_star
    μ_path_forward[:,1] = μ_star

    # Solve forward for states given policies
    for tt in 2:nT
        T_ = GetTransitionMatrix(a_path[:,:,tt-1])
        μ_path_forward[:,tt] = T_*μ_path_forward[:,tt-1]
        K_path_forward[tt] = dot(μ_path_forward[:,tt] , [grid_wealth ; grid_wealth])
    end

    return μ_path_forward, K_path_forward

end