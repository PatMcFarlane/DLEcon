##############################################################################################
# Functions
##############################################################################################

function MakeShockSequences(T,N,ar1share = 0.5,range_ = 2.0)

    # This creates a set of log productivity sequences.
    # Some are AR(1) with randomly chosen persistence.
    # Others are temporary level shifts.
    # Code assumes T >> 20

    # Initialize
    Shocks = zeros(T,N)

    for i in 1:N
        x_ = rand()
        Shocks_ = zeros(T,1)
        if x_ < ar1share
            ρ_ = rand()*0.95
            std_ = (-range_ .+ rand().*(2.0.*range_)).*σ
            t0_ = oftype(1,ceil(10.0*rand()))
            Shocks_ = Shocks_ .+ [zeros(t0_-1,1) ; randn()*std_ ; zeros(T-t0_,1)]
            for tt in t0_+1:T
                Shocks_ = Shocks_ .+ [zeros(tt-1,1) ; ρ_*Shocks_[tt-1,1] ; zeros(T-tt,1)]
            end
            Shocks = Shocks .+ [zeros(T,i-1) Shocks_ zeros(T,N-i)] 
        end
        if x_ >= ar1share
            std_ = (-range_ .+ rand().*(2.0.*range_)).*σ
            t0_ = oftype(1,ceil(10.0*rand()))
            dur_ = oftype(1,ceil(20*rand()))
            Shocks_ = Shocks_ .+ [zeros(t0_-1,1) ; fill(std_,dur_,1) ; zeros(T-(t0_+dur_-1),1)]
            Shocks = Shocks .+ [zeros(T,i-1) Shocks_ zeros(T,N-i)] 
        end
    end

    return Shocks

end

# Obtain sequence space solution path by conventional means 
function Shooting(Shocks)

    T, N = size(Shocks)
    Cmat = zeros(T,N)
    Kmat = zeros(T,N)
    
    for i in 1:N

        Z = Shocks[:,i]

        # Allocate space
        C = fill(Css,T,1)
        K = fill(Kss,T,1)
        
        # Initial guess:
        # Create initial bounds for bisection algorithm
        goodbound_ = 0
        Clow = Css/1.5
        while goodbound_ == 0

            C[1] = Clow
            for tt = 2:Tmax

                K[tt] = max(exp(Z[tt-1]).*K[tt-1].^α .+ (1.0 .- δ).*K[tt-1] - C[tt-1] , 0.01)
                C[tt] = max((β.*C[tt-1].^γ.*(1.0 .- δ .+ α.*exp.(Z[tt]).*K[tt].^(α .- 1.0))).^(1.0 ./ γ), 0.01)

            end

            # Update guess
            disc_ = K[Tmax] - Kss
            if disc_ > 0.0
                Clow = C[1]
                goodbound_ = 1
            end
            if disc_ <= 0.0
                Clow = Clow/1.5
            end

        end
        goodbound_ = 0
        Chigh = Css*1.5
        while goodbound_ == 0

            C[1] = Chigh
            for tt = 2:Tmax

                K[tt] = max(exp(Z[tt-1]).*K[tt-1].^α .+ (1.0 .- δ).*K[tt-1] - C[tt-1] , 0.01)
                C[tt] = max((β.*C[tt-1].^γ.*(1.0 .- δ .+ α.*exp.(Z[tt]).*K[tt].^(α .- 1.0))).^(1.0 ./ γ), 0.01)

            end

            # Update guess
            disc_ = K[Tmax] - Kss
            if disc_ < 0.0
                Chigh = C[1]
                goodbound_ = 1
            end
            if disc_ >= 0.0
                Chigh = Chigh*1.5
            end

        end

        # Iterate to convergence
        C = fill(Css,T,1)
        K = fill(Kss,T,1)
        
        absdisc_ = 1.0e10
        while absdisc_ > tol

            for tt = 2:Tmax

                K[tt] = max(exp(Z[tt-1]).*K[tt-1].^α .+ (1.0 .- δ).*K[tt-1] - C[tt-1] , 0.01)
                C[tt] = max((β.*C[tt-1].^γ.*(1.0 .- δ .+ α.*exp.(Z[tt]).*K[tt].^(α .- 1.0))).^(1.0 ./ γ), 0.01)

            end

            # Update bounds for guess
            disc_ = K[Tmax] - Kss
            if disc_ > 0.0
                Clow = C[1]
                C[1] = (Chigh+Clow)/2.0
            end
            if disc_ < 0.0
                Chigh = C[1]
                C[1] = (Chigh+Clow)/2.0
            end

            absdisc_ = abs(disc_)
            
        end

        Cmat[:,i] = C
        Kmat[:,i] = K

    end

    return Cmat, Kmat

end

# Get K sequence from C sequence and initial condition
function GetK(C,Z,K0 = Kss)

    T, N = size(C)
    K = zeros(T,N)
    K[1,:] = fill(K0,1,N)       # Initial condition
    for tt in 2:T
        
        K[tt,:] = exp.(Z[tt-1,:]).*K[tt-1,:].^α .+ (1.0 .- δ).*K[tt-1,:] .- C[tt-1,:]
        if minimum(K[tt,:]) < 0.0
            K[tt,:] = max.(exp.(Z[tt-1,:]).*K[tt-1,:].^α .+ (1.0 .- δ).*K[tt-1,:] .- C[tt-1,:] , 0.01)
        end
    end

    return K

end

# Get C sequence from K sequence and terminal condition
function GetC(K,Z,CT = Css)

    T, N = size(K)
    C = zeros(T,N)
    C[T,:] = fill(CT,1,N)       # Terminal condition
    for tt in 1:T-1
        C[T-tt,:] = (β.*C[T-tt+1,:].^(-γ).*(1.0 .- δ .+ α.*exp.(Z[T-tt+1,:]).*K[T-tt+1,:].^(α .- 1.0))).^(-1.0 ./ γ)
    end

    return C

end

# Loss function for Approach B1
function loss_b1(m)

    # Draw shock sequence(s) and evaluate neural network
    Shocks = MakeShockSequences(Tmax,Ndraws)
    mx = m(Shocks)
    C_ = mx[1:Tmax-1,:]
    K_ = mx[Tmax-1+1:Tmax-1+Tmax-1,:]

    # Add initial conditions
    C = [C_ ; fill(Css,1,Ndraws)]
    K = [fill(Kss,1,Ndraws) ; K_]

    # Compute Euler equation and transition law errors
    e_1 = β.*C[1:Tmax-1,:].^γ.*(1.0 .- δ .+ α.*exp.(Shocks[2:Tmax,:]).*K[2:Tmax,:].^(α .- 1.0)) .- C[2:Tmax,:].^γ
    e_2 = exp.(Shocks[1:Tmax-1,:]).*K[1:Tmax-1,:].^α .+ (1.0 .- δ).*K[1:Tmax-1,:] .- C[1:Tmax-1,:] .- K[2:Tmax,:]

    # Compute loss and return
    return sum(wgt_euler.*e_1.*e_1 .+ wgt_lom.*e_2.*e_2)

end

# Training function
function nn_train!(opt,m,loss,epochs)
    @showprogress for i in 1:epochs
        grads = Flux.gradient(f -> loss(f), m)
        Flux.update!(opt, m, grads[1])
    end
end

# Get maximum error 
function GetMaxError(err)

    vals, ind = findmax(abs.(err), dims = 1)
    return err[ind]

end





