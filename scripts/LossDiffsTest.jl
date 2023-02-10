using DrWatson
@quickactivate "DLECON"

using Flux
using ReverseDiff

#Create modified Flux NN type that carries descructured parameters
struct DiffNN{M, R, P}
    model::M
    re::R
    p::P

    function DiffNN(model; p = nothing)
        _p, re = Flux.destructure(model)
        if p === nothing
            p = _p
        end
        return new{typeof(model), typeof(re), typeof(p)}(model, re, p)
    end
end

#function to compute the derivative of third output with respect to input
function diff3destruct(re, p, x)
    H = [Flux.gradient(w -> re(p)(w)[3], [y])[1][1] for y in x]
end

#Create an instance of a DiffNN
mod1 = DiffNN(Chain(Dense(1 => 32, relu),
Dense(32 => 32, relu),
Dense(32 => 3)))

#Record parameter values to check that they're updating correctly
checkP = copy(mod1.p)

#Loss function (simple aggregation of NN outputs)
loss(re, p, x) = sum(diff3destruct(re, p, x))

#Get the gradients using Reverse diff
gs = ReverseDiff.gradient(p -> loss(mod1.re, p, rand(3)), mod1.p)

opt = ADAM(0.01)

#Update the NN
Flux.Optimise.update!(opt,mod1.p,gs)

maximum(mod1.p .- checkP)