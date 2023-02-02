using DrWatson
@quickactivate "DLECON"

using Flux

struct TestParas
    expon::Float64
end

TP = TestParas(2.0)

mod1 = Dense(2,1)

function DataGen1(model, TP)

    data = ones(2)

    resid = sum(model(data).^TP.expon)

    return resid

end

grads = Flux.gradient(m -> DataGen1(m,TP), mod1)

2.0.*(mod1.weight*ones(5) .+ mod1.bias).*ones(2)

@code_warntype Flux.gradient(m -> DataGen1(m,TP), mod1)


function DataGen2(model, expon)

    data = ones(2)

    resid = sum(model(data).^expon)

    return resid

end

grads = Flux.gradient(m -> DataGen2(m,2.0), mod1)

@code_warntype Flux.gradient(m -> DataGen2(m,2.0), mod1)

const expon_const = 2.0

grads = Flux.gradient(m -> DataGen2(m,expon_const), mod1)

@code_warntype Flux.gradient(m -> DataGen2(m,expon_const), mod1)