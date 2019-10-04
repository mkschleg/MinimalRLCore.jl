
mutable struct BatchTD <: Optimizer
    α::Float64
    α_mod::Float64
    avg::Bool
    BatchTD(α, α_mod, avg) = new(α, α_mod, avg)
end

function update!(value::VFunction, opt::BatchTD, s_t, s_tp1, r, γ, ρ, terminal)
    α = opt.α*opt.α_mod
    if opt.avg
        α = α/length(ρ)
    end
    δ = r + γ.*(dot.(s_tp1, [value.weights])) - dot.(s_t, [value.weights])
    Δθ = α*sum(ρ.*(δ.*s_t))
    value.weights .+= Δθ
end

mutable struct BatchTDC <: Optimizer
    α::Float64
    β::Float64
    α_mod::Float64
    β_mod::Float64
    avg::Bool
    h::IdDict
    # TDC(α, β) = new(α, β, 1.0, 1.0, true)
    # TDC(α, β, α_mod, β_mod, avg) = new(α, β, α_mod, β_mod, avg)
end

function update!(value::VFunction, opt::BatchTDC, s_t, s_tp1, r, γ, ρ, terminal)
    # α = TD.α*TD.α_mod
    # β = TD.β*TD.β_mod

    h = get!(opt.h, value, zero(value.weight))

    α = opt.α*opt.α_mod
    β = opt.β*opt.β_mod
    if opt.avg
        α = α/length(ρ)
        β = β/length(ρ)
    end

    δ = r + γ.*(dot.(s_tp1, [value.weights])) - dot.(s_t, [value.weights])
    Δθ = α*sum(ρ.*(δ.*s_t - γ.*dot.(s_t, [value.h]).*s_tp1))
    Δh = β*sum(s_t.*(ρ.*δ - dot.(s_t, [value.h])))

    value.weights .+= Δθ
    value.h .+= Δh
end

mutable struct WISBatchTDC <: Optimizer
    α::Float64
    β::Float64
    α_mod::Float64
    β_mod::Float64
    WISTDC(α, β) = new(α, β, 1.0, 1.0)
end

function update!(value::VFunction, opt::WISBatchTDC, s_t, s_tp1, r, γ, ρ, terminal)
    δ = r + γ.*(dot.(s_tp1, [value.weights])) - dot.(s_t, [value.weights])
    Δθ = α*sum(ρ.*(δ.*s_t - γ.*dot.(s_t, [value.h]).*s_tp1))
    Δh = β*sum(s_t.*(ρ.*δ - dot.(s_t, [value.h])))

    value.weights .+= Δθ./sum(ρ)
    value.h .+= Δh./sum(ρ)
end

mutable struct BatchTDC2 <: Optimizer
    α::Float64
    β::Float64
    α_mod::Float64
    β_mod::Float64
    avg::Bool
    TDC2(α, β) = new(α, β, 1.0, 1.0, true)
end

function update!(value::VFunction, opt::BatchTDC2, s_t, s_tp1, r, γ, ρ, terminal)
    # Python code sample = (prev_phi.copy(), phi.copy(), action, reward, state, prev_state, pi[action]/mu[action])
    # td_error = (sample[3] + gamma*(sample[1].dot(weights_tdc_iwer)) - sample[0].dot(weights_tdc_iwer))
    # weights_tdc_iwer = weights_tdc_iwer + alpha*sample[-1]*(td_error*sample[0] - gamma*sample[1]*(sample[0].dot(h_tdc_iwer)))
    # h_tlndc_iwer = h_tdc_iwer + alpha_h*(sample[-1]*td_error - sample[0].dot(h_tdc_iwer))*sample[0]
    α = opt.α*opt.α_mod
    β = opt.β*opt.β_mod
    if opt.avg
        α = α/length(ρ)
        β = β/length(ρ)
    end

    δ = r + γ.*(dot.(s_tp1, [value.weights])) - dot.(s_t, [value.weights])
    Δθ = α*sum(ρ.*(δ.*s_t - γ.*dot.(s_t, [value.h]).*s_tp1))
    Δh = β*sum(s_t.*(ρ.*(δ - dot.(s_t, [value.h]))))

    value.weights .+= Δθ
    value.h .+= Δh
end

mutable struct BatchGTD2 <: Optimizer
    α::Float64
    β::Float64
    α_mod::Float64
    β_mod::Float64
    avg::Bool
    GTD2(α, β) = new(α, β, 1.0, 1.0, true)
    GTD2(α, β, α_mod, β_mod, avg) = new(α, β, α_mod, β_mod, avg)
end

function update!(value::VFunction, opt::BatchGTD2, s_t, s_tp1, r, γ, ρ, terminal)
    # Python code sample = (prev_phi.copy(), phi.copy(), action, reward, state, prev_state, pi[action]/mu[action])
    # td_error = (sample[3] + gamma*(sample[1].dot(weights_gtd2_iwer)) - sample[0].dot(weights_gtd2_iwer))
    # weights_gtd2_iwer = weights_gtd2_iwer + alpha*sample[-1]*(sample[0] - gamma*sample[1])*(sample[0].dot(h_gtd2_iwer))
    # h_gtd2_iwer = h_gtd2_iwer + alpha_h*(sample[-1]*td_error - sample[0].dot(h_gtd2_iwer))*sample[0]
    α = opt.α*opt.α_mod
    β = opt.β*opt.β_mod
    if opt.avg
        α = α/length(ρ)
        β = β/length(ρ)
    end

    δ = r + γ.*(dot.(s_tp1, [value.weights])) - dot.(s_t, [value.weights])
    Δθ = α*sum(ρ.*(s_t - γ.*s_tp1).*(dot.(s_t, [value.h])))
    Δh = β*sum(s_t.*(ρ.*δ - dot.(s_t, [value.h])))

    value.weights .+= Δθ
    value.h .+= Δh
end
