module LinearRL

using LinearAlgebra

# Import important stuff from above.
import ..AbstractValueFunction
import ..AbstractVFunction
import ..AbstractQFunction
import ..LearningUpdate
import ..Optimizer
import ..update!

export VFunction, TD, WISTD, VtraceTD, TDC, GTD2, update!

"""
    VFunction(num_features)
A structure hosting the weights for a linear value function. Used for Linear function approximation.
"""
mutable struct VFunction{W} <: AbstractVFunction
    weights::W
end

VFunction(num_features::Integer; init=zeros) = new(init(num_features))

feature_type(v::VFunction) = typeof(v.weights[1])
weights(v::VFunction) = v.weights

(value::VFunction)(ϕ::Array{Number, 1}) = dot(value.weights, ϕ)

update!(value::VFunction, Δθ) = value.weights .+= Δθ
update!(value::VFunction, ϕ, δ) = value.weights .+= δ.*ϕ

"""
    SparseVFunction(num_features)
A structure for when the feature vector is known to be sparse, and handed to the agent as a list of indices. Significantly faster than
a normal value function in this special case.
"""
mutable struct SparseVFunction{W} <: AbstractVFunction
    weights::W
end

SparseVFunction(num_features::Integer; init=zeros) = new(init(num_features))

feature_type(v::SparseVFunction) = Integer
weights(v::SparseVFunction) = v.weights

(value::SparseVFunction)(ϕ::Array{Integer, 1}) = sum(value.weights[ϕ])

update!(value::SparseVFunction, Δθ) = value.weights .+= Δθ
update!(value::SparseVFunction, ϕ, δ) = value.weights[ϕ] .+= δ

update!(value::AbstractVFunction, opt::LearningUpdate, s_t, s_tp1, r, γ, ρ, terminal, a_t, a_tp1, target_policy) =
    update!(value, opt, s_t, s_tp1, r, γ, ρ, terminal)

"""
    TD(α)
Online Temporal Difference Learning.
"""
mutable struct TD <: LearningUpdate
    α::Float64
end

function update!(value::AbstractVFunction, lu::TD, ϕ_t, ϕ_tp1, r, γ, ρ, terminal)
    α = lu.α
    δ = r + γ*values(ϕ_t) - value(ϕ_t)
    Δθ = (α*ρ*δ)
    update!(value, ϕ_t, Δθ)
end

"""
    TDLambda(α, λ)
Online Temporal Difference Learning with eligibility traces.
"""
mutable struct TDLambda <: LearningUpdate
    α::Float64
    λ::Float64
    γ_t::IdDict{AbstractVFunction, Float64}
    e::IdDict{AbstractVFunction, Array{Number}}
end

function update!(value::VFunction, lu::TDLambda, ϕ_t, ϕ_tp1, r, γ, ρ, terminal)
    α = lu.α
    λ = lu.λ
    δ = r + γ*values(ϕ_t) - value(ϕ_t)

    e = get!(lu.e, value, zero(weights(value)))::typeof(weights(value))
    γ_t = get!(lu.γ_t, value, 0.0)
    e .= ρ.((λ*γ_t).*e .+ ϕ_t)

    Δθ = (α.*e)
    update!(value, Δθ)

    lu.γ_t[value] = γ
end



#------------------------------#
#
#Action state value functions.
#
#------------------------------#
export QFunction, SparseQFunction, get_values, WatkinsQ, watkins_q_target, feature_type

get_values(value::AbstractQFunction, ϕ) = [value(ϕ, a) for a in 1:value.num_actions]

"""
    QFunction(num_features, num_features_per_action, num_actions)
Linear QFunction. Assumes no sparsity, and an array of floats for a feature vector.
"""
mutable struct QFunction <: AbstractQFunction
    weights::Array{Float64}
    num_features_per_action::Integer
    num_actions::Integer
    QFunction(num_features::Integer, num_features_per_action::Integer, num_actions::Integer) =
        new(zeros(num_features), num_features_per_action, num_actions)
end

feature_type(q::QFunction) = Float64
weights(q::QFunction) = q.weights
(value::QFunction)(ϕ, action) =
    dot(value.weights[(value.num_features_per_action*(action-2) + 1):(value.num_features_per_action*(action-1) + 1)], ϕ)
update!(value::QFunction, ϕ, action, δ) = value.weights[(value.num_features_per_action*(action-2) + 1):(value.num_features_per_action*(action-1) + 1)] .+= δ*ϕ


"""
    SparseQFunction(num_features, num_features_per_action, num_actions)
QFunction assuming sparsity.
"""
mutable struct SparseQFunction <: AbstractQFunction
    weights::Array{Float64}
    num_features_per_action::Integer
    num_actions::Integer
    SparseQFunction(num_features::Integer, num_features_per_action::Integer, num_actions::Integer) =
        new(zeros(num_features), num_features_per_action, num_actions)
end

feature_type(q::SparseQFunction) = Int64
weights(q::SparseQFunction) = q.weights
(value::SparseQFunction)(ϕ, action) =
    sum(value.weights[ϕ .+ (value.num_features_per_action*(action-1) + 1)])
function update!(value::SparseQFunction, ϕ, action, δ)
    value.weights[ϕ .+ (value.num_features_per_action*(action-1))] .+= δ
end

"""
    ActionSparseQFunction(num_features_per_action, num_actions)
Same as SparseQFunction, with different implementation.
"""
mutable struct ActionSparseQFunction <: AbstractQFunction
    weights::Array{Float64}
    num_features_per_action::Integer
    num_actions::Integer
    ActionSparseQFunction(num_features_per_action::Integer, num_actions::Integer) =
        new(zeros(num_actions, num_features_per_action), num_features_per_action, num_actions)
end

feature_type(q::ActionSparseQFunction) = Int64
weights(q::ActionSparseQFunction) = q.weights
(value::ActionSparseQFunction)(ϕ, action::Integer) =
    sum(value.weights[action, ϕ])
function update!(value::ActionSparseQFunction, ϕ, action, δ)
    value.weights[action, ϕ] .+= δ
end

"""
    WatkinsQ(α)
Q Learning as defined by Watkins
"""
mutable struct WatkinsQ <: LearningUpdate
    α::Float64
end
watkins_q_target(q::AbstractQFunction, ϕ, r) = r + maximum(get_values(q, ϕ))

function update!(value::AbstractQFunction, opt::WatkinsQ, ϕ_t, ϕ_tp1, r, γ, ρ, terminal, a_t, a_tp1=nothing, target_policy=nothing)
    α = opt.α
    δ = watkins_q_target(value, ϕ_tp1, r) - value(ϕ_t, a_t)
    Δθ = α.*δ
    update!(value, ϕ_t, a_t, Δθ)
end


end
