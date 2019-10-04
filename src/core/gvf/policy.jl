
using StatsBase


"""
    AbstractPolicy
"""
abstract type AbstractPolicy end

function Base.get(π::AbstractPolicy, state_t, action_t)
    throw(DomainError("get(PolicyType, args...) not defined!"))
end

"""
    NullPolicy <: AbstractPolicy

Always returns 1.0
"""
struct NullPolicy <: AbstractPolicy
end
Base.get(π::NullPolicy, state_t, action_t) = 1.0

"""
    PersistentPolicy <: AbstractPolicy

Returns 1 of `action_t == action`
"""
struct PersistentPolicy <: AbstractPolicy
    action::Int64
end

Base.get(π::PersistentPolicy, state_t, action_t) =
    π.action == action_t ? 1 : 0

"""
    RandomPolicy{T<:AbstractFloat} <: AbstractPolicy


"""
struct RandomPolicy{T<:AbstractFloat} <: AbstractPolicy
    probabilities::Array{T,1}
    weight_vec::Weights{T, T, Array{T, 1}}
    RandomPolicy(probabilities::Array{T,1}) where {T<:AbstractFloat} = new{T}(probabilities, Weights(probabilities))
end

Base.get(π::RandomPolicy, state_t, action_t) =
    π.probabilities[action_t]

StatsBase.sample(π::RandomPolicy) = StatsBase.sample(Random.GLOBAL_RNG, π)
StatsBase.sample(rng::Random.AbstractRNG, π::RandomPolicy) = StatsBase.sample(rng, π.weight_vec)
StatsBase.sample(rng::Random.AbstractRNG, π::RandomPolicy, state) = StatsBase.sample(rng, π.weight_vec)

"""
    FunctionalPolicy{F <: Function} <: AbstractPolicy

"""
struct FunctionalPolicy{F <: Function} <: AbstractPolicy
    func::F
end

Base.get(π::FunctionalPolicy, state_t, action_t) = 
    π.func(state_t, action_t)
