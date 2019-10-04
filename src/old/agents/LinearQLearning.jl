
module LinearAgents

import ..JuliaRL: step!, start!, get_action

import ..AbstractAgent

import ..FeatureCreators
import ..Learning

import ..FeatureCreators: AbstractFeatureCreator, create_features, feature_size
import ..Learning.LinearRL: AbstractQFunction, update!, Optimizer, LearningUpdate, feature_type
import ..AbstractQPolicy
import ..get

using Random

export LinearQAgent, TileCoderAgent, get_action, start!, step!

mutable struct LinearQAgent{QT<:AbstractQFunction, FC<:AbstractFeatureCreator, P<:AbstractQPolicy, T<:Number} <: AbstractAgent
    Q::QT
    fc::FC
    π::P
    γ::Float64
    lu::LearningUpdate
    ϕ_t::Array{T, 1}
    ϕ_tp1::Array{T, 1}
    action::Int64
    LinearQAgent(Q::QT, fc::FC, π::P, γ::Float64, lu::LearningUpdate) where{T<:Number, QT<:AbstractQFunction, FC<:AbstractFeatureCreator, P<:AbstractQPolicy} =
        new{QT, FC, P, feature_type(Q)}(Q, fc, π, γ, lu, zeros(feature_type(Q), feature_size(fc)), zeros(feature_type(Q), feature_size(fc)), 0)
end

get_action(agent::LinearQAgent, ϕ; rng=Random.GLOBAL_RNG, kwargs...) = get(agent.π, [agent.Q(ϕ, a) for a = agent.π.actions]; rng=rng)

function start!(agent::LinearQAgent, env_s_tp1; rng=Random.GLOBAL_RNG, kwargs...)

    agent.ϕ_t .= create_features(agent.fc, env_s_tp1)
    agent.action = get_action(agent, agent.ϕ_t; rng=rng)

    return agent.action
end

function step!(agent::LinearQAgent, env_s_tp1, r, terminal; rng=Random.GLOBAL_RNG, kwargs...)

    agent.ϕ_tp1 .= create_features(agent.fc, env_s_tp1)

    update!(
        agent.Q,
        agent.lu,
        agent.ϕ_t,
        agent.ϕ_tp1,
        r,
        agent.γ,
        1.0,
        terminal,
        agent.action)

    agent.ϕ_t .= agent.ϕ_tp1
    agent.action = get_action(agent, agent.ϕ_t; rng=rng)
    return agent.action
end


function TileCoderAgent(opt::LearningUpdate,
                        size_env_state::Integer,
                        num_actions::Integer,
                        tilings::Integer,
                        tiles::Integer,
                        γ::Float64,
                        policy::AbstractQPolicy)

    fc = FeatureCreators.TileCoder(tilings, tiles, size_env_state; wrap=false, wrapwidths=0.0)
    num_features_per_action = (tilings*(tiles+1)^size_env_state)
    num_features = num_features_per_action*num_actions
    Q = Learning.LinearRL.ActionSparseQFunction(
        num_features_per_action,
        num_actions)
    return LinearQAgent(Q, fc, policy, γ, opt)
end

end
