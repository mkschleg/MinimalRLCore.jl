"""
    Episode iterators using https://github.com/JuliaML/Reinforce.jl/blob/master/src/episodes/iterators.jl as inspiration for api

"""

using Random

"""
    AbstractEpisode

# Interface
- `_start!`: This returns the initial state of the episode iteration and starts the environment and agent.
- `_step!`: 
- `_done`: 
"""
abstract type AbstractEpisode end

function Base.iterate(ep::AbstractEpisode, state = _start!(ep))
    _done(ep, state) && return nothing

    s = state[1][1]
    agent_ret = state[1][2]

    s′, r, t, agent_ret = _step!(ep, agent_ret)

    (s, agent_ret, s′, r, t), ((s′, agent_ret), state[2] + 1)
end

function run_episode!(episode::AbstractEpisode)
    num_steps = 1
    for sars ∈ ep
        num_steps += 1
    end
    total_reward(episode), num_steps
end

function run_episode!(f::Base.Callable, episode::AbstractEpisode)
    num_steps = 1
    for sars ∈ ep
        f(sars)
        num_steps += 1
    end
    total_reward(total_reward), num_steps
end

run_episode!(f::Base.Callable, env::AbstractEnvironment, agent::AbstractAgent, args...) = 
    run_episode!(f, Episode(env, agent, args...))

run_episode!(env::AbstractEnvironment, agent::AbstractAgent, args...) =
    run_episode!(ep, Episode(env, agent, args...))


"""
    Episode(env, agent, maxn, rng)

This is a struct for managing the components of an episode iterator. You should only pass a reference to env and agent, while managing the reference seperately.

# Arguments
- `env::AbstractEnvironment`: The Environment (following RLCore interface)
- `agent::AbstractAgent`: The Agent (following RLCore interface)
- `maxn=0`: Max number of steps for the episode.
- `rng=nothing`: The Random Number Generator for the episode (can either be nothing or an AbstractRNG)

"""
mutable struct Episode{E<:AbstractEnvironment, A<:AbstractAgent, F<:Number, AR}

    env::E
    agent::A
    total_reward::F # total reward of the episode

    maxn::Int       # max steps in an episode - should be constant during an episode

    rng::AR
end

Episode(env, agent, maxn=typemax(Int), rng=nothing) =
    Episode{:default}(env, agent, get_reward(env), maxn, rng)

_done(ep::Episode, state) =
    state[2] > ep.maxn || is_terminal(ep.env)

_get_action(action) = action
_get_action(agent_ret::NamedTuple) = agent_ret.action

_env_step!(ep::Episode{<:AbstractEnvironment, <:AbstractAgent, <:Number, <:AbstractRNG}, action) =
    step!(ep.env, action, ep.rng)
_env_step!(ep::Episode{<:AbstractEnvironment, <:AbstractAgent, <:Number, Nothing}, action) =
    step!(ep.env, action)
_agent_step!(ep::Episode{<:AbstractEnvironment, <:AbstractAgent, <:Number, <:AbstractRNG}, s′, r, t) =
    step!(ep.agent, s′, r, t, ep.rng)
_agent_step!(ep::Episode{<:AbstractEnvironment, <:AbstractAgent, <:Number, Nothing}, s′, r, t) =
    step!(ep.agent, s′, r, t)

function _start!(ep::Episode)
    s, agent_ret = if ep.rng isa Nothing
        s = start!(ep.env)
        agent_ret = start!(ep.agent, s)
        s, agent_ret
    else
        s = start!(ep.env, ep.rng)
        agent_ret = start!(ep.agent, s, ep.rng)
        s, agent_ret
    end

    ep.total_reward = zero(ep.total_reward)
    ep.niter = 1
    return (s, agent_ret), ep.niter
end

function _step!(ep::Episode, agent_ret)
    action = _get_action(agent_ret)

    s′, r, t = _env_step!(ep, action)
    agent_ret = _agent_step!(ep, s′, r, t)

    ep.total_reward += r
    
    s′, r, t, agent_ret
end


mutable struct DispatchEpisode{T<:Val, E<:AbstractEpisode} <: AbstractEpisode

    dispatch_type::T
    episode::E
    
end

DispatchEpisode(dispatch, episode::Episode) = DispatchEpisode(Val(dispatch), episode)
DispatchEpisode(dispatch, args...) = DispatchEpisode(Val(dispatch), Episode(args...))

_start!(ep::DispatchEpisode) = _start!(ep.episode)
_done(ep::DispatchEpisode, state) = _done(ep.episode, state)
_step!(ep::DispatchEpisode, agent_ret) = _step!(ep.episode, agent_ret)
Base.iterate(ep::DispatchEpisode, state = _start!(ep)) = Base.iterate(ep.episode, state)
