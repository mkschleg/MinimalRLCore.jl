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
- `total_reward`: Return the total cummulative reward for an episode.
"""
abstract type AbstractEpisode end

function Base.iterate(ep::AbstractEpisode, state = _start!(ep))
    _done(ep, state) && return nothing

    s = state[1][1]
    agent_ret = state[1][2]

    s′, r, t, agent_ret = _step!(ep, agent_ret)

    (s, agent_ret, s′, r, t), ((s′, agent_ret), state[2] + 1)
end

function run_episode!(ep::AbstractEpisode)
    num_steps = 0
    for sars ∈ ep
        num_steps += 1
    end
    total_reward(ep), num_steps
end

function run_episode!(f::Base.Callable, ep::AbstractEpisode)
    num_steps = 0
    for sars ∈ ep
        f(sars)
        num_steps += 1
    end
    total_reward(ep), num_steps
end

run_episode!(f::Base.Callable, env::AbstractEnvironment, agent::AbstractAgent, args...) = 
    run_episode!(f, Episode(env, agent, args...))

run_episode!(env::AbstractEnvironment, agent::AbstractAgent, args...) =
    run_episode!(Episode(env, agent, args...))


"""
    Episode(env, agent, maxn, rng)

This is a struct for managing the components of an episode iterator. You should only pass a reference to env and agent, while managing the reference seperately.

# Arguments
- `env::AbstractEnvironment`: The Environment (following RLCore interface)
- `agent::AbstractAgent`: The Agent (following RLCore interface)
- `maxn=0`: Max number of steps for the episode.
- `rng=nothing`: The Random Number Generator for the episode (can either be nothing or an AbstractRNG)

"""
mutable struct Episode{E<:AbstractEnvironment, A<:AbstractAgent, F<:Number, AR} <: AbstractEpisode

    env::E
    agent::A
    total_reward::F # total reward of the episode

    maxn::Int       # max steps in an episode - should be constant during an episode
    
    rng::AR
end

Episode(env, agent, maxn=typemax(Int), rng=nothing) =
    Episode(env, agent, get_reward(env), maxn, rng)

total_reward(ep::Episode) = ep.total_reward

_done(ep::Episode, state) =
    state[2] > ep.maxn || is_terminal(ep.env)

_get_action(action) = action
_get_action(agent_ret::NamedTuple) = agent_ret.action

function _env_step!(ep::Episode, action)
    if ep.rng isa Nothing
        step!(ep.env, action)
    else
        step!(ep.env, action, ep.rng)        
    end
end

function _agent_step!(ep::Episode, s′, r, t)
    if ep.rng isa Nothing
        step!(ep.agent, s′, r, t)
    else
        step!(ep.agent, s′, r, t, ep.rng)
    end
end

function _agent_end!(ep::Episode, s′, r)
    if ep.rng isa Nothing
        end!(ep.agent, s′, r)
    else
        end!(ep.agent, s′, r, ep.rng)
    end
end

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
    
    return (s, agent_ret), 1
end

function _step!(ep::Episode, agent_ret)
    action = _get_action(agent_ret)

    s′, r, t = _env_step!(ep, action)
    agent_ret = if t
        _agent_step!(ep, s′, r, t)
    else
        _agent_end!(ep, s′, r)
    end

    ep.total_reward += r
    
    s′, r, t, agent_ret
end

