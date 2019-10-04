using Random

export AbstractAgent, start!, step!, get_action

abstract type AbstractAgent end

"""
    start!(agent::AbstractAgent, env_s_tp1, rng=Random.GLOBAL_RNG, kwargs...)

Function for starting the agent for a new episode.

returns an action to get passed to the environment
"""
function start!(agent::AbstractAgent, env_s_tp1; kwargs...)
    return start!(agent, env_s_tp1, Random.GLOBAL_RNG; kwargs...)
end

"""
    start!(agent::AbstractAgent, env_s_tp1, rng=Random.GLOBAL_RNG, kwargs...)
"""
function start!(agent::AbstractAgent, env_s_tp1, rng::AbstractRNG; kwargs...)
    @error "Implement start! function for agent $(typeof(agent))"
end


"""
    step!(agent::AbstractAgent, env_s_tp1, r, terminal; kwargs...)

Function to take a step with an agent. 

Returns an action to get passed to the environment.
"""
function step!(agent::AbstractAgent, env_s_tp1, r, terminal; kwarngs...)
    return step!(agent, env_s_tp1, r, terminal, Random.GLOBAL_RNG; kwargs...)
end

"""
    step!(agent::AbstractAgent, env_s_tp1, r, terminal, rng::AbstractRNG; kwargs...)
"""
function step!(agent::AbstractAgent, env_s_tp1, r, terminal, rng::AbstractRNG; kwargs...)
    @error "Implement step! function for agent $(typeof(agent))"
end

"""
    get_action(agent::AbstractAgent, state; kwargs...)

Returns an action, deterministacally.
"""
get_action(agent::AbstractAgent, state) = get_action(agent, state, Random.GLOBAL_RNG)

"""
    get_action(agent::AbstractAgent, state, rng::AbstractRNG; kwargs...)
"""
function get_action(agent::AbstractAgent, state, rng::AbstractRNG)
    @error "Implement get Action for agent"
end
