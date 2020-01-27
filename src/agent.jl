using Random

export AbstractAgent, start!, step!, get_action

abstract type AbstractAgent end

"""
    start!(agent::AbstractAgent, env_s_tp1; kwargs...)

Function for starting the agent for a new episode. If not overloaded passes GLOBAL_RNG to start!(agent::AbstractAgent, env_s_tp1, rng; kwargs...)

returns an action to get passed to the environment
"""
function start!(agent::AbstractAgent, env_s_tp1, args...)
    @error "Implement start! for $(typeof(agent))"
end


"""
    step!(agent::AbstractAgent, env_s_tp1, r, terminal; kwargs...)

Function to take a step with an agent. If not overloaded passes GLOBAL_RNG to step!(agent::AbstractAgent, env_s_tp1, r, terminal, rng; kwargs...)

Returns an action to get passed to the environment.
"""
function step!(agent::AbstractAgent, env_s_tp1, r, terminal, args...)
    @error "Implement step! for $(typeof(agent))"
end
