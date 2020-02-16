using Random

export AbstractAgent, start!, step!, get_action

abstract type AbstractAgent end

"""
    start!(agent::AbstractAgent, env_s_tp1, args...)

Function for starting the agent for a new episode. 

returns an action to get passed to the environment
"""
function start!(agent::AbstractAgent, env_s_tp1, args...)
    @error "Implement start! for $(typeof(agent))"
end


"""
    step!(agent::AbstractAgent, env_s_tp1, r, terminal, args...)

Function to take a step with an agent.

Returns an action to get passed to the environment.
"""
function step!(agent::AbstractAgent, env_s_tp1, r, terminal, args...)
    @error "Implement step! for $(typeof(agent))"
end
