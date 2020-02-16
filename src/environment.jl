

"""
Represents an abstract environment for reinforcement learning agents. Has several functions that need to be implemented to work. 
All interfaces expect an abstract environment!
"""
abstract type AbstractEnvironment end


"""
    start!(env::AbstractEnvironment, args...)

Function to start the passed environment `env`. There are three variants. Two which start the environment from a random start state (as implemented with reset!) and another which starts the environment from a provided start state. These three variants call the `reset!` functions of the same call signiture.

Returns the starting state of the environment.
"""
function start!(env::AbstractEnvironment, args...)
    reset!(env, args...)
    return get_state(env)
end


"""
    step!(env::AbstractEnvironment, action, args...)

Update the state of the passed environment `env` based on the underlying dynamics and the action.
"""
function step!(env::AbstractEnvironment, action, args...) # -> env, state, reward, terminal
    environment_step!(env, action, args...)
    return get_state(env), get_reward(env), is_terminal(env)
end



#---------------------------#
#   Interface functions
#---------------------------#
"""
    reset!(env::AbstractEnvironment, args...)

Reset the environment to initial conditions based on the random number generator.

You can implement with or without a personally defined RNG. If you choose to not implement with a personally maintained RNG remember this is not a thread safe function.
"""
function reset!(env::AbstractEnvironment, args...)
    # reset!(env::AbstractEnvironment, Random.GLOBAL_RNG; kwargs...)
    @error "Implement reset! for environment $(typeof(env))"
end


"""
    environment_step!(env::AbstractEnvironment, action, args...)

Update the state of the environment based on the underlying dynamics and the action. This is not used directly, but through the step function.

You can implement with or without a personally defined RNG. If you choose to not implement with a personally maintained RNG remember this is not a thread safe function.
"""
function environment_step!(env::AbstractEnvironment, action, args...)
    @error "Implement environment_step for environment $(typeof(env))"
end

"""
    get_reward(env::AbstractEnvironment)

Retrieve reward for the current state of the environment.
"""
function get_reward(env::AbstractEnvironment) # -> determines if the agent_state is terminal
    @error "Implement get_reward for environment $(typeof(env))"
end


"""
    is_terminal(env::AbstractEnvironment)

Check if the environment is in a terminal state
"""
function is_terminal(env::AbstractEnvironment) # -> determines if the agent_state is terminal
    @error "Implement is_terminal for environment $(typeof(env))"
end


"""
    get_state(env::AbstractEnvironment)

Retrieve the current state of the environment
"""
function get_state(env::AbstractEnvironment) # -> determines if the agent_state is terminal
    @error "Implement get_state for environment $(typeof(env))"
end


"""
    get_actions(env::AbstractEnvironment)

Returns the set of actions available to take.
"""
function get_actions(env::AbstractEnvironment)
    return Set()
end
