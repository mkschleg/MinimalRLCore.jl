
using Random: AbstractRNG


"""
Represents an abstract environment for reinforcement learning agents. Has several functions that need to be implemented to work
"""
abstract type AbstractEnvironment end


"""
    start(env::AbstractEnvironment; rng::AbstractRNG, kwargs...)
Function to retrieve a started environment struct without effecting the current environment struct. Returns an environment of the same type.
"""
function start(env::AbstractEnvironment; rng::AbstractRNG=Random.GLOBAL_RNG, kwargs...)
    new_env = copy(env)
    return start!(new_env; rng=rng, kwargs...)
end

"""
    start!(env::AbstractEnvironment; rng::AbstractRNG, kwargs...)

Function to start the passed environment `env`.
"""
function start!(env::AbstractEnvironment; rng::AbstractRNG=Random.GLOBAL_RNG, kwargs...)
    reset!(env; rng=rng, kwargs...)
    return env, get_state(env)
end


"""
   step!(env::AbstractEnvironment, action; rng::AbstractRNG, kwargs...)

Update the state of the environment based on the underlying dynamics and the action.
"""
function step(env::AbstractEnvironment, action; rng::AbstractRNG=Random.GLOBAL_RNG, kwargs...)
    new_env = copy(env)
    return step!(new_env, action; kwargs...)
end

"""
   step!(env::AbstractEnvironment, action; rng::AbstractRNG, kwargs...)

Update the state of the passed environment `env` based on the underlying dynamics and the action.
"""
function step!(env::AbstractEnvironment, action; rng::AbstractRNG=Random.GLOBAL_RNG, kwargs...) # -> env, state, reward, terminal
    environment_step!(env, action; rng=rng, kwargs...)
    return env, get_state(env), get_reward(env), is_terminal(env)
end


#---------------------------#
#   Interface functions
#---------------------------#


"""
    reset!(env::AbstractEnvironment; rng::AbstractRNG, kwargs...)

Reset the environment to initial conditions based on the random number generator.
"""
function reset!(env::AbstractEnvironment; rng::AbstractRNG=Random.GLOBAL_RNG, kwargs...)
    throw("Implement reset! for environment $(typeof(env))")
end

"""
    environment_step!(env::AbstractEnvironment, action; rng::AbstractRNG, kwargs...)

Update the state of the environment based on the underlying dynamics and the action. This is not used directly, but through the step function.
"""
function environment_step!(env::AbstractEnvironment, action; rng::AbstractRNG=Random.GLOBAL_RNG, kwargs...)
    throw("Implement environment_step for environment $(typeof(env))")
end

"""
    get_reward(env::AbstractEnvironment)

Retrieve reward for the current state of the environment.
"""
function get_reward(env::AbstractEnvironment) # -> determines if the agent_state is terminal
    throw("Implement get_reward for environment $(typeof(env))")
end

"""
    is_terminal(env::AbstractEnvironment)

Check if the environment is in a terminal state
"""
function is_terminal(env::AbstractEnvironment) # -> determines if the agent_state is terminal
    throw("Implement is_terminal for environment $(typeof(env))")
end

"""
    get_state(env::AbstractEnvironment)

Retrieve the current state of the environment
"""
function get_state(env::AbstractEnvironment) # -> determines if the agent_state is terminal
    throw("Implement get_state for environment $(typeof(env))")
end

"""
    get_actions(env::AbstractEnvironment)

Returns the set of actions available to take.
"""
function get_actions(env::AbstractEnvironment)
    return Set()
end

"""
    Base.show(io::IO, env::AbstractEnvironment)

Convert the environment to ascii and print to the given io.
"""
# function Base.show(io::IO, env::AbstractEnvironment)
#     # println("Implement Base.show for environment $(typeof(env))")
# end

"""
    render(env::AbstractEnvironment, args...; kwargs...)

Render the environment. (WIP, only works with Gym currently.)
"""
function render(env::AbstractEnvironment, args...; kwargs...)
    println("Render not implemented for environment $(typeof(env))")
end


include("environments/MountainCar.jl")


# end
