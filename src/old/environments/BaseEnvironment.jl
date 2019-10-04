
using Random

module ExampleEnvConst
const state_initial_range = 1:10
end

mutable struct ExampleEnvironment <: AbstractEnvironment
    some_variable::Float64
    state::Int64
    normalized_state::Bool
    ExampleEnvironment(rng=Random.GLOBAL_RNG) = new(rand(rng), rand(rng, ExampleEnvConst.state_initial_range), false)
end

function reset!(env::ExampleEnvironment; rng = Random.GLOBAL_RNG, kwargs...)
    env.some_variable = rand(rng)
    env.state = rand(rng, ExampleEnvConst.state_initial_range)
end

function get_actions(env::AbstractEnvironment)
    return Set()
end

function environment_step!(env::ExampleEnvironment, action::Int64; rng = Random.GLOBAL_RNG, kwargs...)
    env.state = rand(rng, ExampleEnvConst.state_initial_range)
    env.some_variable = env.state + randn(rng)
end

function get_reward(env::ExampleEnvironment) # -> determines if the agent_state is terminal
    return -1
end

function get_state(env::ExampleEnvironment) # -> get state of agent
    if env.normalized_state
        return get_normalized_state(env)
    else
        return [env.state]
    end
end

function is_terminal(env::ExampleEnvironment) # -> determines if the agent_state is terminal
    if env.state == 1
        return true
    end
    return false
end

function get_normalized_state(env::ExampleEnvironment)
    return [env.state/10.0]
end

