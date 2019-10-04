module JuliaRL

greet() = print("Hello Reinforcement Learning Julia!")

export FeatureCreators
include("FeatureCreators.jl")

export Learning
include("Learning.jl")

export ExperienceReplay, WeightedExperienceReplay, size, getindex, add!
include("Replay.jl")

export
    AbstractEnvironment,
    MountainCar,
    GymEnv,
    start,
    start!,
    step!,
    step,
    get_reward,
    get_state,
    is_terminal,
    get_actions,
    render

include("Environments.jl")

export AbstractState

abstract type AbstractState end

export AbstractPolicy, AbstractQPolicy, EpsilonGreedyQPolicy, get

abstract type AbstractPolicy end

abstract type AbstractQPolicy <: AbstractPolicy end

mutable struct EpsilonGreedyQPolicy <: AbstractQPolicy
    ϵ::Float64
    actions::AbstractArray
end

function get(policy::AbstractQPolicy, values::AbstractArray; rng=Random.GLOBAL_RNG)
    action = findmax(values)[2]
    if rand(rng) < policy.ϵ
        action = rand(rng, policy.actions)
    end
    return action
end

export Agent

include("Agent.jl")

export my_module

module my_module
export func

"""
    func(x)
Returns double the number `x` plus `1`.
"""
func(x) = 2x + 1
end

end # module
