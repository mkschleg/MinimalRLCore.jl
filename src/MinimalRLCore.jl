module MinimalRLCore

using Random

greet() = println("Hello Reinforcement Learning!")

export
    AbstractEnvironment,
    start!,
    step!,
    get_actions,
    get_reward,
    is_terminal
include("environment.jl")

export
    AbstractAgent,
    start!,
    step!
include("agent.jl")

export
    Episode,
    run_episode!
include("episode.jl")

export
    AbstractFeatureConstructors,
    create_features,
    feature_size
include("feature_constructors.jl")

# include("features/HashlessTileCoder.jl")

end # module
