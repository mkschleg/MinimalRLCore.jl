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

# export
#     GVF,
#     Horde,
#     GVFParamFuncs
# include("gvf.jl")

export
    AbstractFeatureConstructors,
    create_features,
    feature_size
include("feature_constructors.jl")


# export
#     TileCoder,
#     MinMaxNormalize,
#     MeanStdNormalize
# include("features/TileCoder.jl")
# include("features/Normalize.jl")

# export
#     RandomAgent
# include("agent/random.jl")

# Not tested yet so not exported!
# include("monte_carlo.jl")


end # module
