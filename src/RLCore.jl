module RLCore

using Random
using Logging

greet() = println("Hello Reinforcement Learning!")

export
    AbstractEnvironment,
    get_actions,
    get_reward,
    is_terminal,
    AbstractAgent,
    start!,
    step!

include("environment.jl")
include("agent.jl")

export
    GVF,
    Horde,
    GVFParamFuncs
include("gvf.jl")

export
    AbstractFeatureConstructors,
    create_features,
    feature_size
include("feature_constructors.jl")

export
    TileCoder,
    MinMaxNormalize,
    MeanStdNormalize
include("features/TileCoder.jl")
include("features/Normalize.jl")

export
    RandomAgent
include("agent/random.jl")


end # module
