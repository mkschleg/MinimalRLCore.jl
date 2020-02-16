using Test
using RLCore
using Random

include("test_env.jl")
include("test_agent.jl")
include("test_episode.jl")

function runtests()
    test_env()
    test_agent()
    test_episode()
end

runtests()


