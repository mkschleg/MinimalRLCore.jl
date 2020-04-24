using Test

include("test_env.jl")
include("test_agent.jl")
include("test_episode.jl")
include("test_hashlesstilecoder.jl")

function runtests()
    test_env()
    test_agent()
    test_episode()
    test_hashlesstilecoder()
end

runtests()
