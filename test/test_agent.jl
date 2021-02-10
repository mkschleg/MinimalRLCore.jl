
using MinimalRLCore
using Test
using Random
using StableRNGs

mutable struct TestAgent <: AbstractAgent
    cur_s::Array{Float64, 1}
end
TestAgent(size) = TestAgent(zeros(size))


function MinimalRLCore.start!(agent::TestAgent, s)
    agent.cur_s .= s
    a = rand(1:4)
end

function MinimalRLCore.start!(agent::TestAgent, s, rng::Random.AbstractRNG)
    agent.cur_s .= s
    a = rand(rng, 1:4)
end

function MinimalRLCore.step!(agent::TestAgent, s, r, t)
    agent.cur_s .= s
    a = rand(1:4)
end

function MinimalRLCore.step!(agent::TestAgent, s, r, t, rng::Random.AbstractRNG)
    agent.cur_s .= s
    a = rand(rng, 1:4)
end

function test_agent()

    # TODO: Think of a better way to test agent.
    
    # @testset "Test Agent GlobalRNG" begin
    #     agent = TestAgent(2)
    #     Random.seed!(1)
    #     s = rand(2)
    #     @test start!(agent, s) == 4
    #     @test all(agent.cur_s .== s)
    #     s = rand(2)
    #     @test step!(agent, s, 0, false) == 3
    #     @test all(agent.cur_s .== s)
    # end

    @testset "Test Agent local RNG" begin
        agent = TestAgent(2)
        rng = StableRNG(1)
        s = rand(rng, 2)
        @test start!(agent, s, rng) == 1
        @test all(agent.cur_s .== s)
        s = rand(rng, 2)
        @test step!(agent, s, 0, false, rng) == 2
        @test all(agent.cur_s .== s)
    end

    
end

