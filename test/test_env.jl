using Test
using RLCore
using Random


mutable struct TestEnv <: RLCore.AbstractEnvironment
    name::String
    size::Number
    state::Number
    rew::Number
    TestEnv(name, size) = new(name, size, 0, 0)
end

name(t::TestEnv) = t.name

RLCore.reset!(env::TestEnv, rng::Random.AbstractRNG) = env.state = rand(rng) * env.size
RLCore.get_state(env::TestEnv) = env.state
RLCore.get_reward(env::TestEnv) = env.rew
RLCore.get_actions(env::TestEnv) = (1, 2)
RLCore.is_terminal(env::TestEnv) = false

function RLCore.environment_step!(env::TestEnv,
                                  action,
                                  rng::AbstractRNG; kwargs...)
    env.rew = action*env.state
    return
end

function test_env()
    rng = MersenneTwister(10)
    expected_state = 0.5629122239323647
    @testset "TestEnv" begin
        env = TestEnv("Matt", 5)
        @test start!(env, rng) ≈ expected_state
        @test all(step!(env, 1) .≈ (expected_state, 1*expected_state, false))
        @test all(step!(env, 2) .≈ (expected_state, 2*expected_state, false))
        @test all(RLCore.get_actions(env) .== (1,2))
    end
end
