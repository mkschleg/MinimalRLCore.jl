using Test
using MinimalRLCore
using Random

mutable struct TestEnv <: MinimalRLCore.AbstractEnvironment
    name::String
    size::Int
    state::Float64
    rew::Float64
    TestEnv(name, size) = new(name, size, 0.0, 0.0)
end

name(t::TestEnv) = t.name

MinimalRLCore.get_state(env::TestEnv) = env.state
MinimalRLCore.get_reward(env::TestEnv) = env.rew
MinimalRLCore.get_actions(env::TestEnv) = (1, 2)
MinimalRLCore.is_terminal(env::TestEnv) = false


MinimalRLCore.reset!(env::TestEnv, rng::Random.AbstractRNG=Random.GLOBAL_RNG) =
    env.state = rand(rng) * env.size

MinimalRLCore.reset!(env::TestEnv, state::Number) =
    env.state = state

MinimalRLCore.environment_step!(env::TestEnv, action, rng::AbstractRNG=Random.GLOBAL_RNG) = 
    env.rew = action*env.state


function test_env()
    Random.seed!(10)
    expected_state = 0.5629122239323647
    @testset "TestEnv Global RNG" begin
        env = TestEnv("Matt", 5)
        @test start!(env) ≈ expected_state
        @test all(step!(env, 1) .≈ (expected_state, 1*expected_state, false))
        @test all(step!(env, 2) .≈ (expected_state, 2*expected_state, false))
        @test all(MinimalRLCore.get_actions(env) .== (1,2))
    end

    
    rng = MersenneTwister(10)
    @testset "TestEnv Managed RNG" begin
        env = TestEnv("Matt", 5)
        @test start!(env, rng) ≈ expected_state
        @test all(step!(env, 1, rng) .≈ (expected_state, 1*expected_state, false))
        @test all(step!(env, 2, rng) .≈ (expected_state, 2*expected_state, false))
        @test all(MinimalRLCore.get_actions(env) .== (1,2))
    end

    Random.seed!(10)
    expected_state = rand()
    @testset "TestEnv deterministic state restart" begin
        env = TestEnv("Matt", 5)
        @test start!(env, expected_state) ≈ expected_state
        @test all(step!(env, 1, rng) .≈ (expected_state, 1*expected_state, false))
        @test all(step!(env, 2, rng) .≈ (expected_state, 2*expected_state, false))
        @test all(MinimalRLCore.get_actions(env) .== (1,2))
    end
end
