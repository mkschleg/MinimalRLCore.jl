using MinimalRLCore
using Random
using Test

mutable struct TestEnvEpis <: MinimalRLCore.AbstractEnvironment
    size::Int
    state::Int
end

MinimalRLCore.get_state(env::TestEnvEpis) = env.state
MinimalRLCore.get_reward(env::TestEnvEpis) = env.state == env.size ? 1.0 : 0.0
MinimalRLCore.get_actions(env::TestEnvEpis) = (1, 2)
MinimalRLCore.is_terminal(env::TestEnvEpis) = env.state == env.size

MinimalRLCore.reset!(env::TestEnvEpis, rng::Random.AbstractRNG=Random.GLOBAL_RNG) =
    env.state = rand(rng, 1:env.size)

MinimalRLCore.environment_step!(env::TestEnvEpis, action, rng::AbstractRNG=Random.GLOBAL_RNG) = 
    env.state += action == 1 ? -1 : 1

mutable struct TestAgentEps <: AbstractAgent
    action::Int
end
# TestAgentEps(action) = TestAgentEps(action)

function MinimalRLCore.start!(agent::TestAgentEps, s)
    agent.action
end

function MinimalRLCore.step!(agent::TestAgentEps, s, r, t)
    agent.action
end


function test_episode()

    res = [(3, 2, 4, 0.0, false),
           (4, 2, 5, 0.0, false),
           (5, 2, 6, 0.0, false),
           (6, 2, 7, 0.0, false),
           (7, 2, 8, 0.0, false),
           (8, 2, 9, 0.0, false),
           (9, 2, 10, 1.0, true)]

    env = TestEnvEpis(10, 1)
    agent = TestAgentEps(2)
    @testset "Test Episode" begin
        ret = []
        Random.seed!(1)
        total_rew, steps = MinimalRLCore.run_episode!(env, agent) do (sars)
           push!(ret, sars)
        end
        @test all(ret .== res)
        @test total_rew == 1.0
        @test steps == 7
    end


    @testset "Test Episode Max Steps" begin
        ret = []
        Random.seed!(1)
        total_rew, steps = MinimalRLCore.run_episode!(env, agent, 2) do (sars)
           push!(ret, sars)
        end
        @test all(ret .== res[1:2])
        @test total_rew == 0.0
        @test steps == 2
    end

end

