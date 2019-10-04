

using Random

struct RandomAgent{T}
    actions::T
end
start!(agent::RandomAgent, env_s_tp1, rng::AbstractRNG; kwargs...) = rand(rng, agent.actions)
step!(agent::RandomAgent, env_s_tp1, r, terminal, rng::AbstractRNG; kwargs...) = rand(rng, agent.actions)

