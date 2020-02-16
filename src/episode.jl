"""
    Episode iterators using https://github.com/JuliaML/Reinforce.jl/blob/master/src/episodes/iterators.jl as inspiration for api

"""

using Random


"""
    Episode(env, agent, maxn, rng)

This is a struct for managing the components of an episode iterator. You should only pass a reference to env and agent, while managing the reference seperately.

# Arguments
- `env::AbstractEnvironment`: The Environment (following RLCore interface)
- `agent::AbstractAgent`: The Agent (following RLCore interface)
- `maxn=0`: Max number of steps for the episode.
- `rng=nothing`: The Random Number Generator for the episode (can either be nothing or an AbstractRNG)

"""
mutable struct Episode{E<:AbstractEnvironment, A<:AbstractAgent, F<:AbstractFloat, AR}
    env::E
    agent::A
    total_reward::F # total reward of the episode

    niter::Int      # current step in this episode
    maxn::Int       # max steps in an episode - should be constant during an episode

    rng::AR
end

Episode(env, agent, maxn=0, rng=nothing) =
    Episode(env, agent, get_reward(env), 0, maxn, rng)

function _start!(ep::Episode)
    if ep.rng isa Nothing
        s = start!(ep.env)
        agent_ret = start!(ep.agent, s)
        
        ep.total_reward = zero(ep.total_reward)
        ep.niter = 1
        return (s, agent_ret), ep.niter
    else
        s = start!(ep.env, ep.rng)
        agent_ret = start!(ep.agent, s, ep.rng)
        
        ep.total_reward = zero(ep.total_reward)
        ep.niter = 1
        return (s, agent_ret), ep.niter
    end
end

_done(ep::Episode, state) =
  (ep.maxn != 0 && ep.niter >= ep.maxn) || is_terminal(ep.env)

function Base.iterate(ep::Episode, state = _start!(ep))
    _done(ep, state) && return nothing

    env = ep.env
    agent = ep.agent

    s = state[1][1]
    agent_ret = state[1][2]

    action = if agent_ret isa NamedTuple
        agent_ret.action
    else
        agent_ret
    end

    s′, r, t = if ep.rng isa Nothing
        step!(env, action)
    else
        step!(env, action, ep.rng)
    end

    agent_ret = if ep.rng isa Nothing
        step!(agent, s′, r, t)
    else
        step!(agent, s′, r, t, rng)
    end

    ep.total_reward += r
    ep.niter = state[2]

    (s, agent_ret, s′, r, t), ((s′, agent_ret), ep.niter + 1)
end

function run_episode!(f::Base.Callable, env::AbstractEnvironment, agent::AbstractAgent, args...)
  ep = Episode(env, agent, args...)
  for sars in ep
    f(sars)
  end
  ep.total_reward, ep.niter
end


# function episode!(env, agent, max_eps_steps, total_steps, max_steps; callback=nothing, rng::Random.AbstractRNG=nothing)
#     terminal = false

#     # Start Environment
#     s_t = if rng isa Nothing
#         start!(env)
#     else
#         start!(env, rng)
#     end

#     # Start Agent
#     agent_ret = if rng isa Nothing
#         start!(agent, s_t)
#     else
#         start!(agent, s_t, rng)
#     end

#     total_rew = 0.0
#     steps = 1
#     if !(callback isa Nothing)
#         callback(env, agent, (state=s_t,), agent_ret, steps)
#     end

#     if (total_steps+steps >= max_steps) # 5 Minutes of Gameplay = 18k steps.
#         terminal = true
#     end

#     while !terminal
#         # Figure out action!
#         action = if agent_ret isa NamedTuple
#             agent_ret.action
#         else
#             agent_ret
#         end
        
#         # Step Environment
#         s_tp1, rew, terminal = if rng isa Nothing
#             step!(env, action)
#         else
#             step!(env, action, rng)
#         end

#         # Step Agent
#         agent_ret = if rng isa Nothing
#             step!(agent, s_tp1, rew, terminal)
#         else
#             step!(agent, s_tp1, rew, terminal, rng)
#         end

#         total_rew += rew
#         steps += 1
        
#         if !(callback isa Nothing)
#             callback(env, agent, (state=s_tp1, rew=rew, terminal=terminal), agent_ret, steps)
#         end

#         if (total_steps+steps >= max_steps) || (steps >= max_eps_steps) # 5 Minutes of Gameplay = 18k steps.
#             break
#         end
#     end
#     return total_rew, steps
# end

