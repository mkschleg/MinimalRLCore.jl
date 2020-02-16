using Random
using RLCore

module MountainCarConst
const vel_limit = (-0.07f0, 0.07f0)
const pos_limit = (-1.2f0, 0.5f0)
const pos_initial_range = (-0.6f0, 0.4f0)

const Reverse=1
const Neutral=2
const Accelerate=3
end

"""
    MountainCar
"""
mutable struct MountainCar <: RLCore.AbstractEnvironment
    pos::Float32
    vel::Float32
    function MountainCar(pos=0.0f0, vel=0.0f0)
        mcc = MountainCarConst
        @boundscheck (pos >= mcc.pos_limit[1] && pos <= mcc.pos_limit[2])
        @boundscheck (vel >= mcc.vel_limit[1] && vel <= mcc.vel_limit[2])
        new(pos, vel)
    end
end

RLCore.get_actions(env::MountainCar) = [mcc.Reverse, mcc.Neutral, mcc.Accelerate]
valid_action(env::MountainCar, action) = action in get_action(env)

function RLCore.reset!(env::MountainCar, rng=Random.GlobalRNG)
    env.pos = (rand(rng)*(MountainCarConst.pos_initial_range[2]
                          - MountainCarConst.pos_initial_range[1])
               + MountainCarConst.pos_initial_range[1])
    env.vel = 0.0
end

function RLCore.reset!(env::MountainCar,
                        start_state::T) where {T<:AbstractArray}
        env.pos = start_state[1]
        env.vel = start_state[2]
end

function RLCore.environment_step!(env::MountainCar, action, rng=Random.GlobalRNG)
    
    @boundscheck valid_action(env, action)
    env.vel =
        clamp(env.vel + (action - 2)*0.001 - 0.0025*cos(3*env.pos),
              MountainCarConst.vel_limit...)
    env.pos = clamp(env.pos + env.vel,
                    MountainCarConst.pos_limit...)
end

function RLCore.get_reward(env::MountainCar) # -> determines if the agent_state is terminal
    if env.pos >= MountainCarConst.pos_limit[2]
        return 0
    end
    return -1
end

function RLCore.is_terminal(env::MountainCar) # -> determines if the agent_state is terminal
    return env.pos >= MountainCarConst.pos_limit[2]
end

function RLCore.get_state(env::MountainCar)
    return [env.pos, env.vel]
end
