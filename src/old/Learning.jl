
module Learning

"""
    Optimizer
WIP - Currently LearningUpdate and Optimizer are haphazardly similar....
"""
abstract type Optimizer end
"""
    LearningUpdate
WIP - Currently LearningUpdate and Optimizer are haphazardly similar....
"""
abstract type LearningUpdate end

"""
    AbstractValueFunctionn
Abstract type definition for a value function
"""
abstract type AbstractValueFunction end

weights(q::AbstractValueFunction) = nothing
update!(value::AbstractValueFunction, Δθ) = throw("Implement weight update for $(typeof(value))")
update!(value::AbstractValueFunction, s, δ) = throw("Implement weight update for $(typeof(value))")

feature_type(value::AbstractValueFunction) = Float64

"""
    AbstractVFunction
Abstract type definition for a state value function
"""
abstract type AbstractVFunction <: AbstractValueFunction end


"""
    update!(value::ValueFunction, opt::Optimizer, ρ, s_t, s_tp1, reward, γ, terminal)

# Arguments:
`value::ValueFunction`:
`opt::Optimizer`:
`ρ`: Importance sampling ratios (Array of Floats)
`s_t`: States at time t
`s_tp1`: States at time t + 1
`reward`: cumulant or reward for value function
`γ`: discount factor
`terminal`: Determining termination of the episode (if applicable).
"""
function update!(value::AbstractVFunction, lu::LearningUpdate, ϕ_t, ϕ_tp1, reward, γ, ρ, terminal)
    throw(ErrorException("Implement update for $(typeof(opt)) and $(typeof(value))"))
end


"""
    AbstractQFunction
Abstract type definition for a state-action value function
"""
abstract type AbstractQFunction <: AbstractValueFunction end

update!(value::AbstractQFunction, s, a, δ) = throw("Implement weight update for $(typeof(value))")

"""
    update!(value::ValueFunction, opt::Optimizer, ρ, s_t, s_tp1, reward, γ, terminal)

# Arguments:
`value::ValueFunction`:
`opt::Optimizer`:
`ρ`: Importance sampling ratios (Array of Floats)
`s_t`: States at time t
`s_tp1`: States at time t + 1
`reward`: cumulant or reward for value function
`γ`: discount factor
`terminal`: Determining termination of the episode (if applicable).
`a_t`: Action at time t
`a_tp1`: Action at time t + 1
`target_policy`: Action at time t
"""
function update!(value::AbstractQFunction, lu::LearningUpdate, ϕ_t, ϕ_tp1, reward, γ, ρ, terminal, a_t, a_tp1, target_policy)
    throw(ErrorException("Implement update for $(typeof(opt)) and $(typeof(value))"))
end

# include("learning/Updates.jl")

export LinearRL
include("learning/LinearRL.jl")

export TabularRL
include("learning/TabularRL.jl")



# include("learning/FluxRL.jl")




end






