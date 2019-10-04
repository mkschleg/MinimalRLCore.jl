module TabularRL

using Lazy
import Base.getindex, Base.setindex!

# export ValueFunction, Optimizer, update!, BatchTD, WISBatchTD, VTrace


# abstract type ValueFunction end


"""
StateValueFunction(size)

# Arguments:
`size`: Integer, many integers, or tuple of integers determining the size of the Value table.
"""
mutable struct VFunction <: ValueFunction
    values::Array{Float64}
    VFunction(size::Integer...) = new(zeros(size...))
end

weights(v::VFunction) = v.values
update!(value::VFunction, s, δ) = value[s] .+= δ

@forward ValueFunction.values Base.getindex
@forward ValueFunction.values Base.setindex!


"""
BatchTD(α, batch_size)

# Arguments:
`α::Float64`: The step size used for learning
`batch_size::Integer`: The size of the batch for learning
`α_mod::Float64`: The modifier to the step size (for WIS!)

Parameter struct for BatchTD
"""
mutable struct BatchTD <: LearningUpdate
    α::Float64
    batch_size::Integer
    α_mod::Float64
    avg::Bool
    BatchTD(α::Float64, batch_size::Integer, α_mod=1.0, avg=false) = new(α, batch_size, α_mod, avg)
end

function update!(value::StateValueFunction, opt::BatchTD, ρ, s_t, s_tp1, r, γ, terminal)
    batch_size = opt.batch_size
    α = opt.α*opt.α_mod
    if opt.avg
        α = α/length(ρ)
    end
    δ = zeros(batch_size)
    γ_eff = γ .* (terminal.==false)
    δ = ρ.*(r + γ_eff.*value[s_tp1] - value[s_t])
    for i = 1:batch_size
        value[s_t[i]] += (α)*δ[i]
    end
end

############
#
# WISTD - Batch
#
############

"""
WISBatchTD(α, batch_size)

# Arguments:
`α::Float64`: The step size used for learning
`batch_size::Integer`: The size of the batch for learning

Parameter struct for WISBatchTD

"""
mutable struct WISBatchTD <: LearningUpdate
    α::Float64
    batch_size::Integer
    α_mod::Float64
    WISBatchTD(α, batch_size) = new(α, batch_size, 1.0)
    WISBatchTD(α, batch_size, α_mod) = new(α, batch_size, α_mod)
end

function update!(value::StateValueFunction, opt::WISBatchTD, ρ, s_t, s_tp1, r, γ, terminal)
    batch_size = opt.batch_size
    α = opt.α*opt.α_mod
    δ = zeros(batch_size)
    γ_eff = γ .* (terminal.==false)
    δ = ρ.*(r + γ_eff.*value[s_tp1] - value[s_t])
    ρ_wis = sum(ρ)
    if ρ_wis != 0
        ρ_wis_inv = 1.0/ρ_wis
        for i = 1:batch_size
            value[s_t[i]] += α*ρ_wis_inv*δ[i]
        end
    end
end




"""
VTrace(α, ρ_bar, batch_size)

# Arguments:
`α::Float64`: The step size used for learning
`ρ_bar::Float64`: The clipping parameter for importance sampling.
`batch_size::Integer`: The size of the batch for learning

Parameter struct for WISBatchTD
"""
mutable struct VTrace <: LearningUpdate
    α::Float64
    ρ_bar::Float64
    batch_size::Integer
    α_mod::Float64
    avg::Bool
    VTrace(α, ρ_bar, batch_size) = new(α, ρ_bar, batch_size, 1.0, false)
    VTrace(α, ρ_bar, batch_size, α_mod) = new(α, ρ_bar, batch_size, α_mod, false)
    VTrace(α, ρ_bar, batch_size, α_mod, avg) = new(α, ρ_bar, batch_size, α_mod, avg)
end

function update!(value::StateValueFunction, opt::VTrace, ρ, s_t, s_tp1, r, γ, terminal)
    # batch_size = length(terminal)
    batch_size = opt.batch_size
    α = opt.α*opt.α_mod
    if opt.avg
        α = α/length(ρ)
    end
    ρ_bar = opt.ρ_bar
    γ_eff = γ .* (terminal.==false)
    δ = zeros(batch_size)
    δ .= clamp.(ρ, 0, ρ_bar).*(r +γ_eff.*value[s_tp1] - value[s_t])
    for i = 1:batch_size
        value[s_t[i]] += (α)*δ[i]
    end
end

export QValueFunction, get_q_values, Sarsa, ExpectedSarsa

mutable struct QValueFunction <: ValueFunction
    values::Array{Float64}
    QValueFunction(num_actions::Integer, size...) = new(zeros(num_actions, size...))
end

@forward QValueFunction.values getindex
@forward QValueFunction.values setindex!



get_q_values(q_value::QValueFunction, a::Integer, s...) = q_value[a, s...]
get_q_values(q_value::QValueFunction, a::Integer, s::Int64...) = q_value[a, s...]
get_q_values(q_value::QValueFunction, a::Integer, s::Array{Int64, 1}) = q_value[a, s...]
get_q_values(q_value::QValueFunction, a_vec::Array{Int64,1}, s_vec::Array{Int64, 1}) = [q_value[a_vec[i], s_vec[i]] for i in 1:length(s_vec)]
get_q_values(q_value::QValueFunction, a_vec::Array{Int64,1}, s_vec::Array{Array{Int64, 1}, 1}) = [q_value[a_vec[i], s_vec[i]...] for i in 1:length(s_vec)]
get_q_values(q_value::QValueFunction, a_vec::Array{Int64,1}, s_vec::Array{CartesianIndex, 1}) = [q_value[a_vec[i], s_vec[i]] for i in 1:length(s_vec)]
get_q_values(q_value::QValueFunction, a_vec::Array{Int64,1}, s_vec::Array{CartesianIndex{2}, 1}) = [q_value[a_vec[i], s_vec[i]] for i in 1:length(s_vec)]
get_q_values(q_value::QValueFunction, a, s) = q_value[a, s]

# abstract type QOptimizer <: Optimizer end

# function update!(q_value::QValueFunction, opt::QOptimizer, ρ, s_t, s_tp1, r, γ, terminal, a_t, a_tp1, policy)
#     throw(ErrorException("Implement update for $(typeof(opt))"))
# end

mutable struct Sarsa <: QOptimizer
    α::Float64
    batch_size::Integer
    α_mod::Float64
    avg::Bool
    Sarsa(α, batch_size) = new(α, batch_size, 1.0, false)
    Sarsa(α, batch_size, α_mod) = new(α, batch_size, α_mod, false)
    Sarsa(α, batch_size, α_mod, avg) = new(α, batch_size, α_mod, avg)
end

function update!(q_value::QValueFunction, opt::Sarsa, ρ, s_t, s_tp1, r, γ, terminal, a_t, a_tp1, policy)
    batch_size = opt.batch_size
    α = opt.α*opt.α_mod
    if opt.avg
        α = α/length(ρ)
    end
    γ_eff = γ .* (terminal.==false)
    δ = zeros(batch_size)
    # println(s_t)
    q_t = get_q_values(q_value, a_t, s_t)
    # println(q_value[a_tp1[1], s_tp1[1]])
    # println(typeof(a_tp1), " ", typeof(s_tp1))
    q_tp1 = get_q_values(q_value, a_tp1, s_tp1)
    # println(r)
    # println(q_tp1)
    # println(size(γ_eff), " ", size(q_tp1), " ", size(r))
    δ .= (r + γ_eff.*q_tp1 - q_t)
    # print(δ)
    for i = 1:batch_size
        q_value[a_t[i], s_t[i]] += α*δ[i]
    end
end

"""
Various functions for getting the expected value of a QFunction. (expected over actions).
"""

get_expected_q_value(q_value::QValueFunction, policy::Array{Float64, 1}, s) =
    sum(policy[a]*get_q_values(q_value, a, s) for a in 1:length(policy))
get_expected_q_value(q_value::QValueFunction, policy::Array{Float64, 1}, s::Int64...) =
    sum(policy[a]*get_q_values(q_value, a, s...) for a in 1:length(policy))
get_expected_q_value(q_value::QValueFunction, policy::Array{Float64, 1}, s::Array{Int64, 1}) =
    [sum(policy[a]*get_q_values(q_value, a, s_ex) for a in 1:length(policy)) for s_ex in s]
get_expected_q_value(q_value::QValueFunction, policy::Array{Float64, 1}, s::Array{Array{Int64, 1}, 1}) =
    [get_expectex_q_value(q_value, policy, s_ex...) for s_ex in s]
get_expected_q_value(q_value::QValueFunction, policy::Array{Float64, 1}, s::Array{CartesianIndex, 1}) =
    [get_expected_q_value(q_value, policy, s_ex) for s_ex in s]
get_expected_q_value(q_value::QValueFunction, policy::Array{Array{Float64, 1}, 1}, s::Array{CartesianIndex, 1}) =
    [get_expected_q_value(q_value, policy[s_idx], s_ex) for (s_idx, s_ex) in enumerate(s)]


mutable struct ExpectedSarsa <: QOptimizer
    α::Float64
    batch_size::Integer
    num_actions::Integer
    α_mod::Float64
    avg::Bool
end

function update!(q_value::QValueFunction, opt::ExpectedSarsa,
                 ρ, s_t, s_tp1, r, γ, terminal, a_t, a_tp1,
                 policy::Array{Float64, 1})
    batch_size = opt.batch_size
    α = opt.α*opt.α_mod
    if opt.avg
        α = α/length(ρ)
    end
    num_actions = opt.num_actions
    γ_eff = γ .* (terminal.==false)
    δ = zeros(batch_size)

    q_t = get_q_values(q_value, a_t, s_t)
    q_expected_tp1 = get_expected_q_value(q_value, policy, s_tp1)
    # q_tp1 = get_q_values(q_value, a_tp1, s_tp1)

    δ .= (r + γ_eff.*q_expected_tp1 .- q_t)
    for i = 1:batch_size
        q_value[a_t[i], s_t[i]] += α*δ[i]
    end
end

function update!(q_value::QValueFunction, opt::ExpectedSarsa,
                 ρ, s_t, s_tp1, r, γ, terminal, a_t, a_tp1,
                 policy::Array{Array{Float64, 1}, 1})
    batch_size = opt.batch_size
    α = opt.α*opt.α_mod
    num_actions = opt.num_actions
    γ_eff = γ .* (terminal.==false)
    δ = zeros(batch_size)

    q_t = get_q_values(q_value, a_t, s_t)
    q_expected_tp1 = get_expected_q_value(q_value, policy, s_tp1)
    # q_tp1 = get_q_values(q_value, a_tp1, s_tp1)

    δ .= (r + γ_eff.*q_expected_tp1 .- q_t)
    for i = 1:batch_size
        q_value[a_t[i], s_t[i]] += α*δ[i]
    end
end


end # module TabularRL
