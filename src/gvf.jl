
using Lazy
using StatsBase
using Random

import Base.get, Base.get!

"""
    GVFParamFuncs

Module containing and the GVF parameter function types. Cleaner to keep these in a seperate namespace where the user can decide to `using` if desired.
"""
module GVFParamFuncs

import Base.get

using ..Random
using ..StatsBase
using ..RLCore

export
    AbstractCumulant,
    FeatureCumulant,
    PredictionCumulant,
    ScaledCumulant,
    FunctionalCumulant
include("gvf/cumulant.jl")

export
    AbstractDiscount,
    ConstantDiscount,
    StateTerminationDiscount
include("gvf/discount.jl")

export
    AbstractPolicy,
    NullPolicy,
    PersistentPolicy,
    RandomPolicy,
    FucntionalPolicy
include("gvf/policy.jl")

end

"""
    AbstractGVF

This is the base type for a General Value Function. See the "Horde: A scalable real-time architecture for learning knowledge from unsupervised sensorimotor interaction" paper for more details.
"""
abstract type AbstractGVF end

"""
    get(gvf::AbstractGVF, state_t, action_t, state_tp1, action_tp1, preds_tp1)

Get the parameters for the cumulant, discount, and probability of taking an action given the parameters.
"""
function Base.get(gvf::AbstractGVF, state_t, action_t, state_tp1, action_tp1, preds_tp1) end

"""
    get(gvf::AbstractGVF, state_t, action_t, state_tp1, preds_tp1)

Convenience function: `get(gvf, state_t, action_t, state_tp1, nothing, preds_tp1)`
"""
Base.get(gvf::AbstractGVF, state_t, action_t, state_tp1, preds_tp1) =
    get(gvf::AbstractGVF, state_t, action_t, state_tp1, nothing, preds_tp1)

"""
    get(gvf::AbstractGVF, state_t, action_t, state_tp1)

Convenience function: `get(gvf, state_t, action_t, state_tp1, nothing, nothing)`
"""
Base.get(gvf::AbstractGVF, state_t, action_t, state_tp1) =
    get(gvf::AbstractGVF, state_t, action_t, state_tp1, nothing, nothing)

function cumulant(gvf::AbstractGVF) end
function discount(gvf::AbstractGVF) end
function policy(gvf::AbstractGVF) end

"""
    GVF{C<:AbstractCumulant, D<:AbstractDiscount, P<:AbstractPolicy} <: AbstractGVF

A realized version of a GVF where the cumulant, discount, and policies can be any structure following the AbstractCumulant, AbstractDiscount, or AbstractPolicy api respectively.
"""
struct GVF{C<:GVFParamFuncs.AbstractCumulant,
           D<:GVFParamFuncs.AbstractDiscount,
           P<:GVFParamFuncs.AbstractPolicy} <: AbstractGVF
    cumulant::C
    discount::D
    policy::P
end

cumulant(gvf::G) where {G <: GVF} = gvf.cumulant
discount(gvf::G) where {G <: GVF} = gvf.discount
policy(gvf::G) where {G <: GVF} = gvf.policy

function Base.get(gvf::GVF, state_t, action_t, state_tp1, action_tp1, preds_tp1)
    c = get(gvf.cumulant, state_tp1, action_tp1, preds_tp1)
    γ = get(gvf.discount, state_t, action_t, state_tp1, action_tp1, preds_tp1)
    π_prob = get(gvf.policy, state_t, action_t)
    return c, γ, π_prob
end


"""
    AbstractHorde

An abstract collection of GVFs.
"""
abstract type AbstractHorde end

"""
    Horde{T<:AbstractGVF} <: AbstractHorde

The simplest implementation of a horde as a collection of AbstractGVFs.
"""
struct Horde{T<:AbstractGVF} <: AbstractHorde
    gvfs::Vector{T}
end

function Base.get(gvfh::Horde, state_t, action_t, state_tp1, action_tp1, preds_tp1)
    C = map(gvf -> get(cumulant(gvf), state_tp1, action_tp1, preds_tp1), gvfh.gvfs)
    Γ = map(gvf -> get(discount(gvf), state_t, action_t, state_tp1, action_tp1, preds_tp1), gvfh.gvfs)
    Π_probs = map(gvf -> get(policy(gvf), state_t, action_t), gvfh.gvfs)
    return C, Γ, Π_probs
end

function Base.get!(C::Array{T, 1}, Γ::Array{F, 1}, Π_probs::Array{H, 1}, gvfh::Horde, state_t, action_t, state_tp1, action_tp1, preds_tp1) where {T, F, H}
    C .= map(gvf -> get(cumulant(gvf), state_tp1, action_tp1, preds_tp1), gvfh.gvfs)
    Γ .= map(gvf -> get(discount(gvf), state_t, action_t, state_tp1, action_tp1, preds_tp1), gvfh.gvfs)
    Π_probs .= map(gvf -> get(policy(gvf), state_t, action_t), gvfh.gvfs)
    return C, Γ, Π_probs
end

Base.get(gvfh::Horde, state_tp1, preds_tp1) =
    get(gvfh::Horde, nothing, nothing, state_tp1, nothing, preds_tp1)

Base.get(gvfh::Horde, state_t, action_t, state_tp1) =
    get(gvfh::Horde, state_t, action_t, state_tp1, nothing, nothing)

Base.get(gvfh::Horde, state_t, action_t, state_tp1, preds_tp1) =
    get(gvfh::Horde, state_t, action_t, state_tp1, nothing, preds_tp1)

@forward Horde.gvfs Base.length

