"""
    AbstractCumulant
"""
abstract type AbstractCumulant end

function Base.get(cumulant::AbstractCumulant, state_tp1, action_tp1, preds_tp1)
    throw(DomainError("get(CumulantType, args...) not defined!"))
end


"""
    FeatureCumulant <: AbstractCumulant

Basic Cumulant which takes the value c_tp1 = s_tp1[idx] for 1<=idx<=length(s_tp1)
"""
struct FeatureCumulant <: AbstractCumulant
    idx::Int
end

Base.get(cumulant::FeatureCumulant, state_tp1, action_tp1, preds_tp1) =
    state_tp1[cumulant.idx]

"""
    PredictionCumulant <: AbstractCumulant

Basic cumulant which takes the value c_tp1 = preds_tp1[idx] 
"""
struct PredictionCumulant <: AbstractCumulant
    idx::Int
end

Base.get(cumulant::PredictionCumulant, state_tp1, action_tp1, preds_tp1) =
    preds_tp1[cumulant.idx]

"""
    ScaledCumulant{F<:Number, T<:AbstractCumulant} <: AbstractCumulant

A cumulant which scales another AbstractCumulant
"""
struct ScaledCumulant{F<:Number, T<:AbstractCumulant} <: AbstractCumulant
    scale::F
    cumulant::T
end

Base.get(cumulant::ScaledCumulant, state_tp1, action_tp1, preds_tp1) =
    cumulant.scale*get(cumulant.cumulant, state_tp1, action_tp1, preds_tp1)

"""
    FunctionalCumulant{F<:Function} <: AbstractCumulant

A cumulant that has a user defined function c_tp1 = f(state_tp1, action_tp1, preds_tp1)
"""
struct FunctionalCumulant{F<:Function} <: AbstractCumulant
    f::F
end

Base.get(cumulant::FunctionalCumulant, state_tp1, action_tp1, preds_tp1) =
    cumulant.f(state_tp1, action_tp1, preds_tp1)

