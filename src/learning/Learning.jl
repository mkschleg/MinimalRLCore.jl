

module Learning

abstract type AbstractValueFunction end

mutable struct ValueFunction{T<:AbstractArray, F<:Function, F′<:Function}
    w::T
    σ::F
    σ′::F′
end

(vf::VF)(x) where {VF<:ValueFunction} = vf.σ(vf.w'x)
deriv(vf::VF)(x) where {VF<:ValueFunction} = vf.σ′(vf.w'x).*x

# mutable struct DeepValueFunction{M}
#     model::M
# end

abstract type AbstractLearningUpdate end

struct TD{T<:Number} <: AbstractLearningUpdate
    γ::T
end

function update!(model,
                 lu::TDLearning,
                 opt,
                 s_t::Array{<:AbstractFloat, 1},
                 a_t::I,
                 s_tp1::Array{<:AbstractFloat, 1},
                 r::AF,
                 terminal::Bool) where {AF<:AbstractFloat, I<:Integer}

    ps = params(model)
    γ = lu.γ.*(1 .- terminal)

    v_tp1 = model(s_tp1)
    
    target = (r .+ γ.*v_tp1)
    gs = Flux.gradient(ps) do
        v_t = model(s_t)
        return Flux.mse(target, v_t)
    end
    Flux.Optimise.update!(opt, ps, gs)
end



end

