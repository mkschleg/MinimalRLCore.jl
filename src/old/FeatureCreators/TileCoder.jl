
# Module for tilecoding functionality
include("utilities/TileCoding.jl")

"""
    TileCoder(num_tilings, num_tiles, num_features, num_ints)
Tile coder for coding all features together.
"""
mutable struct TileCoder <: AbstractFeatureCreator
    # Main Arguments
    num_tilings::Int64
    num_tiles::Int64
    num_features::Int64
    num_ints::Int64

    # Optional Arguments
    wrap::Bool
    wrapwidths::Float64

    iht::TileCoding.IHT
    TileCoder(num_tilings, num_tiles, num_features, num_ints=1; wrap=false, wrapwidths=0.0) =
        new(num_tilings, num_tiles, num_features, num_ints, wrap, 0.0, TileCoding.IHT(num_tilings*(num_tiles+1)^num_features * num_ints))
end

function create_features(fc::TileCoder, s; ints=[], readonly=false)
    if fc.wrap
        return 1 .+ TileCoding.tileswrap!(fc.iht, fc.num_tilings, s.*fc.num_tiles, fc.wrapwidths, ints, readonly)
    else
        return 1 .+ TileCoding.tiles!(fc.iht, fc.num_tilings, s.*fc.num_tiles, ints, readonly)
    end
end

feature_size(fc::TileCoder) = fc.num_tilings

