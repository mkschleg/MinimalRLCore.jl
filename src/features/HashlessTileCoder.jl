

abstract type AbstractHashlessTileCoder <: AbstractFeatureConstructor
end

"""
    HashlessTileCoder(tiles_per_dim, bounds_per_dim, num_tilings, wrap, offset)

This is a struct for tile-coding raw states to a list of active indices. The constructor described here uses the helper function `hashlesstilecoder_args` to convert human-readable variables into the quantities used to compute the tiled indices.

# Arguments
- `tiles_per_dim::Vector{<:Integer}`: A vector describing the number of tiles per dimension of the state input. To tile two dimensions of state together with 2 tiles on the first dimension and 3 tiles on the second, set `tiles_per_dim = [2, 3]`.
- `bounds_per_dim::Matrix{<:Real}`: A matrix of real values denoting the upper and lower bounds of each state dimension. If both state dimensions are bounded in `[0, 1]`, pass `bounds_per_dim = [0 0 ; 1 1]`.
- `num_tilings::Integer`: The tilecoder will map state to `num_tilings` indices.
- `wrap::Union{AbstractVector{Bool}, Nothing}`: Nothing, indicating no wrapping, or a vector of booleans indicating whether that index of the state input should be wrapped. Setting `wrap[k] = true` means that the two ends of `bounds_per_dim[:,k]` are equivalent. For example, if `state[k]` describes an angle between 0 and 2π, `wrap[k]` should be set to `true`.
- `offset`: A function that describes how each tiling should be offset from the others. The recommended setting is to use the default odd-numbered offsets, called asymmetric displacement by [Parks and Militzer](https://doi.org/10.1016/S1474-6670(17)54222-6).

"""
mutable struct HashlessTileCoder <: AbstractHashlessTileCoder
    limits::Matrix{Float64}
    norm_dims::Vector{Float64}
    tiling_dims::Vector{Int32}
    wrap_any_dims::Bool
    offsets::Matrix{Float64}
    tiling_loc::Vector{Int}
    tile_loc::Vector{Int}
    num_features::Int
    num_active_features::Int

    function HashlessTileCoder(a...;k...)
        l, n, t, w, o, tgl, til, nf, naf = hashlesstilecoder_args(a...;k...)
        new(l, n, t, w, o, tgl, til, nf, naf)
    end
end


feature_size(fc::T) where {T<:HashlessTileCoder} = fc.num_features
Base.size(fc::HashlessTileCoder) = fc.num_features

"""
Helper function to make constructing each feature vector later easier
"""
function hashlesstilecoder_args(tiles_per_dim::Vector{<:Integer},
                                bounds_per_dim::Matrix{<:Real},
                                num_tilings::Integer;
                                wrap::Union{AbstractVector{Bool}, Nothing}=nothing,
                                offset= n -> collect(1:2:2*n-1) # 1st n odd nums
                                ) 
    n = length(tiles_per_dim)

    # these normalize the ith input float to be between 0 and dim[i] + 1
    limits = bounds_per_dim
    norm_dims = tiles_per_dim ./ (limits[2, :] .- limits[1, :])

    # wrapping means not adding 1 to the ith dim
    if wrap == nothing
        bonus = ones(Bool, n)
    else
        bonus = .!wrap
    end
    wrap_any_dims = any(.!bonus)
    tiling_dims = tiles_per_dim .+ bonus

    # displacement matrix; default is assymetric displacement a la Parks
    # and Militzer https://doi.org/10.1016/S1474-6670(17)54222-6
    offset_vec = offset(n)
    offsets = (offset_vec
               .* hcat([collect(0:num_tilings-1) for _ in 1:n]...)'
               ./ num_tilings
               .% 1)

    # these send each displaced float to the proper index
    tiling_loc = collect(0:num_tilings-1) .* prod(tiling_dims)
    tile_loc = [prod(tiling_dims[1:i-1]) for i in 1:n]

    # the total number of indices needed
    num_features = num_tilings * prod(tiling_dims)
    num_active_features = num_tilings

    return (limits, norm_dims, tiling_dims, wrap_any_dims, offsets, tiling_loc,
            tile_loc, num_features, num_active_features)
end

"""
Constructs a list of active indices from the quantities defined by
`hashlesstilecoder_args`.
"""
function _create_features(fc::T, s) where {T<:AbstractHashlessTileCoder}
    if fc.wrap_any_dims
        # wrapping means modding by dim[i] instead of dim[i] + 1
        off_coords = map(x -> floor(Int, x),
                         ((s .- fc.limits[1, :])
                          .* fc.norm_dims
                          .+ fc.offsets)
                         .% fc.tiling_dims)
    else
        # don't need to mod here, because dim[i] + 1 is bigger than the
        # displaced floats
        off_coords = Int.(floor.((s .- fc.limits[1, :]) .* fc.norm_dims .+ fc.offsets))
    end

    return fc.tiling_loc .+ off_coords' * fc.tile_loc .+ 1
end
create_features(fc::HashlessTileCoder, s) = _create_features(fc, s)



"""
    HashlessTileCoderFull(tiles_per_dim, bounds_per_dim, num_tilings, wrap, offset)

This is a struct for tile-coding raw states. Unlike HashlessTileCoder, HashlessTileCoderFull returns a full binary array where ones denote that an index is active. The constructor described here uses the helper function `hashlesstilecoder_args` to convert human-readable variables into the quantities used to compute the tiled indices.

# Arguments
- `tiles_per_dim::Vector{<:Integer}`: A vector describing the number of tiles per dimension of the state input. To tile two dimensions of state together with 2 tiles on the first dimension and 3 tiles on the second, set `tiles_per_dim = [2, 3]`.
- `bounds_per_dim::Matrix{<:Real}`: A matrix of real values denoting the upper and lower bounds of each state dimension. If both state dimensions are bounded in `[0, 1]`, pass `bounds_per_dim = [0 0 ; 1 1]`.
- `num_tilings::Integer`: The tilecoder will map state to `num_tilings` indices.
- `wrap::Union{AbstractVector{Bool}, Nothing}`: Nothing, indicating no wrapping, or a vector of booleans indicating whether that index of the state input should be wrapped. Setting `wrap[k] = true` means that the two ends of `bounds_per_dim[:,k]` are equivalent. For example, if `state[k]` describes an angle between 0 and 2π, `wrap[k]` should be set to `true`.
- `offset`: A function that describes how each tiling should be offset from the others. The recommended setting is to use the default odd-numbered offsets, called asymmetric displacement by [Parks and Militzer](https://doi.org/10.1016/S1474-6670(17)54222-6).

"""
mutable struct HashlessTileCoderFull <: AbstractHashlessTileCoder
    limits::Matrix{Float64}
    norm_dims::Vector{Float64}
    tiling_dims::Vector{Int32}
    wrap_any_dims::Bool
    offsets::Matrix{Float64}
    tiling_loc::Vector{Int}
    tile_loc::Vector{Int}
    num_features::Int

    function HashlessTileCoderFull(a...;k...)
        l, n, t, w, o, tgl, til, nf = hashlesstilecoder_args(a...;k...)
        new(l, n, t, w, o, tgl, til, nf)
    end
end
feature_size(fc::T) where {T<:HashlessTileCoderFull} = fc.num_features


function create_features(fc::HashlessTileCoderFull, s)
    x = zeros(Int, fc.num_features)

    x[_create_features(fc, s)] .= 1

    return x
end
