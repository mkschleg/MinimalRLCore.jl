
abstract type AbstractFeatureNormalize <: AbstractFeatureConstructor end

"""
    MinMaxNormalize{T} <: AbstractFeatureNormalize

Preprocessor which normalizes from a known set of min and max.
"""
struct MinMaxNormalize{T} <: AbstractFeatureNormalize
    # Main Arguments
    min::Array{T, 1}
    max::Array{T, 1}
    min_p_max::Array{T, 1}
    MinMaxNormalize(min::Array{T, 1}, max::Array{T, 1}) where {T} =
        new{T}(min, max, max .- min)
end

create_features(fc::MinMaxNormalize, s) = (s .- fc.min)./fc.min_p_max
feature_size(fc::MinMaxNormalize) = length(fc.min)

"""
    MeanStdNormalize{T} <: AbstractFeatureNormalize

Normalize the vector to have 0 mean and 1 standard deviation
"""
struct MeanStdNormalize{T} <: AbstractFeatureNormalize
    mean::Array{T, 1}
    std::Array{T, 1}
end

create_features(fc::MeanStdNormalize, s) = (s .- fc.mean)./fc.std
feature_size(fc::MeanStdNormalize) = length(fc.mean)
