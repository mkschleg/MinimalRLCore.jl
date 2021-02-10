export AbstractFeatureConstructor, create_features, feature_size

"""
    AbstractFeatureCreator
An abstract feature creator, for feature transformation from the states.
"""
abstract type AbstractFeatureConstructor end

"""
    create_features
Actually create the features
"""
function create_features end

@deprecate feature_size Base.size

"""
    feature_size(fc::AbstractFeatureCreator)

Depricated! Use Base.size instead. Get size of feature vector the features assume exists.
"""
function feature_size(fc::AbstractFeatureConstructor)
    throw("Implement feature size for $(typeof(fc))")
end


