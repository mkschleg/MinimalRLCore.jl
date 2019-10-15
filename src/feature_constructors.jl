export AbstractFeatureConstructor, create_features, feature_size

"""
    AbstractFeatureCreator
An abstract feature creator, for feature transformation from the states.
"""
abstract type AbstractFeatureConstructor end

"""
    create_features(fc::AbstractFeatureCreator, s)
Actually create the features using only states
"""
function create_features(fc::AbstractFeatureConstructor, s)
    throw("Implement create features for $(typof(fc))")
end

"""
    create_features(fc::AbstractFeatureCreator, s, a)
Actually create the features using actions and states
"""
function create_features(fc::AbstractFeatureConstructor, s, a)
    throw("Implement create features for $(typof(fc))")
end

"""
    feature_size(fc::AbstractFeatureCreator)
Get size of feature vector the features assume exists.
"""
function feature_size(fc::AbstractFeatureConstructor)
    throw("Implement feature size for $(typof(fc))")
end
