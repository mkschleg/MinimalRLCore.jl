
using Test
using Random

using MinimalRLCore

# function test_hashlesstilecoder()

#     @testset "Test HashlessTileCoder" begin
#         tc = MinimalRLCore.HashlessTileCoder([2,2], [0 0 ; 1 1], 4)
#         Random.seed!(1)
#         s = rand(2)
#         @test MinimalRLCore.feature_size(tc) == 36
#         @test all(MinimalRLCore.create_features(tc, s) .== [1, 13, 22, 29])
#     end

#     @testset "Test HashlessTileCoderFull" begin
#         tc = MinimalRLCore.HashlessTileCoderFull([2,2], [0 0 ; 1 1], 4)
#         Random.seed!(1)
#         s = rand(2)
#         @test MinimalRLCore.feature_size(tc) == 36
#         @test all(MinimalRLCore.create_features(tc, s) .== [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
#     end

# end
