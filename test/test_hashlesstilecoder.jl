
using MinimalRLCore
using Test
using Random

function test_agent()

    @testset "Test HashlessTileCoder" begin
        tc = HashlessTileCoder([2,2], [0 0 ; 1 1], 4)
        Random.seed!(1)
        s = rand(2)
        @test feature_size(tc) == 144
        @test all(create_features(tc, s) .== [1, 13, 22, 29])
    end

    @testset "Test HashlessTileCoder" begin
        tc = HashlessTileCoderFull([2,2], [0 0 ; 1 1], 4)
        Random.seed!(1)
        s = rand(2)
        @test feature_size(tc) == 144
        @test all(create_features(tc, s) .== [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
    end


end
