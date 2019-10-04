using Test
using RLCore
using Random

# include("utils.jl")

# function test_env()
#     rng = MersenneTwister(0)
#     @testset "MountainCar" begin
#         @test test_construction(MountainCar, 0.1, 0.04)
#         @test test_construction(MountainCar)
#         @test test_construction(MountainCar, rng)

#         mc = MountainCar(0.1, 0.04, true)
#         @test get_state(mc) == [0.7647058823529412, 0.7857142857142857]
#         rng = MersenneTwister(0)
#         mc = MountainCar(rng)
#         @test get_state(mc) == [0.2236475079774124, 0.0]

#         mc = MountainCar(true)
#         action = 1
#         step!(mc, action)
#         @test get_state(mc) == [0.7038235294117646, 0.475]
#         rng = MersenneTwister(0)
#         start!(mc; rng=rng)
#         @test get_state(mc) == [0.8374397105749485, 0.5]
#     end
# end

function runtests()
    # test_env()
end

runtests()


