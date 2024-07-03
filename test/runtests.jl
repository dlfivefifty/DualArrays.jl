using DualArrays, Test, LinearAlgebra

@testset "DualArrays" begin
    v = DualVector([1,2], [1 2; 3 4])
    @test v[1] == 1
    @test v[2] == 2
    @test_throws BoundsError v[3]
    @test v == [1, 2]

    w = v + v
    @test w == [2,4]
    @test w.jacobian == 2v.jacobian

    @test sin.(v) isa DualVector
    @test sin.(v) == [sin(1), sin(2)]
    @test sin.(v).jacobian == Diagonal(cos.(v.value)) * v.jacobian
end