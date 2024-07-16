using DualArrays, Test, LinearAlgebra, ForwardDiff, BandedMatrices
using DualArrays: Dual

@testset "DualArrays" begin
    @test_throws ArgumentError DualVector([1,2],I(3))
    
    v = DualVector([1,2, 3], [1 2 3; 4 5 6;7 8 9])
    @test v[1] isa Dual
    @test v[1] == Dual(1,[1,2,3])
    @test v[2] == Dual(2,[4,5,6])
    @test v[3] == Dual(3,[7,8,9])
    @test_throws BoundsError v[4]
    @test v == DualVector([1,2, 3], [1 2 3; 4 5 6;7 8 9])

    w = v + v
    @test w == DualVector([2,4,6],[2 4 6;8 10 12;14 16 18])
    @test w.jacobian == 2v.jacobian

    @test sin.(v) isa DualVector
    @test sin.(v).value == [sin(1), sin(2), sin(3)]
    @test sin.(v).jacobian == Diagonal(cos.(v.value)) * v.jacobian

    x,y = v[1:2],v[2:3]
    @test x == DualVector([1,2],[1 2 3;4 5 6])
    @test y == DualVector([2,3],[4 5 6;7 8 9])
    @test x .* y == DualVector([2,6],[6 9 12;26 31 36])
    
    @test sum(x .* y) isa Dual
    @test sum(x .* y) == Dual(8,[32,40,48])

    n = 10
    v = DualVector(1:n, I(n))
    @test v[2:end].jacobian isa BandedMatrix

    @test sum(v[1:end-1] .* v[2:end]).partials == ForwardDiff.gradient(v -> sum(v[1:end-1] .* v[2:end]), 1:n)

    x = Dual(1, [1, 2, 3])
    y = DualVector([2, 3], [4 5 6;7 8 9])
    @test vcat(x) == x
    @test vcat(x, x) == DualVector([1, 1], [1 2 3;1 2 3])
    @test vcat(x, y) == DualVector([1, 2, 3], [1 2 3;4 5 6;7 8 9])
end