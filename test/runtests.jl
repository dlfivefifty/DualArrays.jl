using DualArrays, Test, LinearAlgebra, BandedMatrices, ForwardDiff, ComponentArrays
using DualArrays: Dual
using Lux: relu

@testset "DualArrays" begin
    
    v = DualArray([1.,2, 3], [1 2 3; 4 5 6;7 8 9])
    @testset "constructors" begin
        @test v isa DualVector
        @test eltype(v) == DualArrays.Dual{Float64}
        w = DualArray(ones(2,2),ones(2,2,2,2))
        @test w isa DualMatrix
        @test_throws TypeError DualArray([1,2], [1,2])
        @test_throws ArgumentError DualVector([1,2],I(3))
    end

    @testset "indexing" begin
        @test v[1] isa Dual
        @test v[1] == Dual(1,[1,2,3])
        @test v[2] == Dual(2,[4,5,6])
        @test v[3] == Dual(3,[7,8,9])
        @test_throws BoundsError v[4]
        @test v == DualVector([1,2, 3], [1 2 3; 4 5 6;7 8 9])

        x,y = v[1:2],v[2:3]
        @test x == DualVector([1,2],[1 2 3;4 5 6])
        @test y == DualVector([2,3],[4 5 6;7 8 9])

        n = 10
        w = DualVector(1:n, I(n))
        @test w[2:end].jacobian isa BandedMatrix
    end

    @testset "basic_operations" begin
        w = v + v
        @test w == DualVector([2.,4.,6.],[2. 4 6;8 10 12;14 16 18])
        @test w.jacobian == 2v.jacobian

        u = v - w
        @test u == -v

        x = v.value + v - v.value
        @test x == v

        @test v .^ 2 == DualVector([1, 4, 9], [2 4 6;16 20 24; 42 48 54])

        x,y = v[1:2],v[2:3]
        @test x .* y == DualVector([2,6],[6 9 12;26 31 36])
        @test sum(x .* y) isa Dual
        @test sum(x .* y) == Dual(8,[32,40,48])
    end

    @testset "broadcasting" begin
        @test sin.(v) isa DualVector
        @test sin.(v).value == [sin(1), sin(2), sin(3)]
        @test sin.(v).jacobian == Diagonal(cos.(v.value)) * v.jacobian
    
        @test exp.(v) isa DualVector
        @test exp.(v).value == [exp(1), exp(2), exp(3)]
        @test exp.(v).jacobian == Diagonal(exp.(v.value)) * v.jacobian
    
        @test relu.(v) isa DualVector
        @test relu.(v).value == [relu(1), relu(2), relu(3)]
        @test relu.(v).jacobian == Diagonal( 0.5 * (sign.(v.value) .+ 1)) * v.jacobian
    end

    @testset "misc" begin
        n = 10
        v = DualVector(1:n, I(n))
        @test sum(v[1:end-1] .* v[2:end]).partials == ForwardDiff.gradient(v -> sum(v[1:end-1] .* v[2:end]), 1:n)
    end

    @testset "vcat" begin
        x = Dual(1, [1, 2, 3])
        y = DualVector([2, 3], [4 5 6;7 8 9])
        @test vcat(x) == x
        @test vcat(x, x) == DualVector([1, 1], [1 2 3;1 2 3])
        @test vcat(x, y) == DualVector([1, 2, 3], [1 2 3;4 5 6;7 8 9])
        @test vcat(1, y, 2) == DualVector([1,2,3,2], [0 0 0; 4 5 6; 7 8 9; 0 0 0])
    end

    @testset "reshape" begin
        x = DualVector([1,2,3,4], I(4))
        @test vec(reshape(x, 2, 2)) == x
    end

    @testset "DualMatrix" begin
        x = DualVector(ones(6), I(6))
        y = reshape(x[1:4], 2, 2)
        z = x[5:6]
        @test (y*z).value == [2,2]
        @test (y*z).jacobian == [1 0 1 0 1 1; 0 1 0 1 1 1]
        for t in (([1, 0, 0, 1, 0, 0], [1 0 -1 0 1 0;0 0 0 0 0 0]), ([1, 1, 1, 1, 1, 1], [1 0 -1 0 1 0;0 1 0 -1 0 1]), ([0, 1, 1, 0, -1, -1], zeros(2,6)))
            x = DualVector(t[1], I(6))
            y = reshape(x[1:4], 2, 2)
            z = x[5:6]
            @test relu.(y*[1,-1] + z).jacobian == t[2]
        end
    end

    @testset "BlockMatrixTensor" begin
        
    end
end