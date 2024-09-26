#4-Tensor with data stored in a BlockArray
#Enables sparsity for DualMatrix jacobian.
using BlockArrays, SparseArrays

struct BlockMatrixTensor{T} <: AbstractArray{T, 4}
    data::BlockMatrix{T}
end

function BlockMatrixTensor(x::AbstractArray{T, 4}) where {T}
    n, m, s, t = size(x)
    data = BlockArray(zeros(T, n * s, m * t), fill(s, n), fill(t, m))
    for i = 1:n, j = 1:m
        view(data, Block(i,j)) .= x[i,j,:,:] 
    end
    BlockMatrixTensor(data)
end

size(x::BlockMatrixTensor) = (blocksize(x.data)..., blocksizes(x.data, 1)...)

getindex(x::BlockMatrixTensor, a::Int, b::Int, c, d) = sparse_getindex(x.data[Block(a, b)], c, d)
getindex(x::BlockMatrixTensor, a::UnitRange, b::UnitRange, ::Colon, ::Colon) = BlockMatrixTensor(x.data[Block.(a),Block.(b)])

function show(io::IO,m::MIME"text/plain", x::BlockMatrixTensor)
    print("BlockMatrixTensor containing: \n")
    show(io,m,x.data)
end
show(io::IO,x::BlockMatrixTensor) = show(io, x.data)

# Blockwise broadcast
for op in (:*, :/)
    @eval $op(x::BlockMatrixTensor, y::Number) = BlockMatrixTensor($op(x.data, y))
    @eval $op(x::Number, y::BlockMatrixTensor) = BlockMatrixTensor($op(x, y.data))
end

function broadcasted(f::Function, x::BlockMatrixTensor{T}, y::AbstractMatrix) where {T}
    ret = copy(x.data)
    n, m = blocksize(x.data)
    if blocksize(x.data) == size(y)
        for i = 1:n, j = 1:m
            view(ret, Block(i,j)) .= f.(y[i,j], x.data[Block(i,j)])
        end
    elseif size(y) == (1, 1)
        f.(y[1,1], ret)
    elseif size(y) == (1, m)
        for j = 1:m
            view(ret, :, Block(j)) .= f.(y[1,j], x.data[:, Block(j)])
        end
    elseif size(y) == (n, 1)
        for i = 1:n
            view(ret, Block(i), :) .= f.(y[i, 1], x.data[Block(i), :])
        end
    else
        a = size(x)
        b = size(y)
        throw(DimensionMismatch("Invalid dimensions for broadcasting. Got $a, $b."))
    end
    BlockMatrixTensor(ret)
end

function broadcasted(f::Function, x::AbstractMatrix, y::BlockMatrixTensor{T}) where {T}
    ret = copy(y.data)
    n, m = blocksize(y.data)
    if blocksize(y.data) == size(x)
        for i = 1:n, j = 1:m
            view(ret, Block(i,j)) .= f.(x[i,j], y.data[Block(i,j)])
        end
    elseif size(x) == (1, 1)
        f.(x[1,1], ret)
    elseif size(x) == (1, m)
        for j = 1:m
            view(ret, :, Block(j)) .= f.(x[1,j], y.data[:, Block(j)])
        end
    elseif size(x) == (n, 1)
        for i = 1:n
            view(ret, Block(i), :) .= f.(x[i, 1], y.data[Block(i), :])
        end
    else
        a = size(y)
        b = size(x)
        throw(DimensionMismatch("Invalid dimensions for broadcasting. Got $a, $b."))
    end
    BlockMatrixTensor(ret)
end

function sum(x::BlockMatrixTensor{T}; dims = Colon()) where {T}
    if dims == 1:2
        n, m = size(x, 3), size(x, 4)
        ret = similar(x.data[Block(1, 1)]) * 0
        for i = 1:size(x, 1), j = 1:size(x, 2)
            ret += x.data[Block(i, j)]
        end
        ret
    elseif dims == 2
        n, m, s, t = size(x)
        ret = x.data[Block.(1:n), Block(1)] * 0
        for i = 1:n, j = 1:m
            view(ret, Block(i,1)) .+= x.data[Block(i,j)]
        end
        BlockMatrixTensor(ret)
    end
end

function reshape(x::BlockMatrixTensor, dims::Vararg{Int64})
    #Blockwise reshape
    if dims[3] == size(x, 3) && dims[4] == size(x, 4)
        mortar(reshape(blocks(x.data), dims[1], dims[2]))
    end
end

vec(x::BlockMatrixTensor) = vcat(blocks(x.data)...)
