#4-Tensor with data stored in a BlockMatrix
#Enables sparsity for DualMatrix jacobian.
using BlockArrays, SparseArrays

struct BlockMatrixTensor{T} <: AbstractArray{T, 4}
    data::BlockedMatrix{T}
end

#Construct BlockMatrixTensor from 4-tensor. Useful for testing purposes.
function BlockMatrixTensor(x::AbstractArray{T, 4}) where {T}
    n, m, s, t = size(x)
    data = BlockedMatrix(zeros(T, n * s, m * t), fill(s, n), fill(t, m))
    for i = 1:n, j = 1:m
        view(data, Block(i, j)) .= x[i, j, :, :] 
    end
    BlockMatrixTensor(data)
end

#Construct from matrix of blocks (similar to mortar())
function BlockMatrixTensor(x::Matrix{T}) where {T <: AbstractMatrix}
    n, m = size(x)
    s, t = size(x[1, 1])
    mat = hcat((vcat(x[i,:]...) for i = 1:n)...)
    blockmat = BlockedMatrix(mat, fill(s, n), fill(t, m))
    BlockMatrixTensor(blockmat)
end

size(x::BlockMatrixTensor) = (blocksize(x.data)..., blocksizes(x.data)[1,1]...)

function getindex(x::BlockMatrixTensor, a::Int, b::Int, c, d)
    s, t = size(x, 3), size(x, 4)
    mat = parent(x.data)
    sub = mat[((a - 1)* s + 1):(a * s),((b - 1)* t + 1):(b * t)]
    sub[c, d]
end

function show(io::IO,m::MIME"text/plain", x::BlockMatrixTensor)
    print("BlockMatrixTensor containing: \n")
    show(io,m, x.data)
end
show(io::IO, x::BlockMatrixTensor) = show(io, x.data)

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
            view(ret, Block(i, j)) .= f.(y[i, j], x.data[Block(i, j)])
        end
    elseif size(y) == (1, 1)
        f.(y[1, 1], ret)
    elseif size(y) == (1, m)
        for j = 1:m
            view(ret, :, Block(j)) .= f.(y[1, j], x.data[:, Block(j)])
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
            view(ret, Block(i, j)) .= f.(x[i, j], y.data[Block(i, j)])
        end
    elseif size(x) == (1, 1)
        f.(x[1, 1], ret)
    elseif size(x) == (1, m)
        for j = 1:m
            view(ret, :, Block(j)) .= f.(x[1, j], y.data[:, Block(j)])
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

function sum(x::BlockMatrixTensor; dims = Colon())
    if dims == 1:2
        n, m = size(x, 3), size(x, 4)
        ret = similarzeros(x[1,1,:,:], n, m)
        for i = 1:size(x, 1), j = 1:size(x, 2)
            ret += x[i,j,:,:]
        end
        ret
    elseif dims == 2
        n, m, s, t = size(x)
        # Indexing [:, Block(j)] does not preserve density
        ret = BlockedArray(similarzeros(x[1,1,:,:], n * s, t), fill(s, n), [t])
        for i = 1:n, j = 1:m
            view(ret, Block(i, 1)) .+= x.data[Block(i, j)]
        end
        BlockMatrixTensor(ret)
    end
end

#Helper function for initialising a BlockedMatrix to be populated while preserving sparsity.
similarzeros(x::Matrix, dims...) = zeros(eltype(x), dims...)
similarzeros(x::AbstractMatrix, dims...) = spzeros(eltype(x), dims...)

function reshape(x::BlockMatrixTensor, dims::Vararg{Union{Colon, Int}, 4})
    n, m, s, t = size(x)
    #Block-wise reshape
    if dims[3] isa Colon && dims[4] isa Colon
        ret = BlockedMatrix(similarzeros(x[1, 1, :, :], s * dims[1], t * dims[2]), fill(s, dims[1]), fill(t, dims[2]))
        for j = 1:m, i = 1:n
            c = (j - 1) * n + (i - 1)
            a, b = c % dims[1] + 1, div(c, dims[1]) + 1
            ret[Block(a,b)] = x.data[Block(i,j)]
        end
        BlockMatrixTensor(ret)
    end
end

#'Flatten': converts 4-Tensor to matrix. For BlockMatrixTensors calls parent().
# Mimics the reshape achieved by this for general 4-Tensors
flatten(x::BlockMatrixTensor) = parent(x.data)
flatten(x::AbstractArray{T, 4}) where {T} = reshape(x, size(x, 1) * size(x, 3), size(x, 2) * size(x, 4))
