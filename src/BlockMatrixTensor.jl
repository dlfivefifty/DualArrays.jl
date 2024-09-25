#4-Tensor with data stored in a BlockArray
#Enables sparsity for DualMatrix jacobian.
using BlockArrays, SparseArrays

struct BlockMatrixTensor{T} <: AbstractArray{T, 4}
    data::AbstractBlockMatrix{T}
end

function BlockMatrixTensor(x::AbstractArray{T, 4}) where {T}
    n, m, s, t = size(x)
    data = BlockArray(zeros(T, n * s, m * t), fill(s, n), fill(t, m))
    for i = 1:n, j = 1:m
        data[Block(i,j)] = x[i,j,:,:]
    end
    BlockMatrixTensor(data)
end

size(x::BlockMatrixTensor) = (blocksize(x.data)..., size(x.data[Block(1), Block(1)])...)

getindex(x::BlockMatrixTensor, a::Int, b::Int, c, d) = sparse_getindex(x.data[Block(a), Block(b)], c, d)
getindex(x::BlockMatrixTensor, a::UnitRange, b::UnitRange, ::Colon, ::Colon) = BlockMatrixTensor(x.data[Block.(a),Block.(b)])

function show(io::IO,m::MIME"text/plain", x::BlockMatrixTensor)
    print("BlockMatrixTensor containing: \n")
    show(io,m,x.data)
end
show(io::IO,x::BlockMatrixTensor) = show(io, x.data)

#Blockwise broadcast
function broadcasted(f::Function, x::BlockMatrixTensor, y::AbstractMatrix)
    z = undef
    if size(y, 1) == 1
        z = repeat(y, blocksize(x.data, 1))
    elseif size(y, 2) == 1
        z = repeat(y, 1, blocksize(x.data, 2))
    else
        z = copy(y)
    end
    n, m = size(z)
    blocked = mortar([fill(z[i,j], size(x.data[Block(1,1)])) for i = 1:n, j = 1:m])
    BlockMatrixTensor(f.(blocked, x.data))
end

function broadcasted(f::Function, x::AbstractMatrix, y::BlockMatrixTensor)
    z = undef
    if size(x, 1) == 1
        z = repeat(x, blocksize(y.data, 1))
    elseif size(x, 2) == 1
        z = repeat(y, 1, blocksize(y.data, 2))
    else
        z = copy(x)
    end
    n, m = size(z)
    blocked = mortar([fill(z[i,j], size(y.data[Block(1,1)])) for i = 1:n, j = 1:m])
    BlockMatrixTensor(f.(blocked, y.data))
end

function sum(x::BlockMatrixTensor{T}; dims = Colon()) where {T}
    if dims == 1:2
        n, m = size(x, 3), size(x, 4)
        ret = zeros(T, n, m)
        for i = 1:size(x, 1), j = 1:size(x, 2)
            ret += x.data[Block(i, j)]
        end
        sparse(ret)
    elseif dims == 2
        n, m, s, t = size(x)
        ret = BlockArray(zeros(T, n*s, t), fill(s, n), fill(t, 1))
        for i = 1:n, j = 1:m
            ret[Block(i,1)] += x.data[Block(i,j)]
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
