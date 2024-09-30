#4-Tensor with data stored in a BlockMatrix
#Enables sparsity for DualMatrix jacobian.
using BlockArrays, SparseArrays

struct BlockMatrixTensor{T} <: AbstractArray{T, 4}
    data::BlockMatrix{T}
end

#Construct BlockMatrixTensor from 4-tensor. Useful for testing purposes.
function BlockMatrixTensor(x::AbstractArray{T, 4}) where {T}
    n, m, s, t = size(x)
    data = BlockMatrix(zeros(T, n * s, m * t), fill(s, n), fill(t, m))
    for i = 1:n, j = 1:m
        view(data, Block(i, j)) .= x[i, j, :, :] 
    end
    BlockMatrixTensor(data)
end

# Useful for creating a BlockMatrixTensor from blocks()
BlockMatrixTensor(x::Matrix{T}) where {T <: AbstractMatrix} = BlockMatrixTensor(mortar(x))

size(x::BlockMatrixTensor) = (blocksize(x.data)..., blocksizes(x.data)[1,1]...)

#Indexing entire blocks
getindex(x::BlockMatrixTensor, a::Int, b::Int, ::Colon, ::Colon) = blocks(x.data)[a, b]
getindex(x::BlockMatrixTensor, a::Int, b, ::Colon, ::Colon) = BlockMatrixTensor(reshape(blocks(x.data)[a, b], 1, :))
getindex(x::BlockMatrixTensor, a, b::Int, ::Colon, ::Colon) = BlockMatrixTensor(reshape(blocks(x.data)[a, b], :, 1))
getindex(x::BlockMatrixTensor, a, b, ::Colon, ::Colon) = BlockMatrixTensor(blocks(x.data)[a,b])


# For populating a BlockMatrixTensor
function setindex!(A::BlockMatrixTensor, value, a::Int, b::Int, ::Colon, ::Colon)
    blocks(A.data)[a, b] = value
end

function show(io::IO,m::MIME"text/plain", x::BlockMatrixTensor)
    print("BlockMatrixTensor containing: \n")
    show(io,m, x.data)
end
show(io::IO, x::BlockMatrixTensor) = show(io, x.data)

for op in (:*, :/)
    @eval $op(x::BlockMatrixTensor, y::Number) = BlockMatrixTensor($op(x.data, y))
    @eval $op(x::Number, y::BlockMatrixTensor) = BlockMatrixTensor($op(x, y.data))
end

#Block-wise broadcast 
broadcasted(f::Function, x::BlockMatrixTensor, y::AbstractMatrix) = BlockMatrixTensor(f.(blocks(x.data), y))
broadcasted(f::Function, x::BlockMatrixTensor, y::AbstractVector) = BlockMatrixTensor(f.(x, reshape(y, :, 1)))
broadcasted(f::Function, x::AbstractVecOrMat, y::BlockMatrixTensor) = f.(y, x)

function sum(x::BlockMatrixTensor; dims = Colon())
    # Blockwise sum
    if dims == 1:2
        sum(blocks(x.data))
    elseif dims == 1 || dims == 2
        BlockMatrixTensor(sum(blocks(x.data); dims))
    else
        # For now, treat all other cases as if summing the 4-Tensor
        sum(Array(x); dims = dims)
    end
end

function reshape(x::BlockMatrixTensor, dims::Vararg{Union{Colon, Int}, 4})
    #Reshape block-wise
    #TODO: Implement non-blockwise
    if dims[3] isa Colon && dims[4] isa Colon
        BlockMatrixTensor(reshape(blocks(x.data), dims[1], dims[2]))
    end
end

#'Flatten': converts BlockMatrixTensor to Matrix by removing block structure
# Mimics the reshape achieved by this for general 4-Tensors
flatten(x::BlockMatrixTensor) = hcat((vcat((x[i, j, :, :] for i = 1:size(x, 1))...) for j = 1:size(x, 2))...)
flatten(x::AbstractArray{T, 4}) where {T} = reshape(x, size(x, 1) * size(x, 3), size(x, 2) * size(x, 4))
