# TODO: support non-banded
sparsevcat(As...) = vcat(convert.(BandedMatrix,  As)...)

# Should Partials necessarily be a vector?
# For example, if we index a DualMatrix, should partials retain the shape
# of the perturbations, thereby giving a matrix of partials?
struct Dual{T, Partials <: AbstractArray{T}} <: Real
    value::T
    partials::Partials
end

zero(::Type{Dual{T}}) where {T} = Dual(zero(T), zeros(T,1))
promote_rule(::Type{Dual{T}},::Type{S}) where {T,S} = Dual{promote_type(T,S)}

==(a::Dual, b::Dual) = a.value == b.value && a.partials == b.partials
for op in (:+, :-)
    @eval $op(a::Dual, k::Real) = Dual($op(a.value, k), a.partials)
end
for op in (:*, :/)
    @eval $op(a::Dual, k::Real) = Dual($op(a.value, k), $op(a.partials,k))
end

-(x::Dual) = Dual(-1 * x.value, -1 * x.partials)

*(x::Dual, y::Dual) = Dual(x.value*y.value, x.value*y.partials + y.value*x.partials)
/(x::Dual, y::Dual) = x*Dual(y.value, -y.partials) / y.value^2

sparse_getindex(a...) = layout_getindex(a...)
sparse_getindex(D::Diagonal, k::Integer, ::Colon) = OneElement(D.diag[k], k, size(D,2))
sparse_getindex(D::Diagonal, ::Colon, j::Integer) = OneElement(D.diag[j], j, size(D,1))
sparse_getindex(s::AbstractSparseArray, idx...) = s[idx...]

sparse_getindex(b::BandedMatrix, k::Integer, ::Colon) = sparsevec(
    rowrange(b, k), b[k, rowrange(b, k)], size(b, 2)
    )
sparse_getindex(b::BandedMatrix, ::Colon, k::Integer) = sparsevec(
    colrange(b, k), b[colrange(b, k), k], size(b, 1)
    )

sparse_transpose(s::SparseVector) = sparse(ones(length(s.nzind)),s.nzind,s.nzval, 1, length(s))
sparse_transpose(o::OneElement{T,1}) where {T} = reshape(o, 1, :)
sparse_transpose(x::AbstractVector) = x'

"""
reprents an array of duals given by
    
    values + jacobian * 𝛜.

where 𝛜 is an array with dual parts corresponding to each entry of values.
"""

#Helper function to check valid dimensions of DualArray
function _checkdims(val, jac)
    s1, s2 = size(val), size(jac)
    if length(s1) > length(s2)
        return false
    end
    for (i, x) in enumerate(s1)
        if(x != s2[i])
            return false
        end
    end
    return true
end

struct DualArray{T, N, M, J <: AbstractArray{T, M}} <: AbstractArray{Dual{T}, N}
    value::AbstractArray{T, N}
    jacobian::J
    function DualArray(value::AbstractArray{T,N},jacobian::AbstractArray{T, M}) where {T, N, M}
        if(!_checkdims(value,jacobian))
            s1, s2 = size(value), size(jacobian)
            throw(ArgumentError("N-dimensional array must be of equal size to first N dimensions of jacobian. \n\nGot $s1, $s2"))
        end
        new{T,N,2*N,typeof(jacobian)}(value,jacobian)
    end
end

function DualArray(value::AbstractArray{T,N}, jacobian::AbstractArray{S,M}) where {S,T,N,M}
    t = promote_type(T, S)
    DualArray(convert(AbstractArray{t,N}, value), convert(AbstractArray{t, M}, jacobian))
end

#Aliases and Constructors
const DualVector{T, M} = DualArray{T, 1, 2, M} where {T, M <: AbstractMatrix{T}}
const DualMatrix{T, M} = DualArray{T, 2, 4, M} where {T, M <: AbstractArray{T, 4}}

DualVector(x::AbstractVector, j::AbstractMatrix) = DualArray(x, j)
DualVector(x::AbstractVector, j::ComponentArray) = DualArray(x, j)
DualMatrix(x::AbstractMatrix, j::AbstractArray{T,4}) where {T} = DualArray(x, j)

function DualMatrix(x::AbstractMatrix)
    val = [y.value for y in x]
    jac_blocks = [y.partials for y in x]
    DualMatrix(val, BlockMatrixTensor(jac_blocks))
end

function getindex(x::DualVector, y::Int)
    Dual(x.value[y], sparse_getindex(x.jacobian,y,:))
end
function getindex(x::DualVector, y::UnitRange)
    newval = x.value[y]
    newjac = sparse_getindex(x.jacobian,y,:)
    #newjac = layout_getindex(x.jacobian,y,:)
    DualVector(newval, newjac)
end
getindex(x::DualVector, ::Colon) = x

function getindex(x::DualArray, y::Int...)
    idx = ntuple(i -> i > length(y) ? Colon() : y[i], 2*ndims(x))
    Dual(x.value[y...], x.jacobian[idx...])
end

for op in (:size, :axes, :ndims)
    @eval $op(x::DualArray) = $op(x.value)
end

+(x::DualVector,y::DualVector) = DualVector(x.value + y.value, x.jacobian + y.jacobian)
-(x::DualVector,y::DualVector) = DualVector(x.value - y.value, x.jacobian - y.jacobian)
+(x::DualVector,y::Vector) = DualVector(x.value + y, x.jacobian)
-(x::DualVector,y::Vector) = DualVector(x.value - y, x.jacobian)
+(x::Vector,y::DualVector) = DualVector(y.value + x, y.jacobian)
-(x::Vector,y::DualVector) = DualVector(x - y.value, -y.jacobian)
*(x::AbstractMatrix, y::DualVector) = DualVector(x * y.value, x * y.jacobian)
*(x::Number, y::DualVector) = DualVector(x * y.value, x * y.jacobian)
*(x::DualVector, y::Dual) = DualVector(x.value * y.value, y.value * x.jacobian + x.value * y.partials')
/(x::DualVector, y::Number) = DualVector(x.value / y, x.jacobian/ y)
/(x::DualVector, y::Dual) = x * Dual(y.value, -y.partials) / y.value^2

##
# Note about 4-tensor by matrix multiplication:
#   We can consider matrix-vector multiplication as follows:
#
#       (Ax)[i] = A[i,:] . x
#   
#   This can be generalised to higher dimensions (B a 4-tensor, X a matrix):
#
#       (BX)[i,j] = B[i,j,:,:] . X
#
#   This is useful since if we consider 4-tensors dimension n × m × s × t
#   equivalent to linear maps from n × m matrices to s × t matrices, this 
#   gives us multiplication analogous to applying the linear transform on
#   a matrix in the same way that matrix-vector multiplication applies the
#   linear transformation to the vector. This also gives us a way to 
#   'expand' the Dual part of a DualMatrix.
##

# Assumes that the total number of epsilons is equal, i.e the array of
# perturbations are reshapes of each other.
function *(x::DualMatrix, y::DualVector)
    val = x.value * y.value
    jac = x.value * y.jacobian + flatten(sum(x.jacobian .* y.value'; dims = 2))
    DualVector(val, jac)
end

function *(x::DualMatrix, y::AbstractVector)
    val = x.value * y
    t = promote_type(eltype(x.value), eltype(y))
    jac = zeros(t, length(val), size(x.jacobian, 3) * size(x.jacobian, 4))
    n, m = size(jac, 1), size(x.value, 2)
    for i = 1:n
        new_row = zeros(size(jac, 2))
        for j = 1:m
            new_row += y[j] * vec(x.jacobian[i, j, :, :])
        end
        jac[i,:] = new_row
    end
    DualVector(val, jac)
end

function broadcasted(f::Function,d::DualVector)
    jvals = zeros(eltype(d.value), length(d.value))
    for (i, x) = enumerate(d.value)
        _, df = frule((ZeroTangent(), 1),f, x)
        jvals[i] = df
    end
    DualVector(f.(d.value), Diagonal(jvals)*d.jacobian)
end

function broadcasted(::typeof(*),x::DualVector,y::DualVector)
    newval = x.value .* y.value
    newjac = x.value .* y.jacobian + y.value .* x.jacobian
    DualArray(newval,newjac)
end

function broadcasted(::typeof(*),x::AbstractArray,y::DualArray)
    newval = x .* y.value
    newjac = x .* y.jacobian
    DualArray(newval,newjac)
end

function broadcasted(::typeof(*),x::DualArray,y::AbstractArray)
    newval = x.value .* y
    newjac = y .* x.jacobian
    DualArray(newval,newjac)
end

function broadcasted(::typeof(^),x::DualVector,n::Int)
    newval = x.value .^ n
    newjac = n * x.value .^ (n - 1) .* x.jacobian
    DualVector(newval,newjac)
end

broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::DualVector, ::Val{n}) where n = broadcasted(^, x, n)

###
# sum (higher-dimensional case assuming that shape of perturbations is preserved)
###
function sum(x::DualVector)
    Dual(sum(x.value), vec(sum(x.jacobian; dims=1)))
end

function sum(x::DualArray)
    Dual(sum(x.value), sum(x.jacobian; dims=1:ndims(x)))
end

_jacobian(d::Dual) = permutedims(d.partials)
_jacobian(d::DualVector) = d.jacobian
_jacobian(d::DualVector, ::Int) = d.jacobian
_jacobian(x::Number, N::Int) = Zeros{typeof(x)}(1, N)

_value(d::DualVector) = d.value
_value(x::Number) = x

function vcat(x::Union{Dual, DualVector}...)
    if length(x) == 1
        return x[1]
    end
    value = vcat((d.value for d in x)...)
    jacobian = vcat((_jacobian(d) for d in x)...)
    DualVector(value,jacobian)
end

function vcat(a::Dual ,x::DualVector, b::Dual)
    val = vcat(a.value, x.value, b.value)
    jac = sparsevcat(_jacobian(a), x.jacobian, _jacobian(b))
    DualVector(val, jac)
end

function vcat(a::Real ,x::DualVector, b::Real)
    cols = size(x.jacobian,2)
    val = vcat(a, x.value, b)
    jac = sparsevcat(_jacobian(a, cols), x.jacobian, _jacobian(b, cols))
    DualVector(val, jac)
end

###
#
# enable converting between DualVector and DualMatrix.
# the 4D jacobian can be thought of as a generalisation of the 2D jacobian:
#
#   n × m 2D jacobian:  J[i,j] = ∂fᵢ / ∂xⱼ
#   n × m × s × t 4D jacobian: J[i, j, k, l] = ∂f[i, j] / δx[k, l]
#
###

# reshape does not preserve shape of perturbations (since jacobian can have
# any number of columns)
_tomatrix(v::AbstractVector) = reshape(v, :, 1)

function reshape(x::DualVector,dims::Vararg{Int, 2})
    val = reshape(x.value, dims...)
    blocked_jac = BlockedMatrix(x.jacobian, fill(1, length(x)), [size(x.jacobian, 2)])
    jac = reshape(BlockMatrixTensor(blocked_jac), dims..., :, :)
    DualMatrix(val, jac)
end

_blockvec(x::AbstractArray{T,4}) where {T} = reshape(x, size(x, 1) * size(x, 2), size(x, 3) * size(x, 4))
_blockvec(x::BlockMatrixTensor) = vcat(blocks(x.data)...)

function vec(x::DualArray)
    val = vec(x.value)
    jac = parent(reshape(x.jacobian, length(val), 1, :, :).data)
    DualVector(val, jac)
end

show(io::IO,::MIME"text/plain", x::DualArray) = (print(io,x.value); print(io," + "); print(io,x.jacobian);print("𝛜"))
show(io::IO,::MIME"text/plain", x::DualVector) = (print(io,x.value); print(io," + "); print(io,x.jacobian);print("𝛜"))
show(io::IO,::MIME"text/plain", x::Dual) = (print(io,x.value); print(io," + "); print(io,x.partials);print("𝛜"))