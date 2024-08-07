module DualArrays
export DualVector
using LinearAlgebra, ArrayLayouts, BandedMatrices, FillArrays
import Base: +, ==, getindex, size, broadcast, axes, broadcasted, show, sum, vcat

struct Dual{T, Partials <: AbstractVector{T}} <: Real
    value::T
    partials::Partials
end

==(a::Dual, b::Dual) = a.value == b.value && a.partials == b.partials

sparse_getindex(a...) = layout_getindex(a...)
sparse_getindex(D::Diagonal, k::Integer, ::Colon) = OneElement(D.diag[k], k, size(D,2))
sparse_getindex(D::Diagonal, ::Colon, j::Integer) = OneElement(D.diag[j], j, size(D,1))

"""
reprents a vector of duals given by
    
    values + jacobian * [ε_1,…,ε_n].

For now the entries just return the values.
"""

struct DualVector{T, M <: AbstractMatrix{T}} <: AbstractVector{Dual{T}}
    value::Vector{T}
    jacobian::M
    function DualVector(value::Vector{T},jacobian::M) where {T, M <: AbstractMatrix{T}}
        if(size(jacobian)[1] != length(value))
            x,y = length(value),size(jacobian)[1]
            throw(ArgumentError("vector length must match number of rows in jacobian.\n
            vector length: $x \n
            no. of jacobian rows: $y"))
        end
        new{T,M}(value,jacobian)
    end
end

function DualVector(value::AbstractVector, jacobian::AbstractMatrix)
    T = promote_type(eltype(value), eltype(jacobian))
    DualVector(convert(Vector{T}, value), convert(AbstractMatrix{T}, jacobian))
end

function getindex(x::DualVector, y::Int)
    Dual(x.value[y], sparse_getindex(x.jacobian,y,:))
end

function getindex(x::DualVector, y::UnitRange)
    newval = x.value[y]
    newjac = sparse_getindex(x.jacobian,y,:)
    DualVector(newval, newjac)
end
size(x::DualVector) = length(x.value)
axes(x::DualVector) = axes(x.value)
+(x::DualVector,y::DualVector) = DualVector(x.value + y.value, x.jacobian + y.jacobian)

broadcasted(::typeof(sin),x::DualVector) = DualVector(sin.(x.value),Diagonal(cos.(x.value))*x.jacobian)

function broadcasted(::typeof(*),x::DualVector,y::DualVector)
    newval = x.value .* y.value
    newjac = x.value .* y.jacobian + y.value .* x.jacobian
    DualVector(newval,newjac)
end

function sum(x::DualVector)
    n = length(x.value)
    Dual(sum(x.value), vec(sum(x.jacobian; dims=1)))
end

_jacobian(d::Dual) = permutedims(d.partials)
_jacobian(d::DualVector) = d.jacobian

function vcat(x::Union{Dual, DualVector}...)
    if length(x) == 1
        return x[1]
    end
    value = vcat((d.value for d in x)...)
    jacobian = vcat((_jacobian(d) for d in x)...)
    DualVector(value,jacobian)
end

show(io::IO,::MIME"text/plain", x::DualVector) = (print(io,x.value); print(io," + "); print(io,x.jacobian);print("𝛜"))
end
# module DualArrays
