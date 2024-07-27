module DualArrays
export DualVector
using LinearAlgebra, ArrayLayouts, BandedMatrices
import Base: +, ==, getindex, size, broadcast, axes, broadcasted, show, sum
import ChainRules: frule, ZeroTangent
import DiffRules: diffrule

struct Dual{T} <: Real
    value::T
    partials::Vector{T}
end

==(a::Dual, b::Dual) = a.value == b.value && a.partials == b.partials

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
    Dual(x.value[y], x.jacobian[y,:])
end

function getindex(x::DualVector, y::UnitRange)
    newval = x.value[y]
    newjac = layout_getindex(x.jacobian,y,:)
    DualVector(newval, newjac)
end
size(x::DualVector) = length(x.value)
axes(x::DualVector) = axes(x.value)
+(x::DualVector,y::DualVector) = DualVector(x.value + y.value, x.jacobian + y.jacobian)

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
    DualVector(newval,newjac)
end

function sum(x::DualVector)
    n = length(x.value)
    Dual(sum(x.value), vec(sum(x.jacobian; dims=1)))
end

show(io::IO,::MIME"text/plain", x::DualVector) = (print(io,x.value); print(io," + "); print(io,x.jacobian);print("𝛜"))
end
# module DualArrays
