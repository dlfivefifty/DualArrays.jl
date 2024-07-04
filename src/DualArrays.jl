module DualArrays
export DualVector
using LinearAlgebra, ForwardDiff
import Base: +, getindex, size, broadcast, axes, broadcasted, show, sum

"""
reprents a vector of duals given by
    
    values + jacobian * [ε_1,…,ε_n].

For now the entries just return the values.
"""

struct DualVector{T, M <: AbstractMatrix{T}} <: AbstractVector{T}
    value::Vector{T}
    jacobian::M
end

function getindex(x::DualVector,y::Int)
    ForwardDiff.Dual(x.value[y],Tuple(x.jacobian[y,:]))
end

function getindex(x::DualVector,y::UnitRange)
    newval = x.value[y]
    newjac = x.jacobian[y,:]
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
    ForwardDiff.Dual(sum(x.value),Tuple(sum(x.jacobian[:,i]) for i=1:n))
end

show(io::IO,::MIME"text/plain", x::DualVector) = (print(io,x.value); print(io," + "); print(io,x.jacobian);print("𝛜"))
end
# module DualArrays
