module DualArrays
export DualVector
using LinearAlgebra
import Base: +, getindex, size, broadcast, axes, broadcasted
"""
reprents a vector of duals given by
    
    values + jacobian * [ε_1,…,ε_n].

For now the entries just return the values.
"""

struct DualVector{T, M <: AbstractMatrix{T}} <: AbstractVector{T}
    value::Vector{T}
    jacobian::M
end

getindex(x::DualVector,y::Int) = x.value[y]
size(x::DualVector) = length(x.value)
axes(x::DualVector) = axes(x.value)
+(x::DualVector,y::DualVector) = DualVector(x.value + y.value, x.jacobian + y.jacobian)

broadcasted(::typeof(sin),x::DualVector) = DualVector(sin.(x.value),Diagonal(cos.(x.value))*x.jacobian)
end
# module DualArrays
