module DualArrays
export DualVector


"""
DualVector(value, jacobian)

reprents a vector of duals given by
    
    values + jacobian * [ε_1,…,ε_n].

For now the entries just return the values.
"""

struct DualVector{T} <: AbstractVector{T}
    value::Vector{T}
    jacobian::Matrix{T}
end

end # module DualArrays
