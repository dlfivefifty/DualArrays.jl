module DualArrays

export DualVector, DualArray, DualMatrix, BlockMatrixTensor
export dropzeros

import Base: +, ==, getindex, size, broadcast, axes, broadcasted, show, sum,
             vcat, convert, *, -, ^, /, ndims, hcat, vec, promote_rule, zero,
             reshape, setindex!
using LinearAlgebra, ArrayLayouts, BandedMatrices, FillArrays, ComponentArrays, SparseArrays
import ChainRules: frule, ZeroTangent

include("BlockMatrixTensor.jl")
include("DualArray.jl")

end
# module DualArrays