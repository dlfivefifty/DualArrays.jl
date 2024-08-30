##
# Example: Heat Equation
#
# Goal: Differentiate solutions of the nonlinear heat equation:
#
#     uₜ = uₓₓ + λu²
#
# For some parameter λ both by ForwardDiff.jl and DualArrays.jl.
# Compare performance and results.
##

using LinearAlgebra, BandedMatrices, OrdinaryDiffEqs, Plots, 
ForwardDiff

U = 500
ΔU = 0.01
#Discretise u(x,t) into uₙ(t) = u(xₙ,t) in order to approximate as system of ODEs.
#Use bell curve initial condition
U₀ = exp.(collect(-U*ΔU:ΔU:U*ΔU).^2 * -0.5)

#Discrete laplacian with boundary conditions du₀/dt = duₙ/dt = 0
L = 1/ΔU^2 * BandedMatrix(-1 => [ones(eltype(U₀), length(U₀)-2) ; 0], 0 => [0 ; -2 * ones(eltype(U₀), length(U₀)-2) ; 0], 1 => [0 ; ones(eltype(U₀), length(U₀)-2)])

function solve_eq(λ)
    T = (0.0, 5.0)
    prob = ODEProblem(ode!, U₀, T, λ)
    solve(prob, Tsit5())
end

function ode!(du, u, λ, t)
    du .= L * u + λ * u .^ 2
end

ForwardDiff.derivative(solve_eq,0)