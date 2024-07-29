##
# Solve pendulum ODE:

# x'' + sin(x) = 0

# via discretisation and Newton's method.
##

using LinearAlgebra, ForwardDiff, Plots, DualArrays

#Boundary Conditions
a = 0.1
b = 0.0

#Time step, Time period and number of x for discretisation.
ts = 0.001
Tmax = 5.0
N = Int(Tmax/ts) - 1

#LHS of ode
function f(x)
    n = length(x)
    D = Tridiagonal([ones(Float64, n) / ts ; 0.0], [1.0; -2ones(Float64, n) / ts; 1.0], [0.0; ones(Float64, n) / ts])
    (D * [a; x; b])[2:end-1] + sin.(x)
end

#Newtons method using ForwardDiff.jl
function newton_method_forwarddiff(f, x0, n)
    x = x0
    for i = 1:n
        ∇f = ForwardDiff.jacobian(f, x)
        x = x - ∇f \ f(x)
    end
    x
end

function newton_method_dualvector(f, x0, n)
    x = x0
    l = length(x0)
    for i = 1:n
        ∇f = f(DualVector(x, Matrix(I, l, l))).jacobian
        x = x - ∇f \ f(x)
    end
    x
end

#Initial guess
x0 = zeros(Float64, N)

#Solve and plot both solution and LHS ('deviation' from system)
@time sol = newton_method_forwarddiff(f, x0, 100)
@time sol = newton_method_dualvector(f, x0, 100)
plot(0:ts:Tmax, [a; sol; b])
plot!(0:ts:Tmax, [0; f(sol); 0])

