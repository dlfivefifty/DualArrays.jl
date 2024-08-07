##
# Solve pendulum ODE:

# x'' + sin(x) = 0

# via discretisation and Newton's method.
##


using LinearAlgebra, ForwardDiff, Plots, DualArrays, FillArrays

#Boundary Conditions
a = 0.1
b = 0.0

#Time step, Time period and number of x for discretisation.
ts = 0.1

Tmax = 5.0
N = Int(Tmax/ts) - 1

#LHS of ode
function f(x)
    n = length(x)
    D = Tridiagonal([ones(n) / ts ; 0.0], [1.0; -2ones(n) / ts; 1.0], [0.0; ones(n) / ts])
    (D * [a; x; b])[2:end-1] + sin.(x)
end


function f(u, a, b, Tmax)
    h = Tmax/(length(u)-1)
    [u[1] - a;
     (u[1:end-2] - 2u[2:end-1] + u[3:end])/h^2 + sin.(u[2:end-1]);
     u[end] - b]
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
        ∇f = f(DualVector(x, Eye(l))).jacobian
        x = x - ∇f \ f(x)
    end
    x
end

function newton_method_dualvector2(f, x0, n)
    x = x0
    l = length(x0)
    for i = 1:n
        ∇f = f(DualVector(x, Eye(l)), a, b, Tmax).jacobian
        x = x - ∇f \ f(x)
    end
    x
end

#Initial guess
x0 = zeros(N)

#Solve and plot both solution and LHS ('deviation' from system)
@time sol1 = newton_method_forwarddiff(f, x0, 100);
@time sol2 = newton_method_dualvector(f, x0, 100);
@test sol1 ≈ sol2
@time sol = newton_method_dualvector2(f, x0, 100);

x0 = zeros(N); x0[1] = a; x0[end] = b;

plot(0:ts:Tmax, [a; sol; b])
plot!(0:ts:Tmax, [0; f(sol); 0])
