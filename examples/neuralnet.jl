import Lux: relu
using DualArrays, LinearAlgebra, Plots

#GOAL: Learn exp() using simple one-layer relu neural network.

N = 20
d = 0.1

#Domain over which to learn exp
data = collect(d:d:N*d)

function model(a, b, x)
    return relu.(a .* x + b)
end

function model_loss(w)
     sum((model(w[1:N], w[(N+1):end], data) - exp.(data)) ^ 2)
end

function gradient_descent(n, lr = 0.01)
    weights = ones(Float64, 2 * N)
    for i = 1:n
        dw = DualVector(weights, Matrix(I, 2 * N, 2 * N))
        grads = model_loss(dw).partials
        weights -= lr * grads
    end
    model(weights[1:N], weights[(N + 1):end], data)
end

plot(d:d:N*d, gradient_descent(100))
