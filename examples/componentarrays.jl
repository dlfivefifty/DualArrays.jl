using ComponentArrays, DualArrays


function denselayer(layer, x)
    layer.weight * x + layer.bias
end



# We want to different with respect to b the following function:

function foo(b)
    denselayer(ComponentVector(weight = [1 2; 3 4], bias = b), [5,6])
end

# This works as-is:

foo([5,6])
foo(DualVector([5,6], I(2)))

b = DualVector([5,6], I(2))
w = DualArray([1 2; 3 4], FourTensorIdentity((2,2), (2,2)))
ComponentVector(weight = w, bias = b)


denselayer(ComponentVector(weight = [1 2; 3 4], bias = b), [5,6])

f = x -> x * derivative(y -> x + y, 1)


g = (x,y) -> x + y
f = x -> x * g(x, 1)



function f(a, b)
    a + b
end

