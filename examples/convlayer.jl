using LinearAlgebra, MLDatasets, Plots, DualArrays, Random, FillArrays

#GOAL: Implement and differentiate a convolutional neural network layer
function convlayer(img, ker, xstride = 1, ystride = 1)
    n, m = size(ker)
    t = eltype(ker)

    n2, m2 = size(img)
    n3, m3 = div(n2-n+1,xstride), div(m2-m+1,ystride)
    fmap = zeros(promote_type(eltype(img), t), n3, m3)
    #Apply kernel to section of image
    for i= 1:xstride:n3,j = 1:ystride:m3
        ft = img[i:i+n-1,j:j+m-1] .* ker
        fmap[i,j] = sum(ft)
    end
    fmap
end

function softmax(x)
    s = sum(exp.(x))
    exp.(x) / s
end

function dense_layer(W, b, x, f::Function = identity)
    ret = W*x
    println("Multiplication complete")
    ret += b
    println("Addition Complete")
    f(ret)
end

function cross_entropy(x, y)
    -sum(y .* log.(x))
end

function model_loss(x, y, w)
    ker = reshape(w[1:9], 3, 3)
    weights = reshape(w[10:6769], 10, 676)
    biases = w[6770:6779]
    println("Reshape Complete")
    l1 = vec(DualMatrix(convlayer(x, ker)))
    println("Conv layer complete")
    l2 = dense_layer(weights, biases, l1, softmax)
    println("Dense Layer Complete")
    target = OneElement(1, y+1, 10)
    loss = cross_entropy(l2, target)
    println("Loss complete")
    loss.value, loss.partials
end

function train_model()
    p = rand(6779)
    epochs = 1000
    lr = 0.02
    dataset = MNIST(:train)

    for i = 1:epochs
        train, test = dataset[i]
        d = DualVector(p, I(6779))

        loss, grads = model_loss(train, test, d)
        println(loss)
        p = p - lr * grads
    end
end

train_model()


