using Flux
using Flux: binarycrossentropy, throttle
using JLD

const TRAINDIR = "train"
const TESTDIR = "test"

net = Chain(Dense(123, 64, relu),
            Dense(64, 1, σ),
            )

function readData(data_dir)
    fnames = [x for x in readdir(data_dir) if endswith(x, "jld")]

    Xs = Vector()
    Ys = Vector()

    for (i, fname) in enumerate(fnames)
        print(string(i) * "/" * string(length(fnames)) * "\r")
        x, y = load(joinpath(data_dir, fname), "x", "y")
        # Want to pull /eh/ and /sh/ frames out of the data. Base on the
        # dictionary mapping from the println(size(y))extraction:
        # /sh/ = 10
        # /eh/ = 3
        sh = [i == 10 for i in 1:61]
        eh = [i == 3 for i in 1:61]
        shOrEh = [y[i,:] == sh || y[i,:] == eh for i in 1:size(y,1)]
        idxs = find(y -> y == true, shOrEh)
        #idxs = [y -> y == sh || y == eh, y[i,:]) for i in size(y,1)]
        if length(idxs) == 0
            continue
        end
        x = x[idxs,:]
        x .-= mean(x,2)
        x ./= std(x,2)
        y = y[idxs,:]
        y = [y[i,:] == sh ? 1 : 0 for i in 1:size(y,1)]
        x = [x[i,:] for i in 1:size(x,1)]
        #y = [y[i,:] for i in 1:size(y,1)]
        push!(Xs, x)
        push!(Ys, y)
    end
    return (Xs, Ys)
end

loss(x, y) = sum(binarycrossentropy.(net.(x), y))

function predict(x)
    ŷ = net(x)
    return ŷ
end

function evaluateAccuracy(data)
    correct = Vector()
    for (x, y) in data
        y = indmax.(y)
        ŷ = indmax.(predict.(x))
        correct = vcat(correct,
                        [ŷ_n == y_n for (ŷ_n, y_n) in zip(ŷ, y)])
    end
    sum(correct) / length(correct)
end

println("reading data")
Xs, Ys = readData(TRAINDIR)
println()
data = collect(zip(Xs, Ys))
valData = data[1:100]
data = data[101:end]
opt = ADAM(params(net))
println("training")
Flux.train!(loss, data, opt)
print("Validation tests\r")
valAcc = evaluateAccuracy(valData)
println("Validation acc. $(valAcc)")

# Clean up some memory

valData = 0
data = 0
Xs = 0
Ys = 0
gc()

print("Testing\r")
XsTest, YsTest = readData(TESTDIR)
testData = collect(zip(XsTest, YsTest))
testAcc = evaluateAccuracy(testData)
println("Test acc. $(testAcc)")
