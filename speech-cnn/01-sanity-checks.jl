using Flux
using Flux: crossentropy
using JLD

const TRAINDIR = "train"
const TESTDIR = "test"

net = Chain(Dense(123, 64),
            Dense(64, 2),
            softmax)

function readData(data_dir)
    fnames = [x for x in readdir(data_dir) if endswith(x, "jld")]

    Xs = Vector()
    Ys = Vector()

    for (i, fname) in enumerate(fnames)
        print(string(i) * "/" * string(length(fnames)) * "\r")
        x, y = load(joinpath(data_dir, fname), "x", "y")
        # Want to pull /eh/ and /sh/ frames out of the data. Base on the
        # dictionary mapping from the extraction:
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
        y = y[idxs]
        x = [x[i,:] for i in 1:size(x,1)]
        y = [y[i,:] for i in 1:size(y,1)]
        push!(Xs, x)
        push!(Ys, y)
    end
    return (Xs, Ys)
end

loss(x, y) = sum(crossentropy.(net.(x), y))

println("reading data")
Xs, Ys = readData(TRAINDIR)
println()
data = collect(zip(Xs, Ys))
opt = ADAM(params(net))
println(size(Xs[1][1]))
println("training")
Flux.train!(loss, data, opt)
