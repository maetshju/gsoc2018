using Flux
using Flux: relu
using FileIO

const TRAINDIR = "train"
const TESTDIR = "test"

convSection = Chain(Conv((3, 5), 3=>3, relu; pad = (2, 4)), # TODO: check that the padding here will keep the proper dimensions
                    Flux.maxpool((3,1)), # haven't checked if this is right yet
                    Conv(),
                    Conv(),
                    Conv(),
                    Conv(),
                    Conv(),
                    Conv(),
                    Conv(),
                    Conv(),
                    Conv())

function model(data)

end

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
        x = reshape(x, (size(x)[1], 41, 3))
        y = [y[i,:] for i in 1:size(y,1)]

        println(size(x))
        println(size(y))
        #y = [y[i,:] for i in 1:size(y,1)]
        push!(Xs, x)
        push!(Ys, y)
    end
    return (Xs, Ys)
end

loss(x, y) = sum(crossentropy.(net.(x), y))

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

data = readData(TRAINDIR)
