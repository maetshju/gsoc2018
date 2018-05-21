using Flux
using Flux: relu, crossentropy
using FileIO

const TRAINDIR = "train"
const TESTDIR = "test"

# Data needs to be width x height x channels x number
convSection = Chain(Conv((3, 5), 3=>128, relu; pad=(1, 2)),
                    x -> maxpool(x, (1,3)), # haven't checked if this is right yet; want to pool over height
                    Dropout(0.3),
                    Conv((3, 5), 128=>128, relu; pad=(1, 2)),
                    Dropout(0.3),
                    Conv((3, 5), 128=>128, relu; pad=(1, 2)),
                    Dropout(0.3),
                    Conv((3, 5), 128=>128, relu; pad=(1, 2)),
                    Dropout(0.3),
                    Conv((3, 5), 128=>256, relu, pad=(1, 2)),
                    Dropout(0.3),
                    Conv((3, 5), 256=>256, relu, pad=(1, 2)),
                    Dropout(0.3),
                    Conv((3, 5), 256=>256, relu, pad=(1, 2)),
                    Dropout(0.3),
                    Conv((3, 5), 256=>256, relu, pad=(1, 2)),
                    Dropout(0.3),
                    Conv((3, 5), 256=>256, relu, pad=(1, 2)),
                    Dropout(0.3),
                    Conv((3, 5), 256=>256, relu, pad=(1, 2)),
                    Dropout(0.3))

denseSection = Chain(Dense(3328, 1024, relu),
                     Dense(1024, 1024, relu),
                     Dense(1024, 1024, relu),
                     Dense(1024, 61, identity),
                     softmax)

function model(data)
    afterConv = convSection(data)
    dims = size(afterConv)
    afterConv = reshape(afterConv, (dims[1], prod(dims[2:end])))
    #predictions = mapslices(denseSection, afterConv, [2])
    ŷ = Vector()
    for i in 1:size(afterConv)[1]
        push!(ŷ, denseSection(afterConv[i,:]))
    end
    return ŷ
end

function readData(data_dir)
    fnames = [x for x in readdir(data_dir) if endswith(x, "jld")][1:200]

    Xs = Vector()
    Ys = Vector()

    for (i, fname) in enumerate(fnames)
        print(string(i) * "/" * string(length(fnames)) * "\r")
        x, y = load(joinpath(data_dir, fname), "x", "y")
        x .-= mean(x,2)
        x ./= std(x,2)
        x = reshape(x, (size(x)[1], 41, 3, 1))
        y = [y[i,:] for i in 1:size(y,1)]

        push!(Xs, x)
        push!(Ys, y)
    end
    return (Xs, Ys)
end

function loss(x, y)
    l = sum(crossentropy.(model(x), y))
    println(l)
    l
end

function predict(x)
    ŷ = model(x)
    return ŷ
end

function evaluateAccuracy(data)
    correct = Vector()
    for (x, y) in data
        y = indmax.(y)
        ŷ = indmax.(predict(x))
        correct = vcat(correct,
                        [ŷ_n == y_n for (ŷ_n, y_n) in zip(ŷ, y)])
    end
    sum(correct) / length(correct)
end

println("Gathering data")
Xs, Ys = readData(TRAINDIR)
data = collect(zip(Xs, Ys))
valData = data[1:100]
data = data[101:end]
opt = ADAM(params((convSection, denseSection)))
println()
println("Training")
Flux.train!(loss, data, opt)
print("Validating\r")
println("Validation acc. $(evaluateAccuracy(valData))")
