using Flux
using Flux: relu, crossentropy, logitcrossentropy
using Flux: binarycrossentropy
using NNlib: σ_stable
using CuArrays
using FileIO

include("ctc.jl")

const TRAINDIR = "train"
const TESTDIR = "test"

# Convolutional section as defined in the paper
println("building conv section")

# Data needs to be width x height x channels x number
convSection = Chain(Conv((3, 5), 3=>128, relu; pad=(1, 2)),
                    x -> maxpool(x, (1,3)),
                    # Dropout(0.3),
                    Conv((3, 5), 128=>128, relu; pad=(1, 2)),
                    # Dropout(0.3),
                    Conv((3, 5), 128=>128, relu; pad=(1, 2)),
                    # Dropout(0.3),
                    Conv((3, 5), 128=>128, relu; pad=(1, 2)),
                    # Dropout(0.3),
                    Conv((3, 5), 128=>256, relu, pad=(1, 2)),
                    # Dropout(0.3),
                    Conv((3, 5), 256=>256, relu, pad=(1, 2)),
                    # Dropout(0.3),
                    Conv((3, 5), 256=>256, relu, pad=(1, 2)),
                    # Dropout(0.3),
                    Conv((3, 5), 256=>256, relu, pad=(1, 2)),
                    # Dropout(0.3),
                    Conv((3, 5), 256=>256, relu, pad=(1, 2)),
                    # Dropout(0.3),
                    Conv((3, 5), 256=>256, relu, pad=(1, 2)),
                    # Dropout(0.3)) |> gpu
                    # Dropout(0.3)) |> gpu
		    ) |> gpu

# Dense section comes after the convolutions in the paper
println("Building dense section")
denseSection = Chain(Dense(3328, 1024, relu),
                     Dense(1024, 1024, relu),
                     Dense(1024, 1024, relu),
                     Dense(1024, 61, identity),
                     softmax) |> gpu

"""
    model(x)

Makes class predictions for the data in `x`.

`x` is expected to be a 4D Array a width equal to the number of timesteps,
height equal to 41 (the number of filters), and with 3 channels (one for
the filterbank features, one for the delta coefficietns, and another for
the delta-delta coefficients). The last dimension is the batch size, which
is presently taken to be 1.

`x` is fed into the convolutional section, after which it's reshaped so that
each timestep can be fed into the fully-connected section for classpredictions
at each timestep.
"""
function model(x)
    afterConv = convSection(x)
    dims = size(afterConv)
    afterConv = reshape(afterConv, (dims[1], prod(dims[2:end])))
    ŷ = Vector()
    for i in 1:size(afterConv)[1]
        push!(ŷ, denseSection(afterConv[i,:]))
    end
    return ŷ
end

"""
    readData(dataDir)

Reads in each jld file contained in `dataDir` and normalizes the data.
"""
function readData(dataDir)
    fnames = [x for x in readdir(dataDir) if endswith(x, "jld")][1:200]

    Xs = Vector()
    Ys = Vector()

    for (i, fname) in enumerate(fnames)
        print(string(i) * "/" * string(length(fnames)) * "\r")
        x, y = load(joinpath(dataDir, fname), "x", "y")
        x .-= mean(x,2)
        x ./= std(x,2)
        x = reshape(x, (size(x)[1], 41, 3, 1))
	    x = gpu(x)
        y = gpu.([y[i,:] for i in 1:size(y,1)])
        push!(Xs, x)
        push!(Ys, y)
    end
    return (Xs, Ys)
end

"""
    loss(x, y)

Caclulates the connectionist temporal classification loss for `x` and `y`.
"""
function loss(x, y)
    println("calculating loss")
    m = model(x)
    l = ctc(m, y)
    println(l)
    println("loss: $(l)")
    l
end

"""
    evaluateAccuracy(data)

Calculates percent of correct classifications for each input/output in `data`.
"""
function evaluateAccuracy(data)
    correct = Vector()
    for (x, y) in data
        y = indmax.(y)
        ŷ = indmax.(model(x))
        correct = vcat(correct,
                     [ŷ_n == y_n for (ŷ_n, y_n) in zip(ŷ, y)])
    end
    sum(correct) / length(correct)
end

println("Gathering data")
Xs, Ys = readData(TRAINDIR)
data = collect(zip(Xs, Ys))
valData = data[1:189]
data = data[190:end]
p = params((convSection, denseSection))
opt = ADAM(p)
println()
println("Training")
Flux.train!(loss, data, opt)
print("Validating\r")
println("Validation acc. $(evaluateAccuracy(valData))")
