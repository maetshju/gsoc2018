using Flux
using Flux: relu, crossentropy, logitcrossentropy, @epochs
using Flux.Tracker: back!
using Flux.Optimise: runall, @interrupts, SGD
# using NNlib: σ_stable, logsoftmax
using CuArrays
using JLD, BSON
using Juno

include("warp-ctc.jl")
include("utils.jl")

const TRAINDIR = "train"
const TESTDIR = "test"
const EPS = 1e-7
const BATCHSIZE = 20
const ADAM_EPOCHS = 10
const SGD_EPOCHS = 20

println("Building network")

poolX(x) = maxpool(x, (1,3))
reshapeX(x) = transpose(reshape(x, size(x, 1), prod(size(x)[2:end])))

"""
    net(x)

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
net = Chain(Conv((3, 5), 3=>128, relu; pad=(1, 2)),
            x -> maxpool(x, (1,3)),
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
            Dropout(0.3),
            x -> transpose(reshape(x, size(x, 1), prod(size(x)[2:end]))),
            Dense(3328, 1024, relu),
            Dropout(0.3),
            Dense(1024, 1024, relu),
            Dropout(0.3),
            Dense(1024, 1024, relu),
            Dropout(0.3),
            Dense(1024, 62),
            logsoftmax) |> gpu

"""
    loss(x, y)

Caclulates the connectionist temporal classification loss for `x` and `y`.
"""
function loss(x, y)
    ms = net(x)
#     println("output 1: $(ms[1,:])")
    ls, gs = ctc(cpu(Flux.Tracker.data(ms)), y)
    return ls, gs, ms
end

function ctctrain!(loss, data, opt; cb = () -> ())
    cb = runall(cb)
    opt = runall(opt)
    losses = Vector()
    counter = 0
    @progress for d in data
        ls, gs, ms = loss(d...)
        push!(losses, mean(ls))
        println("example loss: $(losses[end])")
        println("mean loss over time: $(mean(losses))")
        
        @interrupts Flux.Tracker.back!(ms[1:end], gs[1:end])
        cb() == :stop && break
        ls = nothing
        gs = nothing
        ms = nothing
        
        counter += 1
        if counter == BATCHSIZE
            opt()
            counter = 0
        end
    end
    opt()
    println("mean epoch loss: $(mean(losses))")
end

function main()
println("Gathering data")
Xs, Ys = readData(TRAINDIR)
data = collect(zip(Xs, Ys))
valData = data[1:189]
# valData = data[1:10] # this when each item is a batch of 20
trainData = gpu.(data[190:end])
# data = data[21:end] # batch size = 20
p = params(net)
opt = ADAM(p, 10.0^-4)
println()
println("Training")
# Flux.train!(loss, data, opt)
# println("EPOCH 1")
# ctctrain!(loss, trainData, opt)
for i=1:ADAM_EPOCHS
    println("EPOCH $(i)")
    ctctrain!(loss, shuffle(trainData), opt)
    BSON.@save "soft_net30epochs_adamepoch$(i).bson" net
    print("Validating\r")
    println("Validation Phoneme Error Rate. $(evaluatePER(net, gpu.(valData)))")
    valLosses = Vector()
    for d in shuffle(valData)
        append!(valLosses, loss(d...)[1])
    end
    println("Mean validation loss: $(mean(valLosses))")
end
println("Starting ADAM5")
opt= ADAM(p, 10.0^-5)
for i=1:SGD_EPOCHS
    println("EPOCH $(i)")
    ctctrain!(loss, shuffle(trainData), opt)
    BSON.@save "soft_net30epochs_sgdepoch$(i).bson" net
    print("Validating\r")
#     println("Validation Phoneme Error Rate. $(evaluatePER(net, gpu.(valData)))")
    valLosses = Vector()
    for d in shuffle(valData)
        append!(valLosses, loss(d...)[1])
    end
    println("Mean validation loss: $(mean(valLosses))")
end
end

main()
