using Flux
using Flux: relu, crossentropy, logitcrossentropy, @epochs, testmode!, throttle
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
const ADAM_EPOCHS = 100
const SGD_EPOCHS = 10

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
net = Chain(Conv((5, 3), 3=>128, relu; pad=(2, 1)),
            x -> maxpool(x, (1,3)),
            Dropout(0.3),
            Conv((5, 3), 128=>128, relu; pad=(2, 1)),
            Dropout(0.3),
            Conv((5, 3), 128=>128, relu; pad=(2, 1)),
            Dropout(0.3),
            Conv((5, 3), 128=>128, relu; pad=(2, 1)),
            Dropout(0.3),
            Conv((5, 3), 128=>256, relu, pad=(2, 1)),
            Dropout(0.3),
            Conv((5, 3), 256=>256, relu, pad=(2, 1)),
            Dropout(0.3),
            Conv((5, 3), 256=>256, relu, pad=(2, 1)),
            Dropout(0.3),
            Conv((5, 3), 256=>256, relu, pad=(2, 1)),
            Dropout(0.3),
            Conv((5, 3), 256=>256, relu, pad=(2, 1)),
            Dropout(0.3),
            Conv((5, 3), 256=>256, relu, pad=(2, 1)),
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
    println("output 1: $(ms[:,1])")
    l = ctc(ms, y)
    return l
end

function ctctrain!(loss, data, opt; cb = () -> ())
    cb = runall(cb)
    opt = runall(opt)
    losses = Vector()
    batchLosses = Vector()
    counter = 0
    
    @progress for d in data
        l = loss(d...)
        push!(losses, Flux.Tracker.data(l))
        println("example loss: $(losses[end])")
        println("mean loss over time: $(mean(losses))")
        
        @interrupts Flux.Tracker.back!(l)
        cb() == :stop && break
        l = nothing
        gc()
        counter += 1
        if counter == BATCHSIZE
            opt()
            counter = 0
            gc()
        end
    end
    opt()
end

function main()
println("Gathering data")
Xs, Ys = readData(TRAINDIR)
data = collect(zip(Xs, Ys))
valData = gpu.(data[1:189])
trainData = data[190:end]
trainData = gpu.(trainData)
# data = data[21:end] # batch size = 20
p = params(net)
opt = ADAM(p, 10.0^-4)
println()
println("Training")
for i=1:ADAM_EPOCHS
    println("EPOCH $(i)")
    trainData = shuffle(trainData)
    valData = shuffle(valData)
    ctctrain!(loss, trainData, opt)
    println("Saving epoch results")
    BSON.@save "trackedloss100_epoch$(i).bson" net
    testmode!(net)
    print("Validating")
    println("Validation Phoneme Error Rate. $(evaluatePER(net, valData))")
    valLosses = Vector()
    for d in shuffle(valData)
        append!(valLosses, Flux.Tracker.data(loss(d...)))
    end
    println("Mean validation loss: $(mean(valLosses))")
    testmode!(net, false)
end
# println("Starting SGD")
# opt= SGD(p, 10.0^-5)
# for i=1:SGD_EPOCHS
#     println("EPOCH $(i)")
#     ctctrain!(loss, shuffle(trainData), opt)
#     BSON.@save "mel_rmsprop20sgd10_sgdepoch$(i).bson" net
#     print("Validating\r")
#     testmode!(net)
#     println("Validation Phoneme Error Rate. $(evaluatePER(net, gpu.(valData)))")
#     valLosses = Vector()
#     for d in shuffle(valData)
#         append!(valLosses, loss(d...)[1])
#     end
#     println("Mean validation loss: $(mean(valLosses))")
#     testmode!(net, false)
# end
end

main()
