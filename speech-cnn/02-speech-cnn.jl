using Flux
using Flux: relu, crossentropy, logitcrossentropy, @epochs, testmode!
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
const ADAM_EPOCHS = 20
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
            
            
# function model(x)
#     println("sending to conv")
#     afterConv = convSection(x)
#     exit()
#     predictions = Vector()
#     for i=1:size(x, 4)
#         preDense = x[:, :, :, i]
#         preDense = transpose(preDense, size(preDense, 1), prod(size(preDense)[2:end]))
#         afterDense = denseSection(x)
#         push!(predictions, afterDense)
#     end
#     return predictions
# end
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
#     idx = 1
#     lenData = length(data)
#     while idx <= lenData
#         batch = data[idx:min(lenData, idx+(BATCHSIZE-1))]
#         
#         batchX = gpu(cat(4, [x for (x, y) in batch]...))
#         println(size(batchX))
#         batchY = [y for (x, y) in batch]
#         ls, gs, ms = loss(batchX, batchY)
#         idx += BATCHSIZE
#     end
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
#             clamp!.(params(model), -20, 20)
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
valData = gpu.(data[1:189])
# valData = data[1:10] # this when each item is a batch of 20
trainData = data[190:end]
# idx = 1
# lenData = length(trainData)
# longestTimeSteps = maximum([size(x, 1) for (x, y) in trainData])
# for (i, (x, y)) in enumerate(trainData)
#     stepsToAdd = longestTimeSteps - size(x, 1)
#     x = vcat(x, zeros(stepsToAdd, size(x)[2:end]...))
#     trainData[i] = (x, y)
#     # y doesn't matter because they all end in silence
# end
trainData = gpu.(trainData)
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
    BSON.@save "adam2sgd10_adamepoch$(i).bson" net
    testmode!(net)
    print("Validating\r")
    println("Validation Phoneme Error Rate. $(evaluatePER(net, gpu.(valData)))")
    valLosses = Vector()
    for d in shuffle(valData)
        append!(valLosses, loss(d...)[1])
    end
    println("Mean validation loss: $(mean(valLosses))")
    testmode!(net, false)
end
println("Starting SGD")
opt= SGD(p, 10.0^-5)
for i=1:SGD_EPOCHS
    println("EPOCH $(i)")
    ctctrain!(loss, shuffle(trainData), opt)
    BSON.@save "adam2sgd10_sgdepoch$(i).bson" net
    print("Validating\r")
    testmode!(net)
    println("Validation Phoneme Error Rate. $(evaluatePER(net, gpu.(valData)))")
    valLosses = Vector()
    for d in shuffle(valData)
        append!(valLosses, loss(d...)[1])
    end
    println("Mean validation loss: $(mean(valLosses))")
    testmode!(net, false)
end
end

main()
