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
            softmax) |> gpu

"""
    loss(x, y)

Caclulates the connectionist temporal classification loss for `x` and `y`.
"""
function loss(x, y)
#     println("calculating loss")
#     ms = model.(x)
    ms = net(x)
#     println(typeof(Flux.Tracker.data(ms[1])))
#     mat = hcat([cpu(Flux.Tracker.data(x)) for x in ms]...)
#     println(typeof(ms[1]))
    # ls = ctc.(ms, y; gpu=true, eps=true)
#     ls = ctc.(ms, y)
    ls, gs = ctc(cpu(Flux.Tracker.data(ms)), y)
    #ls = Vector()
#     for (m, yI) in zip(ms, y)
#         push!(ls, ctc(m, yI; gpu=true, eps=true))
#     end
    # println(l)
    #println("mean loss: $(l/min(50, length(y)))")
#     println(size(ls))
#     println(size(gs))
#     println("mean loss: $(mean(ls))")
    mean(ls)
    return ls, gs, ms
end

function ctctrain!(loss, data, opt, parameters; cb = () -> ())
    cb = runall(cb)
    opt = runall(opt)
    losses = Vector()
    counter = 0
    @progress for d in data
        ls, gs, ms = loss(d...)
        push!(losses, mean(ls))
#         println("l1: $(ls[1])")
#         println("ls: $(ls)")
        println("example loss: $(losses[end])")
        println("mean loss over time: $(mean(losses))")
#         println("blanks: $(gs[62,:])")
#         println("silences: $(gs[1,:])")
#         exit()
#         println(parameters[end-1].grad)
        println(typeof(gs))
        println(typeof(ms[1]))
        println(size(ms[1]))
#         for i=1:length(ms)
# #             println(typeof(ms[i]))
# #             println(typeof(gs[:,i]))
# #             for j=1:length(ms[i])
# #                 @interrupts Flux.Tracker.back!(ms[i][j], gs[j,i])
# #             end
#             @interrupts Flux.Tracker.back!(ms[i][1:end], gs[:,i][1:end])
#         end
        
        @interrupts Flux.Tracker.back!(ms[1:end], gs[1:end])
#         println(ms.tracker.grad)
#         for (p, g) in zip(ms[:,end], gs[:,end])
#             @interrupts back!(p, g)
#         end
#         println(parameters[end-1].grad)
#         println(ms.tracker.grad)
#         for i=1:size(ms, 2)
#             @interrupts Flux.Tracker.back!(ms[:,i], gs[:,i])
#         end
#         opt()
        cb() == :stop && break
        ls = nothing
        gs = nothing
        ms = nothing
        
        counter += 1
        if counter == 20
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
chunkSize = 200
ctctrain!(loss, trainData, opt, p)
#=for i=1:1
    trainData = data[190:end]
    println("EPOCH $(i)")
    while length(trainData) > chunkSize
        trainOn = gpu.(trainData[1:chunkSize])
        ctctrain!(loss, trainOn, opt, p)
        trainData = trainData[chunkSize+1:end]
        trainOn = nothing
        gc()
    end
    if length(trainData) > 0
        ctctrain!(loss, gpu.(trainData), opt, p)
    end
    BSON.@save "net_epoch$(i).bson" net
    print("Validating\r")
    println("Validation Phoneme Error Rate. $(evaluatePER(net, gpu.(valData)))")
end=#
# @epochs 1 ctctrain!(loss, data, opt, p); print("Validating\r"); println("Validation Phoneme Error Rate. $(evaluatePER(valData))")
# for (x, y) in data
#     losses = loss(x, y)
#     len = length(losses)
#     for (i, l) in enumerate(losses)
#         print("$(i)/$(len)\r")
#         back!(l)
#         opt()
#     end
#     println("loss $(mean(losses))")
# end
# change out of training mode so dropout isn't used during evaluation
# testmode!(model, false)
end

main()
