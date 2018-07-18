using Flux
using Flux: relu, crossentropy, logitcrossentropy, @epochs, flip, reset!
using Flux.Tracker: back!
using Flux.Optimise: runall, @interrupts, SGD
# using NNlib: σ_stable, logsoftmax
using CuArrays
using JLD, BSON
using Juno

include("utils.jl")
include("warp-ctc.jl")

const TRAINDIR = "train"
const TESTDIR = "test"
const EPS = 1e-7
const BATCHSIZE = 20
const EPOCHS = 1

f1 = LSTM(123, 250) |> gpu
b1 = LSTM(123, 250) |> gpu

f2 = LSTM(500, 250) |> gpu
b2 = LSTM(500, 250) |> gpu

f3 = LSTM(500, 250) |> gpu
b3 = LSTM(500, 250) |> gpu

f4 = LSTM(500, 250) |> gpu
b4 = LSTM(500, 250) |> gpu

f5 = LSTM(500, 250) |> gpu
b5 = LSTM(500, 250) |> gpu

output = Dense(500, 62) |> gpu

blstm1(x) = relu.(vcat.(f1.(x), flip(b1, x)))
blstm2(x) = relu.(vcat.(f2.(x), flip(b2, x)))
blstm3(x) = relu.(vcat.(f3.(x), flip(b3, x)))
blstm4(x) = relu.(vcat.(f4.(x), flip(b4, x)))
blstm5(x) = relu.(vcat.(f5.(x), flip(b5, x)))

lstm_section = Chain(blstm1,
                        blstm2,
                        blstm3,
                        blstm4,
                        blstm5
                        ) |> gpu
            
function net(x)
    postLstm = lstm_section(x)
    return logsoftmax.(output.(postLstm))
end

"""
    loss(x, y)

Caclulates the connectionist temporal classification loss for `x` and `y`.
"""
function loss(x, y)
    x = reshape(x, size(x, 1), prod(size(x)[2:end]))
    newx = [x[1,:]]
    for i=2:size(x,1)
        append!(newx, [x[i,:]])
    end
#     x = [x[i,:] for i in 1:size(x, 1)]
    x = gpu.(newx)
    ms = net(x)
    ms = hcat(cpu.(ms)...)
#     ms = test(x[1])
    ls, gs = ctc(cpu(Flux.Tracker.data(ms)), y)
    Flux.reset!((f1, f2, f3, f4, f5, b1, b2, b3, b4, b5))
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
trainData = data[190:end]
println(size(trainData[1][1]))
# data = data[21:end] # batch size = 20
p = params((f1, f2, f3, f4, f5, b1, b2, b3, b4, b5, output))
opt = ADAM(p, 10.0^-4)
println()
println("Training")
for i=1:EPOCHS
    println("EPOCH $(i)")
    ctctrain!(loss, trainData, opt)
    BSON.@save "rnn_epoch$(i).bson" net
    print("Validating\r")
    println("Validation Phoneme Error Rate. $(evaluatePER(net, gpu.(valData)))")
end
end

main()
