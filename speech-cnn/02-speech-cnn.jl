using Flux
using Flux: relu, crossentropy, logitcrossentropy
using Flux.Tracker: back!
using Flux.Optimise: runall, @interrupts
using NNlib: σ_stable, logsoftmax
using CuArrays
using JLD
using Juno

include("warp-ctc.jl")
include("utils.jl")

const TRAINDIR = "train"
const TESTDIR = "test"
const EPS = 1e-7
# const BATCHSIZE = 20

# Convolutional section as defined in the paper
println("Building conv section")

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
                     Dense(1024, 62, identity),
                     #softmax) |> gpu
                     ) |> gpu
                     


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
    #println("afterconv $(afterConv)")
    dims = size(afterConv)
    afterConv = reshape(afterConv, (dims[1], prod(dims[2:end])))
    ŷ = Vector()
    # ŷ  = gpu(zeros(size(afterConv, 1), 62))
    ŷ  = softmax(denseSection(afterConv[1,:]))
    for i in 2:size(afterConv, 1)
        # push!(ŷ, softmax(denseSection(afterConv[i,:])))
	# ŷ[i,:] = softmax(denseSection(afterConv[i,:]))
	ŷ  = hcat(ŷ ,softmax(denseSection(afterConv[i,:])))
    end
    #ŷ  = [yI .+ EPS for yI in ŷ ]
    #ŷ  = [softmax(yI - maximum(yI)) for yI in ŷ ]
    #ŷ  = softmax.(ŷ )
    #ŷ  = collect(Iterators.flatten(ŷ ))
    #ŷ  = gpu(hcat(ŷ))
    ŷ  = gpu(ŷ )
    return ŷ 
    #return gpu(collect(Iterators.flatten(ŷ)))
    #return ŷ 
end

"""
    loss(x, y)

Caclulates the connectionist temporal classification loss for `x` and `y`.
"""
function loss(x, y)
#     println("calculating loss")
#     ms = model.(x)
    ms = model(x)
    # ls = ctc.(ms, y; gpu=true, eps=true)
#     ls = ctc.(ms, y)
    ls, gs = ctc(ms', y)
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
    return ls, gs
end

function ctctrain!(loss, data, opt, parameters; cb = () -> ())
    cb = runall(cb)
    opt = runall(opt)
    losses = Vector()
    @progress for d in data
        ls, gs = loss(d...)
#         println(gs)
        push!(losses, mean(ls))
        println("mean loss over time: $(mean(losses))")
        for (p, g) in zip(parameters[end], gs)
            @interrupts back!(p, g)
        end
        opt()
        cb() == :stop && break
    end
    println("mean epoch loss: $(mean(losses))")
end

function main()
println("Gathering data")
Xs, Ys = readData(TRAINDIR)
data = collect(zip(Xs, Ys))
#valData = data[1:189]
valData = data[1:10] # this when each item is a batch of 20
#data = data[190:end]
data = data[21:end] # batch size = 20
p = params((convSection, denseSection))
opt = ADAM(p)
println()
println("Training")
println("EPOCH 1")
# Flux.train!(loss, data, opt)
ctctrain!(loss, data, opt, p)
println("EPOCH 2")
ctctrain!(loss, data, opt, p)
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
print("Validating\r")
println("Validation acc. $(evaluatePER(valData))")
end

main()
