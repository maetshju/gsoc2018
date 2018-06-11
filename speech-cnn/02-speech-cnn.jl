using Flux
using Flux: relu, crossentropy, logitcrossentropy, back!
using NNlib: σ_stable
using CuArrays
using JLD

include("ctc.jl")

const TRAINDIR = "train"
const TESTDIR = "test"
const EPS = 1e-7

function F(A, blank)
    seq = [A[1]]
    for a in A[2:end]
        if seq[end] != a && a != blank
            push!(seq, a)
        end
    end
    if seq == [blank]
        seq = []
    end
    return seq
end

"""
    lev(s, t)

Levenshtein distance for any iterable, not just strings. Implemented from the
pseudocode on the Wikipedia [page for Levenshtein distance]
(https://en.wikipedia.org/wiki/Levenshtein_distance).

# Parameters
* **s** The first iterable in the comparison; can be a string, Array, tuple,
    etc., so long as it can be indexed
* **t** The second iterable in the comparison; can be a string, Array, tuple,
    etc., so long as it can be indexed

# Returns
* The calculated Levenshtein distance between `s` and `t`
"""
function lev(s, t)
    m = length(s)
    n = length(t)
    d = Array{Int}(zeros(m+1, n+1))

    for i=2:(m+1)
        @inbounds d[i, 1] = i-1
    end

    for j=2:(n+1)
        @inbounds d[1, j] = j-1
    end

    for j=2:(n+1)
        for i=2:(m+1)
            @inbounds if s[i-1] == t[j-1]
                substitutionCost = 0
            else
                substitutionCost = 1
            end
            @inbounds d[i, j] = min(d[i-1, j] + 1, # Deletion
                            d[i, j-1] + 1, # Insertion
                            d[i-1, j-1] + substitutionCost) # Substitution
        end
    end

    @inbounds return d[m+1, n+1]
end

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
    for i in 1:size(afterConv)[1]
        push!(ŷ, denseSection(afterConv[i,:]))
    end
    #ŷ  = [yI .+ EPS for yI in ŷ ]
    ŷ  = [softmax(yI - maximum(yI)) for yI in ŷ ]
    #ŷ  = softmax.(ŷ )
    return ŷ
end

"""
    readData(dataDir)

Reads in each jld file contained in `dataDir` and normalizes the data.
"""
function readData(dataDir)
    fnames = [x for x in readdir(dataDir) if endswith(x, "jld")]

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
    l = ctc(m, y; gpu=true, eps=true)
    # println(l)
    println("mean loss: $(l/length(y))")
    l
end

"""
    evaluateFrameAccuracy(data)

Calculates percent of correct classifications for each input/output in `data`.
"""
function evaluateFrameAccuracy(data)
    correct = Vector()
    for (x, y) in data
        y = indmax.(y)
        ŷ = indmax.(model(x))
        correct = vcat(correct,
                     [ŷ_n == y_n for (ŷ_n, y_n) in zip(ŷ, y)])
    end
    sum(correct) / length(correct)
end

"""
    evaluatePER(data)

Evaluates performance by calculating phoneme error rate on `data`
"""
function evaluatePER(data)
    edits = 0
    len = 0
    for (x, y) in data
        y = F(indmax.(y), 62)
        ŷ = F(indmax.(model(x)), 62)
        e = lev(y, ŷ)
	# println("y $(y)")
	# println("ŷ  $(ŷ )")
        edits += e
        len += length(y)
    end

    return edits / len
end

function main()
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
#=for (x, y) in data
    losses = loss(x, y)
    len = length(losses)
    for (i, l) in enumerate(losses)
        print("$(i)/$(len)\r")
        back!(l)
        opt()
    end
    println("loss $(mean(losses))")
end=#
print("Validating\r")
println("Validation acc. $(evaluatePER(valData))")
end

main()
