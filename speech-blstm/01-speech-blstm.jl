using Flux
using Flux: crossentropy, flip
using BSON
using CuArrays

const TRAIN_DIR = "train"
const BATCH_SIZE = 32
const EPOCHS = 5

forward = LSTM(26, 100) |> gpu
backward = LSTM(26, 100) |> gpu
output = Dense(200, 61) |> gpu

blstm(x) = vcat.(forward.(x), flip(backward, x))

# """
#   blstm(x)
#   
# Bidirection LSTM layer on `x`. Truncates gradients from LSTMs every 200 steps.
# """
# function blstm(x)
#   step = 200
#   nSteps = length(x)
#   outputs = Vector()
#   processed = 0
#   for i=1:step:div(nSteps, step)
#     append!(outputs, vcat.(forward.(x[i:i+step-1]), flip(backward, x[i:i+step-1])))
#     processed += 1
#     Flux.truncate!((forward, backward))
#   end
#   remaining = nSteps % step
#   append!(outputs, vcat.(forward.(x[processed*step+1:end]), flip(backward, x[processed*step+1:end])))
#   return outputs
# end

function net(x)
  b = blstm(x)
  softmax.(output.(b))
end

function loss(x, y)
  
  n = net(gpu.(x))
  l = sum(crossentropy.(n, gpu.(y)))
  println(l)
  Flux.reset!((forward, backward))
  return l
end

function readData(dataDir)
  fnames = [x for x in readdir(dataDir) if endswith(x, "bson")]

  Xs = Vector()
  Ys = Vector()

  for (i, fname) in enumerate(fnames)
      print(string(i) * "/" * string(length(fnames)) * "\r")
      BSON.@load joinpath(dataDir, fname) x y
      x .-= mean(x,2)
      x ./= std(x,2)
      x = [x[i, :] for i in 1:size(x, 1)]
      y = [y[i, :] for i in 1:size(y, 1)]
      push!(Xs, x)
      push!(Ys, y)
  end
  maxSteps = maximum([length(x) for x in Xs])
  Xs = map(x-> append!(x, [zeros(26) for i in 1:(maxSteps-length(x))]), Xs)
  Ys = map(y -> append!(y, [append!([1], Bool.(zeros(60))) for i in 1:(maxSteps-length(y))]),  Ys)
  return (Xs, Ys)
end

function framewiseAccuracy(model, data)
  correct = Vector()
  len = length(data)
  for (i, (x, y)) in enumerate(data)
    while x[end] == zeros(26)
      x = x[1:end-1]
      y = y[1:end-1]
    end
    print("$(i)/$(len)\r")
    y = indmax.(y)
    ŷ = indmax.(model(gpu.(x)))
    Flux.reset!((forward, backward))
#     println(y)
#     println(ŷ)
    correct = append!(correct, ŷ .== y)
#     correct = append!(correct, [ŷ_n == y_n for (ŷ_n, y_n) in zip(ŷ, y)])
  end
  println()
  sum(correct) / length(correct)
end

function makeBatches(Xs, Ys)
  collected = shuffle(collect(zip(Xs, Ys)))
  counter = 0
  data = Vector()
  local batchX
  local batchY
  for (i, (x, y)) in enumerate(collected)
    if counter == 0
      batchX = x
      batchY = y
    else
      batchX = hcat.(batchX, x)
      batchY = hcat.(batchY, y)
    end
    counter += 1
    if counter == BATCH_SIZE || i == length(collected)
      while batchX[end] == zeros(26, BATCH_SIZE)
        batchX = batchX[1:end-1]
        batchY = batchY[1:end-1]
      end
      push!(data, (batchX, batchY))
      counter = 0
    end
  end
  return data
end

Xs, Ys = readData(TRAIN_DIR)
valXs = Xs[1:189]
valYs = Ys[1:189]
p = params((forward, backward, output))
# opt = ADAM(p, 10.0^-4)
opt = Momentum(p, 10.0^-5)
valData = collect(zip(valXs, valYs))
println()
for epoch=1:EPOCHS
  println("EPOCH $(epoch)")
  trainData = makeBatches(Xs[190:end], Ys[190:end])
  # data = collect(zip(Xs, Ys))
  # valData = data[1:189]
  # trainData = data[190:end]
  # loss(Xs[1:2], Ys[1:2])
  Flux.train!(loss, trainData, opt)
  trainData = nothing
  gc()
  forward = cpu(forward)
  backward = cpu(backward)
  output = cpu(output)
#   BSON.@save "model_epoch$(epoch).bson" forward backward output
  forward = gpu(forward)
  backward = gpu(backward)
  output = gpu(output)
  println("Validating...")
  println("Validation acc $(framewiseAccuracy(net, valData))")
  gc()
end
