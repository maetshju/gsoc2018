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
# 	    x = gpu(x)
        #y = gpu.([y[i,:] for i in 1:size(y,1)])
# 	y = gpu(y)
        push!(Xs, x)
        push!(Ys, y)
    end

#     batchedXs = Xs[1:BATCHSIZE]
#     batchedYs = Ys[1:BATCHSIZE]
# 
#     println("tbx: $(typeof(batchedXs))")
#     println("txs: $(typeof(Xs))")
# 
#     lXs = length(Xs)
# 
#     for i=2:ceil(Int64, length(Xs)/BATCHSIZE)
#         startI = (i-1) * BATCHSIZE + 1
#         lastI = min(lXs, i*BATCHSIZE)
# 
#         push!(batchedXs, Xs[startI:lastI])
#         push!(batchedYs, Ys[startI:lastI])
#     end

    # Xs = [Xs[((i-1)*BATCHSIZE+1):min(length(Xs),i*BATCHSIZE)] for i in 1:ceil(Int64, length(Xs)/BATCHSIZE)]
    # Ys = [Ys[((i-1)*BATCHSIZE+1):min(length(Ys),i*BATCHSIZE)] for i in 1:ceil(Int64, length(Ys)/BATCHSIZE)]
    return (Xs, Ys)
#     return (batchedXs, batchedYs)
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
        y = F(indmax.([y[i,:] for i=1:size(y,1)]), 62)
        ŷ  = model(x)
        ŷ  = indmax.([ŷ[:,i] for i=1:size(ŷ,2)])
        println(y)
        println(ŷ )
        e = lev(y, ŷ)
	# println("y $(y)")
	# println("ŷ  $(ŷ )")
        edits += e
        len += max(length(y), length(ŷ ))
    end

    return edits / len
end
