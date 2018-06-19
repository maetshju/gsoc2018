# a port of Baidu's C++ warp-ctc package
# GitHub: https://github.com/baidu-research/warp-ctc/
# paper: https://arxiv.org/pdf/1512.02595.pdf

using Flux

function log_plus_f(p1, p2)
    if isinf(p1)
        return p2
    end

    if isinf(p2)
        return p1
    end

    return p1 + log(1+exp(p2 - p1))
end

function F(A)
    prev = A[1]
    z = [prev]
    for curr in A[2:end]
        if curr != prev && curr != blank
            push!(z, curr)
        end
        prev = curr
    end
    return z
end

function countRepeats(A)
    repeats = 0
    for (i,elem) in enumerate(A)
        if i > 1 && A[i] == A[i-1]
            repeats += 1
        end
    end
    return repeats
end

function computeAlphaKernel(probs, labelSize, uttLength, repeatsInLabel,
                            labelsWithoutBlanks, labelsWithBlanks, alphas,
                            nllForward, blankLabel)


    const tid = threadIdx().x
    const L = labelSize
    const T = uttLength
    const S = 2*L + 1
    const prob_offset = out_dim
    const repeats = repeatsInLabel

    # TODO: What in the world is this?
    const NV = NT * VT

    if L + repeats > T
        return nothing
    end

    # add blanks to labels
    labelsWithBlanks = [blank]
    for label in labelsWithoutBlanks
        push!(labelsWithBlanks, label)
        push!(labelsWithBlanks, blankLabel)
    end

    ## TODO: May need to add the final if(tid==0) portion

    labels = labelsWithBlanks

    for idx=tid:blockdim().x:S
        alpha[idx] = -Inf
    end

    ## TODO: load labels into shared memory

    sync_threads()
    start = (L + repeats < T) ? 0 : 1
    last = S > 1 ? 2 : 1

    for i=tid:blockDim().x:(last-start)
        alpha[i + start] = log(probs(prob_offset + label[i+start]))
    end

    sync_threads()

    for t=2:T
        startCurrRow = t * S
        startPrevRow = (t - 1) * S
        startProbCol = 1

        if tid == 0
            if start == 0
                alpha[startCurrRow] = alpha[startPrevRow] +
                                      log(probs[startProbCol + blankLabel - 1])
            elseif start == 1
                alpha[startCurrRow] = alpha[startPrevRow]
            end
        end

        sync_threads()

        for idx=(tid+1):blockDim().x:(S-1)
            prevSum = log_plus_f(alpha[idx + startPrevRow], alpha[(idx-1) + startPrevRow])

            if label[idx] != blankLabel && idx != 1 && label[idx] != label[idx-2]
                prevSum = log_plus_f(prev_sum, alpha[(idx-2) + startPrevRow])
                alpha[idx + startCurrRow] = prevSum + log(probs[prob_offset + startProbCol + label[idx]])
            end
        end

        sync_threads()

        if tid == 0
            loglike = -Inf
            val = 2*(L-1) + 1 - (L + repeats == T ? 1 : 0)

            start = val * (L != 0) + start
            last = val * (L != 0) + last

            for i=start:last
                loglike = log_plus_f(loglike, alpha[i + (T-1) * S])
            end
        end

        nllForward[blockIdx().x] = -loglike
    end
end

function ctc(ŷ, y)
    labels = indmax.([y[i,:] for i=1:size(y,1)])
    z = F(labels)
    U′ = 2*length(z) + 1
    alphas = gpu([Flux.Tracker.TrackedReal(Float32(-Inf)) for x in 1:(size(ŷ,1) * U′)])

    println("beginning alphas computation")
    @cuda (size(alphas, 1), U′) computeAlphaKernel(ŷ, length(z), U′, countRepeats(labels), z, Array{typeof(z)}(length(z)*2 + 1), alphas,
        Array{typeof(alphas)}(size(ŷ,1)), 62)

    println("alphas done")
end
