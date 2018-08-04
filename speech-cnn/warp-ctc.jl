# a port of the GPU kernels from Baidu's C++ warp-ctc package
# GitHub: https://github.com/baidu-research/warp-ctc/
# paper: https://arxiv.org/pdf/1512.02595.pdf

using CUDAnative, CUDAdrv, Flux
using Flux.Tracker: @grad
const EPS = 1e-7

function log_plus_f(p1, p2)
    
    isinf(p1) && return p2
    isinf(p2) && return p1

    # always want the greater number on the left in the exponentiation;
    # the magnitude difference may end up making the number very positive
    # which will cause exp() to return Inf
    # E.g., a = -900, b = -800, will give exp(-800 - -900), which will be
    # Inf for Float32 values
    if p1 < p2
        p1, p2 = p2, p1
    end

    return p1 + CUDAnative.log(1+CUDAnative.exp(p2 - p1))
end


function logadd(a, b)

    isinf(a) && return b
    isinf(b) && return a
    
    if a < b
        a, b = b, a
    end
    
    return a + log(1+exp(b-a))
end

function logsum(a)
    s = -Inf32
    for item in a
        s = logadd(s, item)
    end
    return s
end

function F(A, blank)
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

function computeAlphaKernel(probs, labelSize, uttLength, repeats, labelsWithoutBlanks, labelsWithBlanks, alpha, blankLabel)

    tid = threadIdx().x
    L = labelSize
    T = uttLength
    S = length(labelsWithBlanks)
    
    if L + repeats > T
        return nothing
    end
    
    labels = labelsWithBlanks
    
    start = (L + repeats <= T) ? 0 : 1
    last = S > 1 ? 2 : 1
    
    i = tid
    while i <= last - start
        alpha[start + i] = probs[labels[start + i]]
        i += blockDim().x
    end
    
    sync_threads()
    
    for t=2:T
        startCurRow = (t-1) * S
        startPrevRow = (t-2) * S
        startProbCol = (t-1) * div(length(probs), T)
        
        if tid == 1 && !(1 < S - 2*(T-t) - 1)
            if start == 0
                alpha[startCurRow + 1] = probs[startProbCol + blankLabel] + alpha[startPrevRow + 1]
            elseif start == 1
                alpha[startCurRow + 1] = alpha[startPrevRow + 1]
            end
        end
        
        sync_threads()
        
        idx = tid+1
        while idx <= S
        
            
            prevSum = log_plus_f(alpha[startPrevRow + idx], alpha[startPrevRow + idx-1])
            
            if labels[idx] != blankLabel && idx != 2 && labels[idx] != labels[idx-2]
                prevSum = log_plus_f(prevSum, alpha[startPrevRow + idx-2])
            end
            
            if idx < S - 2*(T-t) - 1
                alpha[idx + startCurRow] = -Inf32
            else
                alpha[startCurRow + idx] = prevSum + probs[startProbCol + labels[idx]]
            end
        
            idx += blockDim().x
        end
        
        sync_threads()
    end
    return nothing
end

function computeBetasAndGradKernel(probs, labelSize, uttLength,
                                    repeatsInLabel, labelsWithBlanks,
                                    alphas, beta, output, accum,
                                    grad, blankLabel)
    
    tid = threadIdx().x
    L = labelSize
    T = uttLength
    S = 2*L + 1
    repeats = repeatsInLabel
    
    labels = labelsWithBlanks
    alpha = alphas[blockIdx().x]
    NT = blockDim().x
    
    # From Modern GPU documentation
    # NT = number of threads per thread block
    # VT = values per thread (registers?)
    # NV = NT * VT (number of values per thread block)
    
    if (L+repeats) > T
        return nothing
    end
    
    start = S > 1 ? S-2 : 0
    last = L + repeats < T ? S : S-1
    
    sync_threads()
    
    
    startCurRow = (T-1)*S
    startProbCol = (T-1) * div(length(probs), T)
    
    i = tid
    # add one to offset index to match Julia starting to index from 1
    while i <= last - start + 1

        # commented out section is how warp-ctc does it, which is different than
        # how Graves specifies it in his work
    #         beta[startCurRow + i + start] =
#             CUDAnative.log(probs[startProbCol + labels[i+start]])
        beta[startCurRow + i + start] = 0
        output[startCurRow + i + start] = beta[startCurRow + i + start] + alphas[startCurRow + i + start]
        i += blockDim().x
    end
    
    sync_threads()
    
    if tid == 1
        startAccRow = startProbCol
        startOutputRow = startCurRow
        
        for i=1:S
            labelIdx = labels[i]
            accum[startAccRow + labelIdx] = log_plus_f(accum[startAccRow + labelIdx], output[startOutputRow + i])
        end
    end
    
    sync_threads()
    
    idx = tid
    while idx <= div(length(grad), T)
#         
        startProbRow = (T - 1) * div(length(probs), T)
        startOutputRow = (T - 1) * S
        
        s = -Inf32
        for i=1:S
            s = log_plus_f(s, output[startOutputRow + i])
        end
        
        # ∂L/∂a (where a is activation before softmax)
#         grad[startProbRow + idx] = probs[startProbRow + idx] - CUDAnative.exp(accum[startProbRow + idx] - s)
        # ∂L/∂a (where a is activation before logsoftmax)
        grad[startProbRow + idx] = CUDAnative.exp(probs[startProbRow + idx]) - CUDAnative.exp(accum[startProbRow + idx] - s)
        # ∂L/∂y (where y is network output with regular softmax)
#         grad[startProbRow + idx] = -1 * CUDAnative.exp(accum[startProbRow + idx] - (s + CUDAnative.log(probs[startProbRow + idx])))
        # ∂L/∂y (where y is network output with logsoftmax)
#         grad[startProbRow + idx] = -1 * CUDAnative.exp(accum[startProbRow + idx] - (s + probs[startProbRow + idx]))
        idx += blockDim().x
    end
    
    sync_threads()
#     
    t = T-1
    while t >= 1
#     for t=(T-1):(-1):1
        startCurRow = (t-1)*S
        startNextRow = t*S
        startProbCol = t * div(length(probs), T)

        if t < T
            
            idx = tid
            while idx <= S-1
                
                nextSum = log_plus_f(beta[startNextRow + idx] + probs[startProbCol + labels[idx]], beta[startNextRow + idx+1] + probs[startProbCol + labels[idx+1]])
                
                if labels[idx] != blankLabel && idx != S-1 && labels[idx] != labels[idx+2]
                    nextSum = log_plus_f(nextSum, beta[startNextRow + idx + 2] + probs[startProbCol + labels[idx+2]])
                end
                
                if idx > 2*t
                    beta[idx + startCurRow] = -Inf32
                else
#                     beta[idx + startCurRow] = nextSum + probs[startProbCol + labels[idx]]
                    beta[idx + startCurRow] = nextSum
                        
                end
#                 beta[idx + startCurRow] = t
                
                idx += NT
            end
        
            sync_threads()
            
            if tid == 1 && last == S
                beta[startCurRow + S] = beta[startNextRow + S] + probs[startProbCol + blankLabel]
            end
            
            sync_threads()
            
            idx = tid
            while idx <= S
                output[startCurRow + idx] = alphas[idx+startCurRow] + beta[startCurRow + idx]
                idx += blockDim().x
            end
            
            sync_threads()
        end
        
        
        sync_threads()
#         
        if tid == 1
        
            startAccRow = (t-1) * div(length(accum), T)
            startOutputRow = (t-1) * S
            
            for i=1:S
                labelIdx = labels[i]
                accum[startAccRow + labelIdx] = log_plus_f(accum[startAccRow + labelIdx], output[startOutputRow + i])
            end
        end
        
        sync_threads()
        
        idx = tid
        while idx <= div(length(grad), T)
#         
            startProbRow = (t - 1) * div(length(probs), T)
            startOutputRow = (t - 1) * S
            
            s = -Inf32
            for i=1:S
                s = log_plus_f(s, output[startOutputRow + i])
            end
        # ∂L/∂a (where a is activation before softmax)
#         grad[startProbRow + idx] = probs[startProbRow + idx] - CUDAnative.exp(accum[startProbRow + idx] - s)
        # ∂L/∂a (where a is activation before logsoftmax)
        grad[startProbRow + idx] = CUDAnative.exp(probs[startProbRow + idx]) - CUDAnative.exp(accum[startProbRow + idx] - s)
        # ∂L/∂y (where y is network output with regular softmax)
#         grad[startProbRow + idx] = -1 * CUDAnative.exp(accum[startProbRow + idx] - (s + CUDAnative.log(probs[startProbRow + idx])))
        # ∂L/∂y (where y is network output with logsoftmax)
#             grad[startProbRow + idx] = -1 * CUDAnative.exp(accum[startProbRow + idx] - (s + probs[startProbRow + idx]))
            idx += blockDim().x
        end
        
        sync_threads()
        
        t -= 1
        sync_threads()
        # because of course, it wouldn't work without this earlier return statement
        # otherwise, some of the gradient values become 0
        t == 0 && return
    end

    return nothing
end

function ctc(ŷ::CuArrays.CuArray, y; activation=logsoftmax)

    ŷ = activation(ŷ)
    
    blank = Int32(size(ŷ, 1))
    labels = indmax.([y[i,:] for i=1:size(y,1)])
    z = F(labels, blank)
    z′ = [blank]
    for label in z
        push!(z′, label)
        push!(z′, blank)
    end
    T = size(ŷ, 2)
    U′ = 2*length(z) + 1
    alphas = gpu([-Inf32 for x in 1:(size(ŷ,2) * U′)])
    betas = gpu([-Inf32 for x in 1:(size(ŷ,2) * U′)])
    
    nRepeats = countRepeats(labels)

    @cuda (1, U′) computeAlphaKernel(ŷ, length(z), size(ŷ,2), nRepeats, gpu(z), CUDAdrv.CuArray(z′), alphas, blank)
    # Julia 0.7 and updated CUDAnative function call
#     @cuda threads=U′ computeAlphaKernel(ŷ, length(z), size(ŷ,2), nRepeats, gpu(z), CUDAdrv.CuArray(z′), alphas, blank)
    grads = gpu([-Inf32 for x in 1:length(ŷ)])
    output = gpu([-Inf32 for x in 1:(size(ŷ,2) * U′)])
    accum = gpu([-Inf32 for x in 1:length(ŷ)])
    
    @cuda (1, U′) computeBetasAndGradKernel(ŷ, length(z), size(ŷ,2), nRepeats, CUDAdrv.CuArray(z′), alphas, betas, output, accum, grads, blank)
    # Julia 0.7 and updated CUDAnative function call
#     @cuda threads=U′ computeBetasAndGradKernel(ŷ, length(z), size(ŷ,2), nRepeats, CUDAdrv.CuArray(z′), alphas, betas, output, accum, grads, blank)
    
    ls = Array(reshape(Array(output), U′, T)')
    ls = -1 .* mapslices(logsum, ls, 2)
    gs = reshape(Array(grads), size(ŷ,1), size(ŷ,2))
    println(alphas)
    println(betas)
    
    ŷ = alphas = betas = output = accum = grads = nothing
    
    return mean(ls), gs
end

ctc(ŷ::TrackedArray, y::AbstractArray) = Flux.Tracker.track(ctc, ŷ, y)

@grad function ctc(ŷ, y)
    ls, gs = ctc(Flux.Tracker.data(ŷ), Flux.Tracker.data(y))
    return ls, Δ -> (Δ .* gpu(gs), Δ)
end
