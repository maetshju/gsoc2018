# a port of the GPU kernels from Baidu's C++ warp-ctc package
# GitHub: https://github.com/baidu-research/warp-ctc/
# paper: https://arxiv.org/pdf/1512.02595.pdf

using CUDAnative, CUDAdrv, Flux

function log_plus_f(p1, p2)
    
    isinf(p1) && return p2
    isinf(p2) && return p1
#     if isinf(p1)
#         return p2
#     elseif isinf(p2)
#         return p1
#     end
    s = p1 + CUDAnative.log(1+CUDAnative.exp(p2 - p1))
    # With two very small numbers such as, such as -900 and -800, this will return
    # Inf32 because -800 - -900 = 100, and exp(100) is Inf with Float32 values.
    # Rationally, this should be -Inf32, because the calculation is ln(exp(-900) + exp(-800)),
    # which while not truly 0, is effectively 0 and is too small to be represented with 32 bit
    # floating point numbers
    s == Inf32 && return -Inf32
    return s
end


function logadd(a, b)

    if isinf(a)
        return b
    elseif isinf(b)
        return a
    end
    
    if isnan(a) || isnan(b)
        error("NAN HAPPENED IN LOGADD")
    end
    return a + log(1+exp(b-a))
end

function logsum(a)
    s = a[1]
    for item in a[2:end]
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
    S = 2*L + 1
    
    if L + repeats > T
        return nothing
    end
    
    labels = labelsWithBlanks
    
    start = (L + repeats < T) ? 0 : 1
    last = S > 1 ? 2 : 1
    
#     i = tid
#     while i <= last - start
#         alpha[start + i] = CUDAnative.log(probs[labels[start + i]])
#         i += blockDim().x
#     end

    if tid == 1
        alpha[1] = CUDAnative.log(probs[labels[1]])
        alpha[2] = CUDAnative.log(probs[labels[2]])
    end
    
    sync_threads()
    
    for t=2:T
        startCurRow = (t-1) * S
        startPrevRow = (t-2) * S
        startProbCol = (t-1) * div(length(probs), T)
        
        ## First column stuff, if we need it, goes here
        
        idx = tid + 1
        while idx <= S
        
            
            prevSum = log_plus_f(alpha[startPrevRow + idx], alpha[startPrevRow + idx-1])
            
            if labels[idx] != blankLabel && idx != 2 && labels[idx] != labels[idx-2]
                prevSum = log_plus_f(prevSum, alpha[startPrevRow + idx-2])
            end
            
            if idx < S - 2*(T-t) - 1
                alpha[idx + startCurRow] = -Inf32
            else
                alpha[startCurRow + idx] = prevSum + CUDAnative.log(probs[startProbCol + labels[idx]])
            end
        
            idx += blockDim().x
        end
        
        sync_threads()
    end
    return nothing
end

# function computeAlphaKernel(probs, labelSize, uttLength, repeatsInLabel,
#                             labelsWithoutBlanks, labelsWithBlanks, alpha, nllForward, blankLabel)
# 
# 
#     tid = threadIdx().x
#     L = labelSize
#     T = uttLength
#     S = 2*L + 1
# # #     const prob_offset = out_dim
#     repeats = repeatsInLabel
# # 
# #     # TODO: What in the world is this?
# # #     const NV = NT * VT
# # 
#     if L + repeats > T
#         return nothing
#     end
# # 
# #     
# 
#     ## TODO: May need to add the final if(tid==0) portion
# 
#     labels = labelsWithBlanks
#     
#     ## Shouldn't be necessary because alpha was already initialized with -Inf32
# #     for idx=tid:blockDim().x:S
# #         alpha[idx] = ninf
# #     end
# # 
# #     ## TODO: load labels into shared memory
# # 
#     sync_threads()
#     start = (L + repeats <= T) ? 0 : 1
#     last = S > 1 ? 2 : 1
# 
#     i = tid
#     while i <= (last - start)
#         alpha[i + start] = CUDAnative.log(probs[labels[i+start]])
#         i += blockDim().x
#     end
#     
#     sync_threads()
# 
#     for t=2:T
#         # need to subtract from `t` becuase Julia indexes from 1
#         startCurrRow = (t-1) * S
#         startPrevRow = (t-2) * S
#         startProbCol = (t-1) * div(length(probs), T)
# 
#         if tid == 1
#             if start == 0
#                 alpha[startCurrRow+1] = alpha[startPrevRow+1] +
#                                       CUDAnative.log(probs[startProbCol+1 + blankLabel])
#             elseif start == 1
#                 alpha[startCurrRow+1] = alpha[startPrevRow+1]
#             end
#         end
# 
#         sync_threads()
# 
#         idx = tid
#         while idx <= S
# #         for idx=(tid+1):blockDim().x:(S-1)
#             prevSum = log_plus_f(alpha[idx + startPrevRow], alpha[(idx-1) + startPrevRow])
#             
#             if idx > 2 && labels[idx] != blankLabel && labels[idx] != labels[idx-2]
# 
#                 prevSum = log_plus_f(prevSum, alpha[(idx-2) + startPrevRow])
#                 
#             end
#             if idx < S - 2*(T-t) - 1
#                 alpha[idx + startCurrRow] = -Inf32
#             else
#                 alpha[idx + startCurrRow] = prevSum + CUDAnative.log(probs[startProbCol + labels[idx]])
# #                 alpha[idx + startCurrRow] = labels[idx]
# #                 alpha[idx + startCurrRow] = S - 2*(T-t) - 1
#             end   
#             idx += blockDim().x
#         end
# 
#         sync_threads()
# 
#         if tid == 0
#             loglike = -Inf32
#             val = 2*(L-1) + 1 - (L + repeats == T ? 1 : 0)
# 
#             start = val * (L != 0) + start
#             last = val * (L != 0) + last
#  
#              for i=start:last
#                 loglike = log_plus_f(loglike, alpha[i + (T-1) * S])
#              end
#              nllForward[blockIdx().x] = -loglike
#         end
#         
#         sync_threads()
#         
#     end
# 
#     return nothing
# end

function computeBetasAndGradKernel(probs, labelSize, uttLength,
                                    repeatsInLabel, labelsWithBlanks,
                                    alphas, beta, output, accum, nllForward, nllBackward,
                                    grad, blankLabel)
    
    tid = threadIdx().x
    L = labelSize
    T = uttLength
    S = 2*L + 1
    repeats = repeatsInLabel
    logPartition = -nllForward[blockIdx().x]
    
    labels = labelsWithBlanks
    alpha = alphas[blockIdx().x]
#     output = similar(alphas)
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
#         s = - s
#         grad[startProbRow + idx] = probs[startProbRow + idx] - s * CUDAnative.exp(accum[startProbRow + idx])
        grad[startProbRow + idx] = probs[startProbRow + idx] - CUDAnative.exp(accum[startProbRow + idx] - s)
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

        # TODO: should this be t < T?
        if t < T
            
            idx = tid
            i = 0
            while idx <= S
                
                nextSum = log_plus_f(beta[startNextRow + idx], beta[startNextRow + idx+1])
                
                if labels[idx] != blankLabel && idx != S-1 && labels[idx] != labels[idx+2]
                    nextSum = log_plus_f(nextSum, beta[startNextRow + idx + 2])
                end
                
#                 beta[i] = nextSum + CUDAnative.log(probs[startProbCol + labels[idx]])
                if idx > 2*t
                    beta[idx + startCurRow] = -Inf32
                else
                    beta[idx + startCurRow] = nextSum + CUDAnative.log(probs[startProbCol + labels[idx]])
                end
#                 beta[idx + startCurRow] = t
                
                idx += NT
                i += 1
            end
        
            sync_threads()
#             
            if tid == 0 && last == S
                beta[S-1] = beta[S-1] + CUDAnative.log(probs[startProbCol + blankLabel])
            end
            
            idx = tid
            while idx <= S
                output[startCurRow + idx] = alphas[idx+startCurRow] + beta[startCurRow + idx]
                idx += blockDim().x
            end
            
            sync_threads()
        end
        
#         idx = tid
#         while idx <= div(length(accum), T)
#             startAccRow = (t-1) * div(length(accum), T)
#             accum[startAccRow + idx] = -Inf32
#             idx += blockDim().x
#         end
        
        sync_threads()
#         
        if tid == 1
#         while idx <= T
        
            startAccRow = (t-1) * div(length(accum), T)
            startOutputRow = (t-1) * S
            
            for i=1:S
                labelIdx = labels[i]
                accum[startAccRow + labelIdx] = log_plus_f(accum[startAccRow + labelIdx], output[startOutputRow + i])
            end
            
#             labelIdx = labels[idx]
            
#             accum[startAccRow + labelIdx] = idx
#             log_plus_f(accum[startAccRow + labelIdx], output[startOutputRow + idx])
            
#             idx += blockIdx().x
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
#         s = - s
#         grad[startProbRow + idx] = probs[startProbRow + idx] - s * CUDAnative.exp(accum[startProbRow + idx])
        grad[startProbRow + idx] = probs[startProbRow + idx] - CUDAnative.exp(accum[startProbRow + idx] - s)
        
            idx += blockDim().x
        end
        
        sync_threads()
        
#         if t == 1 && tid == 1
#             loglike = -Inf32
#             val = 2 * (L-1) + 1 - (L + repeats == T ? 1 : 0)
#             
#             start = -val * (L != 0) + start
#             last = -val * (L != 0) + last
# #         
#             for i=start:last
#                 loglike = log_plus_f(loglike, beta[i])
#             end
#         
#             nllBackward[blockIdx().x] = -loglike
#         end
        
        t -= 1
        sync_threads()
    end

    return nothing
end

function ctc(ŷ, y)

    blank = size(ŷ, 2)
#     println(blank)
#     println(size(ŷ))
#     blank = 62
#     blank = 4
    labels = indmax.([y[i,:] for i=1:size(y,1)])
#     labels = argmax.([y[i,:] for i=1:size(y,1)])
    z = F(labels, blank)
    z′ = [blank]
    for label in z
        push!(z′, label)
        push!(z′, blank)
    end
    T = size(ŷ, 1)
    U′ = 2*length(z) + 1
#     alphas = Flux.TrackedArray([-Inf for x in 1:(size(ŷ,1) * U′)])
    alphas = CUDAdrv.CuArray([-Inf32 for x in 1:(size(ŷ,1) * U′)])
    betas = CUDAdrv.CuArray([-Inf32 for x in 1:(size(ŷ,1) * U′)])
#     println()
#     println(ŷ )
#     println(size(alphas))
#     println(typeof(alphas))
    #ŷ = Flux.Tracker.data(ŷ )
    ŷ  = CUDAdrv.CuArray(cpu(Flux.Tracker.data(ŷ )))
    nRepeats = countRepeats(labels)
#     println("labels: $(z′)")
    
    
    alphalikelihoods = CUDAdrv.CuArray{Float32}(size(ŷ,1))
    betalikelihoods = similar(alphalikelihoods)

#     println("beginning alphas computation")
    @cuda (U′, U′) computeAlphaKernel(ŷ, length(z), size(ŷ,1), nRepeats, CUDAdrv.CuArray(z), CUDAdrv.CuArray(z′), alphas, blank)
#     @cuda threads=U′ computeAlphaKernel(ŷ, length(z), size(ŷ,1), nRepeats, CUDAdrv.CuArray(z), CUDAdrv.CuArray(z′), alphas, alphalikelihoods, blank)
#     
# #     @cuda threads=U′ alpha2kernel(ŷ, alphas, T, U′, alphalikelihoods, CUDAdrv.CuArray(z′), blank)
# 
#     println("alphas done")
    grads = CUDAdrv.CuArray([-Inf32 for x in 1:length(ŷ)])
# 
#     println("extracting alphas")
#     println(Array(alphas))
#     display(reshape(Array(alphas), 7, 4)')
#     println()
#     println("alphas extracted")
    
#     println(typeof(betas))
#     println("beginning betas computation")
    output = CUDAdrv.CuArray([-Inf32 for x in 1:(size(ŷ,1) * U′)])
    accum = CUDAdrv.CuArray([-Inf32 for x in 1:length(ŷ)])
    @cuda (U′, U′) computeBetasAndGradKernel(ŷ, length(z), size(ŷ,1), nRepeats, CUDAdrv.CuArray(z′), alphas, betas, output, accum, alphalikelihoods, betalikelihoods, grads, blank)
#     @cuda threads=U′ computeBetasAndGradKernel(ŷ, length(z), size(ŷ,1), nRepeats, CUDAdrv.CuArray(z′), alphas, betas, output, accum, alphalikelihoods, betalikelihoods, grads, blank)
#     println("betas computed")

    # TODO: these produce an illegal memory access
#     println(size(betas))
#     display(Array(reshape(Array(betas), 7, 4)'))
#     println()
#     display(Array(reshape(Array(output), 7, 4)'))
#     println()
#     println(Array(grads))
#     display(Array(reshape(Array(grads), 4, 4)'))
#     println()
    
    ls = Array(reshape(Array(output), U′, T)')
#     display(ls)
#     println()
#     println(typeof(ls))
#     println(ls[1,:])
#     println(logsum(ls[1,:]))
#     l = logsum.([ls[x,:] for x in 1:size(ls,1)])
#     println(any(isinf, mapslices(logsum, ls, 1)))
    ls = mapslices(logsum, ls, 2)
    ls = ls .* -1
    gs = permutedims(reshape(Array(grads), size(ŷ,2), size(ŷ,1)), [2,1])
#     println(any(isinf, Array(ŷ)))
#     println(any(x -> x == -Inf32, exp.(Array(accum))))
#     print("Is inf32? ")
#     println(any(x -> x == Inf32, exp.(Array(accum))))
#     println(any(isinf, gs))
#     println(exp.(Array(accum)[(end-62):end]))
#     println(gs[56,:])
#     println(size(gs))
    gs = vec(mapslices(sum, gs, 1))
#     display(gs)
#     println()
    return vec(ls), gs
end
