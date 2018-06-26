# a port of the GPU kernels from Baidu's C++ warp-ctc package
# GitHub: https://github.com/baidu-research/warp-ctc/
# paper: https://arxiv.org/pdf/1512.02595.pdf

using CUDAnative, CUDAdrv

function log_plus_f(p1, p2)
    if isinf(p1)
        return p2
    end

    if isinf(p2)
        return p1
    end

    return p1 + CUDAnative.log(1+exp(p2 - p1))
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

function computeAlphaKernel(probs, labelSize, uttLength, repeatsInLabel,
                            labelsWithoutBlanks, labelsWithBlanks, alpha, nllForward, blankLabel)


    tid = threadIdx().x
    L = labelSize
    T = uttLength
    S = 2*L + 1
# #     const prob_offset = out_dim
    repeats = repeatsInLabel
# 
#     # TODO: What in the world is this?
# #     const NV = NT * VT
# 
    if L + repeats > T
        return nothing
    end
# 
#     

    ## TODO: May need to add the final if(tid==0) portion

    labels = labelsWithBlanks
    
    ## Shouldn't be necessary because alpha was already initialized with -Inf32
#     for idx=tid:blockDim().x:S
#         alpha[idx] = ninf
#     end
# 
#     ## TODO: load labels into shared memory
# 
    sync_threads()
    start = (L + repeats <= T) ? 0 : 1
    last = S > 1 ? 2 : 1

    i = tid
    while i <= (last - start)
        alpha[i + start] = CUDAnative.log(probs[labels[i+start]])
        i += blockDim().x
    end
    
    sync_threads()

    for t=2:T
        # need to subtract from `t` becuase Julia indexes from 1
        startCurrRow = (t-1) * S
        startPrevRow = (t-2) * S
        startProbCol = (t-1) * div(length(probs), T)

        if tid == 1
            if start == 0
                alpha[startCurrRow+1] = alpha[startPrevRow+1] +
                                      CUDAnative.log(probs[startProbCol+1 + blankLabel])
            elseif start == 1
                alpha[startCurrRow+1] = alpha[startPrevRow+1]
            end
        end

        sync_threads()

        idx = tid
        while idx <= S
#         for idx=(tid+1):blockDim().x:(S-1)
            prevSum = log_plus_f(alpha[idx + startPrevRow], alpha[(idx-1) + startPrevRow])
            
            if idx > 2 && labels[idx] != blankLabel && labels[idx] != labels[idx-2]

                prevSum = log_plus_f(prevSum, alpha[(idx-2) + startPrevRow])
                
            end
            if idx < S - 2*(T-t) - 1
                alpha[idx + startCurrRow] = -Inf32
            else
                alpha[idx + startCurrRow] = prevSum + CUDAnative.log(probs[startProbCol + labels[idx]])
#                 alpha[idx + startCurrRow] = labels[idx]
#                 alpha[idx + startCurrRow] = S - 2*(T-t) - 1
            end   
            idx += blockDim().x
        end

        sync_threads()

        if tid == 0
            loglike = -Inf32
            val = 2*(L-1) + 1 - (L + repeats == T ? 1 : 0)

            start = val * (L != 0) + start
            last = val * (L != 0) + last
 
             for i=start:last
                loglike = log_plus_f(loglike, alpha[i + (T-1) * S])
             end
             nllForward[blockIdx().x] = -loglike
        end
        
        sync_threads()
        
    end

    return nothing
end

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
        i += blockDim().x
    end
    
    sync_threads()
#     
    for t=(T-1):(-1):1
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
            # TODO: the output[idx] gives a runtime incompatibility
            while idx <= S
                output[startCurRow + idx] = alphas[idx+startCurRow] + beta[startCurRow + idx]
                idx += NT
            end
            
            sync_threads()
        end
        
        #     # TODO: gradient calculation goes here
        
#         for i=1:length(accum)
#             accum[i] = -Inf32
#         end
#         
#         idx = tid
#         while idx <= S
#         
#             startAccRow = (t-1) * div(length(accum), T)
#             startOutputRow = (t-1) * S
#             labelIdx = labels[idx]
#             
#             accum[startAccRow + labelIdx] = log_plus_f(accum[startAccRow + labelIdx], output[startOutputRow + idx])
#             
#             idx += blockIdx().x
#         end
#         
#         sync_threads()
#         
#         idx = tid
#         while idx <= div(length(grad), T)
#         
#             startProbRow = (t - 1) * div(length(probs), T)
#             startOutputRow = (t - 1) * S
#             
#             s = -Inf32
#             for i=1:S
#                 s = log_plus_f(s, output[startOutputRow + i])
#             end
#             s = - s
# #             
# #             grad[startProbRow + idx] = probs[startprobRow] - s * exp(accum[startProbRow + idx])
# #         
# #             idx += blockIdx().x
#         end
        
        sync_threads()
        
        if t == 1 && tid == 1
            loglike = -Inf32
            val = 2 * (L-1) + 1 - (L + repeats == T ? 1 : 0)
            
            start = -val * (L != 0) + start
            last = -val * (L != 0) + last
#         
            for i=start:last
                loglike = log_plus_f(loglike, beta[i])
            end
        
            nllBackward[blockIdx().x] = -loglike
        end
        
        sync_threads()
    end

    return nothing
end

function ctc(ŷ, y)
#     blank = 62
    blank = 4
#     labels = indmax.([y[i,:] for i=1:size(y,1)])
    labels = argmax.([y[i,:] for i=1:size(y,1)])
    z = F(labels, 62)
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
    println()
    println(ŷ )
    println(size(alphas))
    println(typeof(alphas))
    #ŷ = Flux.Tracker.data(ŷ )
    ŷ  = CUDAdrv.CuArray(ŷ )
    nRepeats = countRepeats(labels)
    
    
    alphalikelihoods = CUDAdrv.CuArray{Float32}(size(ŷ,1))
    betalikelihoods = similar(alphalikelihoods)

    println("beginning alphas computation")
#     @cuda (size(alphas, 1), U′) computeAlphaKernel(ŷ, length(z), U′, nRepeats, CUDAdrv.CuArray(z), CUDAdrv.CuArray(z′), alphas, alphalikelihoods, blank)
    @cuda threads=U′ computeAlphaKernel(ŷ, length(z), size(ŷ,1), nRepeats, CUDAdrv.CuArray(z), CUDAdrv.CuArray(z′), alphas, alphalikelihoods, blank)
#     
# #     @cuda threads=U′ alpha2kernel(ŷ, alphas, T, U′, alphalikelihoods, CUDAdrv.CuArray(z′), blank)
# 
    println("alphas done")
    grads = similar(ŷ)
# 
    println("extracting alphas")
    display(Array(alphalikelihoods))
    println()
    println(Array(alphas))
    display(reshape(Array(alphas), 7, 4)')
    println()
    println("alphas extracted")
    
    println(typeof(betas))
    println("beginning betas computation")
    output = similar(betas)
    accum = similar(grads)
#     @cuda (size(betas, 1), U′) computeBetasAndGradKernel(ŷ, length(z), U′, nRepeats, CUDAdrv.CuArray(z′), alphas, alphalikelihoods, betalikelihoods, grads, blank)
    @cuda threads=U′ computeBetasAndGradKernel(ŷ, length(z), size(ŷ,1), nRepeats, CUDAdrv.CuArray(z′), alphas, betas, output, accum, alphalikelihoods, betalikelihoods, grads, blank)
    println("betas computed")

    # TODO: these produce an illegal memory access
    println(size(betas))
    display(Array(reshape(Array(betas), 7, 4)'))
    println()
    display(Array(betalikelihoods))
    println()
    display(Array(reshape(Array(output), 7, 4)'))
    println()
    display(Array(reshape(Array(grads), 4, 4)'))
    println()
end
