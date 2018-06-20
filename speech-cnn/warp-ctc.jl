# a port of Baidu's C++ warp-ctc package
# GitHub: https://github.com/baidu-research/warp-ctc/
# paper: https://arxiv.org/pdf/1512.02595.pdf

using Flux
using CUDAnative, CUDAdrv

function log_plus_f(p1, p2)
    if isinf(p1)
        return p2
    end

    if isinf(p2)
        return p1
    end

    return p1 + log(1+exp(p2 - p1))
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
                            labelsWithoutBlanks, labelsWithBlanks, alpha, nllForward, blankLabel, ninf)


    
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
    start = (L + repeats < T) ? 0 : 1
    last = S > 1 ? 2 : 1

    i = tid
    while i <= (last - start)
        alpha[i + start] = log(probs[labels[i+start]])
        i += blockDim().x
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

        idx = tid+1
        while idx <= S-1
#         for idx=(tid+1):blockDim().x:(S-1)
            prevSum = log_plus_f(alpha[idx + startPrevRow], alpha[(idx-1) + startPrevRow])
            
            if labels[idx] != blankLabel && idx != 1 && labels[idx] != labels[idx-2]

                prevSum = log_plus_f(prevSum, alpha[(idx-2) + startPrevRow])
                alpha[idx + startCurrRow] = prevSum + log(probs[startProbCol + labels[idx]])
                
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
        
    end

    return nothing
end

function computeBetasAndGradKernel(probs, labelSize, uttLength,
                                    repeatsInLabel, labelsWithBlanks,
                                    alphas, nllForward, nllBackward,
                                    grad, blankLabel, ninf)
                                    
    tid = threadIdx().x
    L = labelSize
    T = uttLength
    S = 2*L + 1
    repeats = repeatsInLabel
    logPartition = -nllForward[blockIdx().x]
    
    labels = labelsWithBlanks
    alpha = alphas[blockIdx().x]
    output = similar(alphas)
    
    if (L+repeats) > T
        return nothing
    end
    
    start = S > 1 ? S-2 : 0
    last = L + repeats < T ? S : S-1
    
    sync_threads()
    
    i = tid
    while i <= last - start
        beta[i+start] = log(probs[(T-1) + label[i+start]])
        i += blockDim().x
    end
    
    if t < T-1
        # for t in reverse(1:T)
        for t=T::-1:1
            startCurRow = t*S
            startProbCol = t
            
            idx = tid
            i = 1
            while idx < S-1
                
                nextSum = log_plus_f(beta[idx], beta[idx+1])
                
                if label[idx] != blankLabel && idx != S-2 && label[idx] != label[idx+2]
                    nextSum = log_plus_f(nextSum, beta[idx+1])
                end
                
                beta[i] = nextSum + log(probs(startProbCol + labels[idx]]))
                
                idx += NT
                i += 1
            end
        
            sync_threads()
            
            if tid == 0 && last == S
                beta[S-1] = beta[S-1] + log(probs[startProbCol + blankLabel])
            end
            
            idx = tid
            while idx < S
                output[idx] = alpha[idx+startCurRow] + beta[idx]
                idx += NT
            end
            
            sync_threads()
        end
        
        sync_threads()
        
        # TODO: gradient calculation goes here
        
        if t == 0 && tid == 0
            loglike = -Inf32
            val = 2 * (L-1) + 1 - (L + repeats == T ? 1 : 0)
            
            start = -val * (L != 0) + start
            last = -val * (L != 0) + last
            
            for i=start:last
                loglike = log_plus_f(loglike, beta[i])
            end
            
            nllBackward[blockIdx().x] = -loglike
        end
        
        sync_threads()
    end
end

function ctc(ŷ, y)
    blank = 62
    labels = indmax.([y[i,:] for i=1:size(y,1)])
    z = F(labels, 62)
    z′ = [blank]
    for label in z
        push!(z′, label)
        push!(z′, blank)
    end
    U′ = 2*length(z) + 1
#     alphas = Flux.TrackedArray([-Inf for x in 1:(size(ŷ,1) * U′)])
    alphas = CUDAdrv.CuArray([-Inf32 for x in 1:(size(ŷ,1) * U′)])
    println(size(alphas))
    println(typeof(alphas))
    ŷ = Flux.Tracker.data(ŷ )
    
    alphalikelihoods = CUDAdrv.CuArray{Float32}(size(ŷ,1))
    betalikelihoods = similar(alphalikelihoods)

    println("beginning alphas computation")
    @cuda (size(alphas, 1), U′) computeAlphaKernel(ŷ, length(z), U′, countRepeats(labels), CUDAdrv.CuArray(z), CUDAdrv.CuArray(z′), alphas, alphalikelihoods, blank, -Inf32)

    println("alphas done")
    
    betas = similar(alphas)
    println("beginning betas computation")
    @cuda(size(betas, 1), U′) computeBetasAndGradKernel(ŷ,, length(z), U′, countReapts(label), CUDAdrv.CuArray(z′), alphas, alphalikelihoods, betalikelihoods, GRAD, blank, -Inf32)
end
