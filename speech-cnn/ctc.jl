using Flux: gpu
using CUDAnative
using Memoize
include("shareddict.jl")

#EPS = 1e-7

function ctc(ŷ, y; gpu=true, eps=true)

    """
        logadd(a, b)

    Adds log-space `a` and `b` such that the result equals `log(exp(a)+exp(b))`
    """
        function logadd(a, b)
        if isinf(a)
            return b
        elseif isinf(b)
            return a
        end
	if isnan(a) || isnan(b)
	    println("NAN HAPPENED IN LOGADD")
	end
        a = a + log(1+exp(b-a))
        return nothing
    end
#     function logadd(a, b)
#         if isinf(a)
#             return b
#         elseif isinf(b)
#             return a
#         end
# 	if isnan(a) || isnan(b)
# 	    println("NAN HAPPENED IN LOGADD")
# 	end
#         return a + log(1+exp(b-a))
#     end

    """
        logsum(a)

    Sums the elements in `a` such that the result equals `log(sum(exp.(a)))`
    """
    function logsum(a)
        local s
        s = a[1]
        for item in a[2:end]
            s = logadd(s, item)
	    if isnan(s)
    	        println("NANS HAVE ENTERED")
            end
        end
        return s
    end
    
    function logsum(a, s)
        s = a[1]
        for item in a[2:end]
            logadd(s, item)
        end
        return nothing
    end

    """
        F(A)

    Removes blanks and repetitions in the sequence `A`

    This is the function `F` as defined in Graves (2012)
    """
    function F(A)
        prev = A[1]
        z = [prev]
        for curr in A[2:end]
            if curr != prev && curr != blank
                push!(z, curr)
    ``        end
            prev = curr
        end
        return z
    end

    """
        addBlanks(z)

    Adds blanks to the start and end of `z`, and between item in `z`
    """
    function addBlanks(z)

        z′ = [blank]
        for label in z
            push!(z′, label)
            push!(z′, blank)
        end
        return z′
    end

    blank = size(ŷ[1], 1)

    println(typeof(ŷ ))
    println(size(ŷ ))
    println(typeof(y))
    println(typeof(y))

    if eps
        #ŷ  = [yI .+ EPS for yI in ŷ ]
	ŷ  += EPS
    end
    #println(ŷ[1])

    println(typeof(ŷ ))
    local lgŷ
    #=if gpu
        lgŷ = [CUDAnative.log.(ŷI) for ŷI in ŷ]
    else
        lgŷ = [log.(ŷI) for ŷI in ŷ]
    end=#
    lgŷ  = Flux.gpu(CUDAnative.log.(ŷ )')
    println("lgyhat type $(typeof(lgŷ ))")
    println(size(lgŷ ))
    #println(lgŷ )
    z = F(indmax.([y[i,:] for i=1:size(y,1)]))
    z′ = addBlanks(z)
    T = size(ŷ, 2)
    U′ = length(z′)
    println( U′)
    
#     alphas = Dict{Tuple{Int64,Int64},Flux.Tracker.TrackedReal}()

    """
        α(t, u)

    Calculates the α coefficient for time `t` and label `u`
    """
    
    function α(t, u, alpha)
    
        if haskey(alphas, (t,u))
           return alphas[(t,u)]
        end
        
        local v

        if t == u == 1
            v = lgŷ[t, blank]
#             return lgŷ[t, blank]

        elseif t == 1 && u == 2
            v = lgŷ[t, Flux.Tracker.data(z[1])]
#             return lgŷ[t, Flux.Tracker.data(z[1])]

        elseif t == 1 && u > 2
            v = Flux.Tracker.TrackedReal(Float32(log(0)))
#             return Flux.Tracker.TrackedReal(Float32(log(0)))

        elseif u < U′ - 2(T - t) - 1
            v = Flux.Tracker.TrackedReal(Float32(log(0)))
#             return Flux.Tracker.TrackedReal(Float32(log(0)))

        else
            v = Vector()
            idx = u - 2
            idx += z′[u] == blank || (u > 2 && z′[u-2] == z′[u])
            idx = max(1, idx)
            
            for i=idx:u
                var = Flux.Tracker.TrackedReal(0)
                @cuda (1,1) α(t-1, i, var)
                push!(v, var)
            end
            
            @cuda (1,1) logsum(v, loggedsum)
            v = lgŷ[t, Flux.Tracker.data(z′[u])] + v
        end

        parInsert(alphas, (t,u), v)
        return nothing
    end

    """
        β(t, u)

    Calculates the β coefficient at time `t` and label `u`
    """
    @memoize Dict function β(t, u)

        if t == T && u >= U′ -1
            return Flux.Tracker.TrackedReal(Float32(log(1)))
        end

        if t == T && u < U′ - 1
            return Flux.Tracker.TrackedReal(Float32(log(0)))
        end

        if u > 2t || u > U′ + 1
            return Fux.Tracker.TrackedReal(Float32(log(0)))
        end

        idx = u+2
        idx -= z′[u] == blank || (idx < U′ && z′[u+2] == z′[u])
        idx = min(idx, U′)

        vals = [β(t+1, i) + lgŷ[t+1, Flux.Tracker.data(z′[i])] for i=u:idx]
	if any(isnan.(vals))
	    println("NANS IN BETA ($(t), $(u))")
	end

        return logsum(vals)
    end
    
#     function alphaiter(alphas)
#         alphas[1,1] = lgŷ[1, blank]
#         alphas[1,2] = lgŷ[1, Flux.Tracker.data(z[1])]
#         for t=2:T
#             for u=1:U′
#                 idx = u-2
#                 idx += z′[u] == blank || (u > 2 && z′[u-2] == z′[u])
#                 idx = max(1, idx)
#                 
#                 alphas[t,u] = lgŷ[t, Flux.Tracker.data(z′[u])] + logsum(alphas[t-1, idx:u])
#             end
#         end
#         return nothing
#     end
#     
#     function alphaiter2(alphas)
#         i = (blockIdx().x-1) * blockDim().x + threadIdx().x
#         t = floor(Int, i / size(alphas, 1)) + 1
#         u = i % size(alphas, 1)
#         
#         if t == u == 1
#             alphas[t,u] = lgŷ[1, blank]
#         elseif t == 1 && u == 2
#             alphas[t,u] = lgŷ[1, Flux.Tracker.data(z[1])]
#         else
#             idx = u-2
#             idx += z′[u] == blank || (u > 2 && z′[u-2] == z′[u])
#             idx = max(1, idx)
#             
#             alphas[t,u] = lgŷ[t, Flux.Tracker.data(z′[u])] + logsum(alphas[t-1, idx:u])
#         end
#         
#         return nothing
#     end
    
    function onealpha(t, u, prevt)
        idx = u-2
        idx += z′[u] == blank || (u > 2 && z′[u-2] == z′[u])
        idx = max(1, idx)
        return lgŷ[t, Flux.Tracker.data(z′[u])] + logsum(prevt[idx:u])
    end
    
    println("Beginning alphas")
    alphas = Flux.gpu([Flux.Tracker.TrackedReal(-Inf) for x in 1:T])
    alphas[1,1] = lgŷ[1, blank]
    alphas[1,2] = lgŷ[1, Flux.Tracker.data(z[1])]
    for t=2:T
#         prev = repeat(alphas[t-1,:], outer=U′)
        prev = alphas[t-1,:]
        alphas = vcat(alphas, map(x -> onealpha(t, x, prev)))
#         alphas = vcat(alphas, onealpha.(enumerate(prev)...))
    end
    println("Alphas done")
    
#     karr = Flux.gpu(SharedArray([(0, 0) for x in 1:2^26]))
#     varr = Flux.gpu(SharedArray([0.0 for x in 1:2^26]))
#     karr = Flux.gpu([(0, 0) for x in 1:2^26])
#     varr = Flux.gpu([0.0 for x in 1:2^26])
#     alphas = SharedDict(karr, varr, (0,0), 0.0)
    
    # TODO: make alpha use the parallel hash table for memoization

    println("beginning alphas")
#     a = Flux.Tracker.TrackedReal(0)
#     @cuda (1,12) α(T, U′, a)
#     mat = Flux.gpu(Flux.Tracker.TrackedReal.Float32.((log.(zeros(T, U′)))))
    mat = similar(ŷ)
    println(typeof(mat))
#     mat = Flux.gpu(Matrix{Flux.Tracker.TrackedReal}(T, U′))
    @cuda (1,12) alphaiter(mat)
    println("alphas done")
    println("a $(mat[1,1])")
    map(b -> β(b[1], b[2]), beta)

    #=losses = Vector()
    for t=1:T
        v = [alpha[t,u] + beta[t,u] for u in 1:U′]
        push!(losses, -logsum(v))
    end

    return sum(losses)=#

    println("mapping done")

    # TODO: Make alpha and beta type stable

    # s = logsum([α(1, u) + β(1, u) for u in 1:U′])
    losses = Vector()
    println(size(ŷ ,2))
    for t=T:-1:1
        v = [α(t, u) + β(t, u) for u in 1:U′]
	#println(v)
        l = -logsum(v)
	#println(l)
        push!(losses, l)
    end

    #return sum(losses[end:-1:max(end-50,1)])

    return sum(losses)
    #return mean(losses)
    #return losses
end
