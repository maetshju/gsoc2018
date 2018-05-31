using Flux: gpu
using CuArrays
using Memoize

# TODO: need to convert to log space
# TODO: see if we need the loss to be a Tracked value

function ctc(ŷ, y)

    # function logadd(a, b)
    #     return log(a) + log(1+exp(log(b) - log(a)))
    # end

    """
        logadd(a, b)

    Returns the value of log(a + b), assuming that a and b
    have already been converted to log space
    """
    function logadd(a, b)
        # if !isinf(a) && isinf(b)
        #     return a
        # elseif isinf(a) && !isinf(b)
        #     return b
        # end
        if isinf(a) || isinf(b)
            return log(exp(a) + exp(b))
        end
        return a + log(1+exp(b-a))
    end

    function logadd(a)
        s = a[1]
        for item in a[2:end]
            s = logadd(s, item)
        end
        return s
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

    function addBlanks(z)

        z′ = [blank]
        for label in z
            push!(z′, label)
            push!(z′, blank)
        end
        return z′
    end

    # blank = 62
    blank = length(ŷ[1])

    # ŷ = Flux.Tracker.data.(ŷ)
    lgŷ = [CUDAnative.log.(ŷI) for ŷI in ŷ]
    z = F(indmax.(y))
    z′ = addBlanks(z)
    T = length(ŷ)
    U′ = length(z′)

    @memoize function α_r(t, u)

        if t == u == 1
            return lgŷ[t][blank]
        end
        
        if t == 1 && u == 2
            return lgŷ[t][Flux.Tracker.data(z[1])]
        end
        

        if t == 1 && u > 2
            return 0
        end
        
        idx = u - 2
        idx += z′[u] == blank || (u > 2 && z′[u-2] == z′[u])
        idx = max(1, idx)
        
        vals = Vector()
        
        for n=idx:u
            push!(vals, α_r(t-1, n))
        end
        
        return lgŷ[t][Flux.Tracker.data(z′[u])] + logadd(vals)
        
#        return lgŷ[t][Flux.Tracker.data(z′[u])] + logadd([n -> α_r(t-1, n) for n in idx:u])
    end



    function α()
        println("entering alpha")
        mat = param(gpu(log.(zeros(T, U′))))
        println("filling (1, 1)")
        println(typeof(lgŷ[1][blank]))
        println(lgŷ[1][blank])
        mat[1,1] = lgŷ[1][blank]
        println("filling (1, 2")
        mat[1,2] = lgŷ[1][Flux.Tracker.data(z[1])]

        println("beginning alpha iterations")

        for t=2:T
            for u=1:U′
                if u >= U′ - 2 * (T - t) - 1
                    idx = u-2
                    if z′[u] == blank || (u > 2 && z′[u-2] == z′[u])
                        idx += 1
                    end
                    idx = max(1,idx)
                    # mat[t,u] = ŷ[t][z′[u]] * sum(mat[t-1, idx:u])
                    # s = mat[t-1, idx]
                    # for n=(idx+1):u
                    #     s = logadd(s, mat[t-1, n])
                    # end
                    mat[t,u] = lgŷ[t][Flux.Tracker.data(z′[u])] + logadd(mat[t-1, idx:u])
                end
            end
        end
        return mat
    end

    @memoize function β_r(t, u)
        if u >= U′ -1
            return 1
        end

        if u < U′ - 1
            return 0
        end

        idx = u+1
        idx += z′[u] == blank || (idx < U′ && z′[u+2] == z′[u])
        idx = min(idx, U′)
        
        vals = Vector()
        for n=u:idx
           push!(vals, β_r(t+1, n) + lgŷ[t+1][Flux.Tracker.data(z′[n])])
        end
        
        
#        return logadd([n -> (β_r(t+1, n) + lgŷ[t+1][z′[n]]) for n in u:idx])
        return logadd(vals)

        s = β_r(t+1, u) * ŷ[t][z′[u]]

        for i=u:idx
            s += β_r(t+1, i)
        end
    end

    function β()
        mat = gpu(log.(zeros(T, U′)))
        mat[T,1:U′-2] = log(0)
        for t=T:-1:1
            for u=U′:-1:1
                if u > 2*t
                    mat[t,u] = log(0)
                elseif t == T && u >= U′ - 1
                    mat[t,u] = log(1)
                elseif t == T && u < U′ - 1
                    mat[t,u] = log(0)
                else
                    idx = u+2
                    idx -= z′[u] == blank || (idx < U′ && z′[u+2] == z′[u])
                    idx = min(idx, U′)

                    # mat[t,u] = sum(mat[t+1,u:idx] .* ŷ[t+1][z′[u:idx]])

                    prods = mat[t+1,u:idx] .+ lgŷ[t+1][z′[u:idx]]
                    # s = prods[1]
                    # for p in prods[2:end]
                    #     s = logadd(s, p)
                    # end
                    mat[t, u] = logadd(prods)
                end
            end
        end
        return mat
    end

#    alpha = α()
#    beta = β()


    s = logadd([α_r(1, u) + β_r(1, u) for u in 1:U′])
    for t=2:length(ŷ)
        vals = [α_r(t, u) + β_r(t, u) for u in 1:U′]
#        push!(s, logadd(vals))
        s += logadd(vals)
    end

#    s = Vector()
#    for t=1:length(ŷ)
#        coeffs = Vector()
#        for u = 1:U′
#            push!(coeffs, α_r(t, u) + β_r(t, u))
#        end
##        coeffs = [u -> (α_r(t, u) + β_r(t, u)) for u in 1:U′]
#        println("s $(typeof(s))")
#        println("coeffs $(logadd(coeffs)) $(typeof(logadd(coeffs)))")
#        push!(s, logadd(coeffs))
#    end
    
#    s = -sum(s)
    s = -s

#    s = sum(-[logadd(alpha[t,1:U′] .+ beta[t,1:U′]) for t in 1:T])
    return s
end
