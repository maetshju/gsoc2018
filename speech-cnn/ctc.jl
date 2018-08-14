

"""
    logadd(a, b)

Adds log-space `a` and `b` such that the result equals `log(exp(a)+exp(b))`
"""
function logadd(a, b)
    isinf(a) && return b
    isinf(b) && return a
    
    if a < b
        a, b = b, a
    end
    
    return a + log(1+exp(b-a))
end

"""
    logsum(a)

Sums the elements in `a` such that the result equals `log(sum(exp.(a)))`
"""
function logsum(a)
    local s
    s = a[1]
    for item in a[2:end]
        s = logadd(s, item)
    end
    return s
end

"""
    F(A, blank)

Removes blanks and repetitions in the sequence `A`

This is the function `F` as defined in Graves (2012)
"""
function F(A, blank)
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
function addBlanks(z, blank)

    z′ = [blank]
    for label in z
        push!(z′, label)
        push!(z′, blank)
    end
    return z′
end

function ctc(ŷ::Array, y; activation=logsoftmax)

    ŷ = activation(ŷ)
    blank = size(ŷ, 1)
    
    z = F(indmax.([y[i,:] for i=1:size(y,1)]), blank)
    z′ = addBlanks(z, blank)
    T = size(ŷ, 2)
    U = length(z)
    U′ = length(z′)

    """
        α(t, u)

    Calculates the α coefficient for time `t` and label `u`
    """
    
    α = Array{Float64}(T, U′)
    for t=1:T
        for u=1:U′
            if t == u == 1
                α[t,u] = ŷ[t, blank]
            elseif t == 1 && u == 2
                α[t,u] = ŷ[t, z′[2]]
            elseif t == 1 && u > 2
                α[t,u] = -Inf
            elseif u < U′ - 2(T - t) - 1
                α[t,u] = -Inf
            else
                idx = u - 2
                idx += z′[u] == blank || (u > 2 && z′[u-2] == z′[u])
                idx = max(1, idx)
                
                α[t,u] = ŷ[z′[u], t] + logsum(α[t-1, idx:u])
            end
        end
    end
    
    β = Array{Float64}(T, U′)
    for i=1:length(β)
        β[i] = -Inf
    end
    β[T,U′] = 0.0
    
    for t=T:-1:1
        for u=(U′-1):-1:1
            if t == T && u >= U′ - 1
                β[t,u] = 0.0
            elseif t == T && u < U′ - 1
                continue
            elseif u > 2t || u > U′ + 1
                continue
            else
                idx = u+2
                idx -= z′[u] == blank || (idx < U′ && z′[u+2] == z′[u])
                idx = min(idx, U′)
                
                v = [β[t+1,i] + ŷ[z′[i], t+1] for i=u:idx]
                β[t, u] = logsum(v)
            end
        end
        if t < T-1
            β[t, U′] = β[t+1, U′] + ŷ[blank, t]
        end
    end
    
    losses = Vector()
    for t=1:T
        v = [α[t,u] + β[t,u] for u in 1:U′]
        push!(losses, -logsum(v))
    end
    
    accum = reshape([-Inf for x=1:length(ŷ )], size(ŷ ))
    grads = reshape([-Inf for x=1:length(ŷ )], size(ŷ ))
    
    for t=1:T
        for u=1:U′
            accum[z′[u], t] = logadd(accum[z′[u], t], α[t,u] + β[t,u])
        end
        for u=1:size(grads, 1)
            grads[u,t] = exp(ŷ[u, t]) - exp(accum[u, t] - -losses[t])
        end
    end

    return mean(losses), grads
end
