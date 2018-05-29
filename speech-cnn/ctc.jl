using Flux: gpu
using CuArrays

# TODO: need to convert to log space
# TODO: see if we need the loss to be a Tracked value

function ctc(ŷ, y)

    function logadd(a, b)
        return log(a) + log(1+exp(log(b) - log(a)))
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
    blank = length(y[1])

    println(typeof(ŷ))
    ŷ = Flux.Tracker.data.(ŷ)
    println(typeof(ŷ))
    z′ = addBlanks(F(indmax.(y)))
    T = length(ŷ)
    U′ = length(z′)

    function α()
        mat = gpu(zeros(T, U′))
        mat[1,1] = ŷ[1][blank]
        mat[1,2] = ŷ[1][z′[1]]

        for t=2:T
            for u=1:U′
                if u >= U′ - 2 * (T - t) - 1
                    idx = u-2
                    if z′[u] == blank || (u > 2 && z′[u-2] == z′[u])
                        idx += 1
                    end
                    idx = max(1,idx)
                    mat[t,u] = ŷ[t][z′[u]] * sum(mat[t-1, idx:u])
                end
            end
        end
        return mat
    end


    function α(t, u)
        println(z′)
        println(ŷ)
        mat = gpu(zeros(t, u))
        mat[1,1] = ŷ[1][blank]
        mat[1,2] = ŷ[1][z′[1]] # z′ is the modified version of the output sequence
        for i=2:t # start at 2 because everything in row 1 except [1,1:2] is 0
            for j=1:u
                idx = j-2
                idx += z′[j] == blank || z′[j-2] == z′[j]
                mat[i,j] = ŷ[i][z′[j] * sum(mat[i-1, idx:j])]
            end
        end
        return mat[t,u]
    end

    function β()
        println("entering beta")
        mat = gpu(zeros(T, U′))
        mat[T,1:U′-2] = 0
        for t=T:-1:1
            for u=U′:-1:1
                if u > 2*t
                    mat[t,u] = 0
                elseif t == T && u >= U′ - 1
                    mat[t,u] = 1
                elseif t == T && u < U′ - 1
                    mat[t,u] = 0
                else
                    idx = u+1
                    idx += z′[u] == blank || (idx < U′ && z′[u+2] == z′[u])
                    idx = min(idx, U′)
                    mat[t,u] = sum(mat[t+1,u:idx] .* ŷ[t][z′[u:idx]])
                end
            end
        end
        return mat
    end

    function β(t, u)
        mat = gpu(ones[T-t. U′-u])
        for i=T:-1:t
            for j=U′:-1:u
                if i == T && j >= U′ - 1
                    mat[i,j] = 1 # not necessary if we just skip this condition altogether
                elseif j < U′ - 1
                    mat[i,j] = 0
                else
                    idx = j+1
                    idx += z′[j] == blank || z′[j+2] == z′[j]

                    mat[i,j] = sum(mat[i+1,j:idx] .* ŷ[i][z′[j:idx]])
                end
            end
        # since we've built this from bottom-right to upper-left, we should be
        # okay, yes?
        end
        return mat[1,1]
    end

    # s = 0
    # for t=1:length(ŷ)
    #     s += sum([α(t,u) * β(t,u) for u in 1:U′])
    # end
    # s = sum([α(T,u) * β(T,u) for u in 1:U′])
    alpha = α()
    beta = β()

    s = 0
    for t=1:length(ŷ)
        s += -log(sum(alpha[t,1:U′] .* beta[t,1:U′]))
    end

    return s
end
