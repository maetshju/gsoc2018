using Flux: gpu
using CuArrays

function ctc(x, y)

    function logadd(a, b)
        return log(a) + log(1+exp(log(b) - log(a)))
    end

    function F(A)
        prev = A[1]
        z = [prev]
        for curr in A[2:end]
            if curr != prev && curr != blank
                push!(l, curr)
            end
            prev = curr
        end
        return z
    end

    function addBlanks(z)

        prev = z[1]
        z′ = [prev]
        for curr in z
            if curr == prev
                push!(z′, blank)
            end
            push!(z′, curr)
            prev = curr
        end
        push!(z′, blank)
        insert!(z′, 1, blank)
        return z′
    end

    blank = 62

    ŷ = softmax.(model(x))
    z′ = addBlanks(F(indmax.(y)))
    T = size(ŷ, 1)
    U′ = length(z′)

    function α(t, u)
        mat = zeros(t, u)
        mat[1,1] = ŷ[1,blank]
        mat[1,2] = ŷ[1, z′[1]] # z′ is the modified version of the output sequence
        for i=2:t # start at 2 because everything in row 1 except [1,1:2] is 0
            for j=1:u
                idx = j-2
                idx += z′[j] == blank || z′[j-2] == z′[j]
                mat[i,j] = ŷ[i,z′[j] * sum(mat[i-1, idx:j])]
            end
        end
        return mat[t,u]
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

                    mat[i,j] = sum(mat[i+1,j:idx] .* ŷ[i, z′[j:idx]])
                end
            end
        # since we've built this from bottom-right to upper-left, we should be
        # okay, yes?
        end
        return mat[1,1]
    end

    s = 0
    for t=1:size(ŷ, 2)
        s += sum([α(t,u) * β(t,u) for u in 1:U′])
    end

    return -log(s)
end
