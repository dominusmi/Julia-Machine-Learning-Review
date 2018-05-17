include("Utilities.jl")


function H(U)
    result = 0
    N = length(U)
    clusters = Set(U)
    for c in clusters
        a = NumberOf(c,U)
        result += (a/N) * log(a/N)
    end
    return -result
end


function H(U,V)
    size(U) == size(V) || throw(DimensionMismatch("size of U not equal to size of V"))
    result = 0
    N = length(U)
    clusters_U = Set(U)
    clusters_V = Set(V)
    for cu in clusters_U
        for cv in clusters_V
            n = NumberOfIntersection(cu,U,cv,V)
            if n!=0
                result += (n/N) * log(n/N)
            end
        end
    end
    return -result
end


function Expected_MI(U::AbstractArray,V::AbstractArray)
    size(U) == size(V) || throw(DimensionMismatch("size of U not equal to size of V"))
    result = 0
    N = length(U)
    U_clusters = Set(U)
    V_clusters = Set(V)
    R = length(U_clusters)
    C = length(V_clusters)
    CT = contingency_table(U,V)
    a = reshape(sum(CT,2),R)
    b = reshape(sum(CT,1),C)
    for i = 1:R
        for j = 1:C
            for nij = max(1,a[i]+b[j]-N):min(a[i],b[j])
                result += (nij/N)*log(N*nij/(a[i]*b[j])) *
                (factorial(a[i]) * factorial(b[j]) * factorial(N - a[i]) * factorial(N - b[j]))/
                (factorial(N) * factorial(nij) * factorial(a[i] - nij) * factorial(b[j] - nij) * factorial(N - a[i] - b[j] + nij))
            end
        end
    end
    return result
end

function H_conditional(U::AbstractArray,V::AbstractArray)
    size(U) == size(V) || throw(DimensionMismatch("size of U not equal to size of V"))
    result = 0
    N = length(U)
    U_clusters = Set(U)
    V_clusters = Set(V)
    R = length(U_clusters)
    C = length(V_clusters)
    CT = contingency_table(U,V) 
    a = reshape(sum(CT,2),R)
    b = reshape(sum(CT,1),C)
    for i = 1:R
        for j = 1:C
            result += (CT[i,j]/N) * log(CT[i,j]/b[j])
        end    
    end
    return result
end
