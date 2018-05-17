include("Utilities.jl")
include("Entropy.jl")

function mutual_info_score(labels_true::AbstractArray,labels_pred::AbstractArray)
    size(labels_true) == size(labels_pred) || throw(DimensionMismatch("size of labels_true not equal to size of labels_pred"))
    N = length(labels_true)

    clusters_true = Set(labels_true)
    clusters_pred = Set(labels_pred)
    Mutual_Info = 0

    for ct in clusters_true
        for cp in clusters_pred
            intersection_number = NumberOfIntersection(ct,labels_true,cp,labels_pred)
            true_number = NumberOf(ct,labels_true)
            pred_number = NumberOf(cp,labels_pred)
            temp = (intersection_number/N)*log(N*intersection_number/(true_number*pred_number))
            if !isnan(temp)
                Mutual_Info  += temp
            end
        end
    end
    return Mutual_Info
end

function normalized_mutual_info_score(labels_true::AbstractArray,labels_pred::AbstractArray, mode::String = "sqrt")
    NMI = 0
    if mode == "sqrt"
        NMI = mutual_info_score(labels_true,labels_pred)/sqrt(H(labels_true)*H(labels_pred))
    elseif mode == "max"
        NMI = mutual_info_score(labels_true,labels_pred)/max(H(labels_true),H(labels_pred))
    elseif mode == "min"
        NMI = mutual_info_score(labels_true,labels_pred)/min(H(labels_true),H(labels_pred))
    elseif mode == "sum"
        NMI = 2*mutual_info_score(labels_true,labels_pred)/(H(labels_true)+H(labels_pred))
    elseif mode == "joint"
        NMI = 2*mutual_info_score(labels_true,labels_pred)/H(labels_true,labels_pred)
    end
    return NMI
end

function adjusted_mutual_info_score(labels_true::AbstractArray, labels_pred::AbstractArray ,mode::String = "max")
    AMI = 0
    if mode == "sqrt"
        AMI = (mutual_info_score(labels_true,labels_pred) -  Expected_MI(labels_true,labels_pred))/
        (sqrt(H(labels_true)*H(labels_pred)) - Expected_MI(labels_true,labels_pred))
    elseif mode == "max"
        AMI = (mutual_info_score(labels_true,labels_pred) -  Expected_MI(labels_true,labels_pred))/
        (max(H(labels_true),H(labels_pred)) - Expected_MI(labels_true,labels_pred))
    elseif mode == "min"
        AMI = (mutual_info_score(labels_true,labels_pred) -  Expected_MI(labels_true,labels_pred))/
        (min(H(labels_true),H(labels_pred)) - Expected_MI(labels_true,labels_pred))
    elseif mode == "sum"
        AMI = (mutual_info_score(labels_true,labels_pred) -  Expected_MI(labels_true,labels_pred))/
        ((H(labels_true)+H(labels_pred))/2 - Expected_MI(labels_true,labels_pred))
    end
    return AMI
end

function adjusted_rand_score(labels_true::AbstractArray,labels_pred::AbstractArray)
    size(U) == size(V) || throw(DimensionMismatch("size of U not equal to size of V"))

    N = length(labels_pred)
    CT = contingency_table(labels_true,labels_pred)
    a = reshape(sum(CT,2),length(Set(labels_true)))
    b = reshape(sum(CT,1),length(Set(labels_pred)))

    Index = sum(CT.*(CT.-1)./2)
    Expected_Index = sum(a.*(a.-1)./2)*sum(b.*(b.-1)./2)/(N*(N-1)/2)
    Max_Index = 0.5 * (sum(a.*(a.-1)./2) + sum(b.*(b.-1)./2))

    return (Index-Expected_Index)/(Max_Index-Expected_Index)
end
