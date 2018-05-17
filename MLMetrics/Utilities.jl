
"""
NumberOf returns the number of an item in a list
"""

function NumberOf(item,list)
    return length(find(list.==item))
end

"""
NumberOfIntersection returns the number of items at
intersection of for two items of two lists
"""
function NumberOfIntersection(item_1,list_1,item_2,list_2)
    logic_1 = item_1.==list_1
    logic_2 = item_2.==list_2
    intersection_number = length(find(logic_1+logic_2.==2))
    return intersection_number
end

"""
contingency_table creates contingency table
"""

function contingency_table(U::AbstractArray,V::AbstractArray)
    U_clusters = collect(Set(U))
    V_clusters = collect(Set(V))
    R = length(U_clusters)
    C = length(V_clusters)
    CT = zeros(Int64,R,C)
    for i = 1:R
        for j = 1:C
            CT[i,j] = NumberOfIntersection(U_clusters[i],U,V_clusters[j],V)
        end
    end
    return CT
end
