include("../Mondrian_Forest_Classifier.jl");

import Base.copy
using Distributions
using MLBase

function reset_root(T::Mondrian_Tree)
    j = get(T.root)
    if j.node_type != [false,false,true]
        if get(j.parent).node_type == [false,false,true]
            T.root = j.parent
            return
        else
            j = get(j.parent)
        end
    end
end

"""
`function Extend_Mondrian_Tree!(T::Mondrian_Tree,λ::Float64,X::Array{Float64} where N,Y::Int64) `

This function extends an already existing Mondrian Tree by ONE new datapoint that gets incorperated in the tree.

`Input`: Mondrian Tree T (abstract type Mondrian_Tree), Lifetime parameter λ (Float64), 1dim Array of Features Array X (Array of Float64), classlabel Y (Int64)

`Output`: Mondrian Tree

`Files needed to run this function`: "Mondrian_Forest_Classifier.jl", "Mondrian_extention.jl"

This function calls the function Extend_Mondrian_Block.

The usage of the function "expand!" is recommended to expand Mondrian Trees as it has a nicer user interface.

"""
## Algorithm 3 in the Paper "Mondrian Forests: Efficient Online Random Forests"
function Extend_Mondrian_Tree!(T::Mondrian_Tree,λ::Float64,X::Array{Float64},Y::Int64, D)
    ϵ=get(T.root)
    Extend_Mondrian_Block!(T,λ,ϵ,X,Y, D)
    return T
end


"""
`function Extend_Mondrian_Block!(T::Mondrian_Tree,λ::Float64,j::Mondrian_Node,X::Array{Float64},Y::Int64)`

This function extends a Mondrian Block to incorporate ONE new datapoint.

`Input`: Mondrian Tree T (abstract type Mondrian_Tree), Lifetime parameter λ (Float64), Mondrian Node j (abstract type Mondrian_Node), 1dim Array of Features Array X (Array of Float64), classlabel Y (Int64)

`Output`: Modified nodes in the input tree

`Files needed to run this function`: Mondrian_Forest_Classifier.jl", "Mondrian_extention.jl"

This function calls the functions

- Extended_dimension()

- sample_extended_split_dimension

- Sample_Mondrian_Block!

- update_intervals


"""

## Algorithm 4 in the Paper
function Extend_Mondrian_Block!(T::Mondrian_Tree,λ::Float64,j::Mondrian_Node,X::Array{Float64},Y::Int64, D)
reset_root(T)
if sum(j.c .> 0) == 1  #check if all labels are identical
     Θ = update_intervals(get(j.Θ),X)        # update extent
     j.Θ=Θ
         if findmax(j.c)[2] == Y
             #i = findmax(j.c)[2]
             #j.c[i] = j.c[i]+1
             UpdatePosteriorCounts(j,Y)
             return
         else
            j.node_type = [true,false,false]
            p_data,p_labels = paused_train_data(j,X,Y)
            j.τ = get(j.parent).τ
            Sample_Mondrian_Block!(j,get(j.Θ),λ,T,convert(Array{Float64,2},p_data),convert(Array{Int64,1},p_labels))
         end

else
    #println("not a leaf")
    E = rand(Exponential(1/Extended_dimension(get(j.Θ),X)))  #sample value E
    if j.node_type[3]==true
        # check if the node we're looking at is the root (if yes the split time is assumed to be 0)
            #println("j is the root")
        τₚ = 0
    else
            #println("j is not the root")
        τₚ = (get(j.parent)).τ                               # if it's not the root get the split time of the node above j
    end
    if τₚ + E < j.τ                                          # check if our split happens in time
        d,x= sample_extended_split_dimension(get(j.Θ),X)     # sample new split dimension / split direction
        Θ = update_intervals(get(j.Θ),X)                     # get the boxes for the new node
        j_wave = init_j_wave(T,j,E,d,x,Θ)
        j.parent=j_wave
        j_prime = init_j_prime(j_wave,Y)
        A,j_wave,j_prime = check_left_right!(j_wave,j,j_prime,X,d,x)
        ind = get_data_indices(get(j_prime.Θ), D[1])
        t = D[2][ind]
        for k in 1:length(j_prime.c)
            j_prime.c[k] = length(t[t.==k])
        end
            j_prime.τ = τₚ+E
        Sample_Mondrian_Block!(j_prime,get(j_prime.Θ),λ,T,D[1][ind,:],D[2][ind])


    else                                 # if the split didn't occur in time
        Θ = update_intervals(get(j.Θ),X)        # update the boxes of j
        j.Θ=Θ
        if j.node_type != [false,true,false]    # check if j is a leaf
            if X[get(j.δ)] <= get(j.ζ)           # if the new datapoint is in the boxes associated with the left child of j -> Extend towards the left child, else the right

                Extend_Mondrian_Block!(T,λ,get(j.left),X,Y,D)
            else

                Extend_Mondrian_Block!(T,λ,get(j.right),X,Y,D)

            end
        end
    end
end
end

"""
`function expand!(T::Mondrian_Tree_Classifier,X::Array{Float64,N} where N,Y::Array{Int64},λ::Float64)`

This function expands an already sampled Mondrian Tree Classifier by a desired number of datapoints.

`Input`: Mondrian Tree Classifier T (abstract type Mondrian_Tree_Classifier), array of features X (Array of Float64), array of class labels (1dim of Float 64), Lifetime parameter λ (Float 64)

Each row in the array X represents one set of features, the corresponding row in Y represents the class label.

`Output`: Mondrian Tree Classifier with incoporated new datapoints

`Files needed to run this function`: Mondrian_Forest_Classifier.jl", "Mondrian_extention.jl"

This function calls the function Extend_Mondrian_Tree.
"""

function expand!(T::Mondrian_Tree_Classifier,X::Array{Float64,N} where N,Y::Array{Int64},λ::Float64)
    # puts the extention in a nice framework, allows to extend by multiple datapoints
    for i=1:length(X[:,1])
        T.X = vcat(T.X,X[i,:]')
        T.Y = vcat(T.Y,Y[i])
        T.Tree=Extend_Mondrian_Tree!(T.Tree,λ,X[i,:],Y[i], (T.X, T.Y));
    end
    compute_predictive_posterior_distribution!(T.Tree,10*size(X,2))
    return T
end

"""
`function expand_forest!(MF::Mondrian_Forest_Classifier,X_extend, Y_extend,λ)`

This function expands an already sampled Mondrian Forest Classifier by a desired number of datapoints.


`Input`: Mondrian Forest Classifier MF (abstract type Mondrian_Forest_Classifier), array of features X_extend to extend the forest on, Array of class labels Y_extend corresponding to the new features, life time parameter λ

`Output`: Mondrian Forest Classifier with incoporated new datapoints

`Files needed to run this function`: Mondrian_Forest_Classifier.jl", "Mondrian_extention.jl"

This function calls the function expand!.
"""

function expand_forest!(MF::Mondrian_Forest_Classifier,X_extend, Y_extend,λ)
    X=MF.X
    Y=MF.Y
    if size(X)[2] != size(X_extend)[2]
        println("Error - the number of features in the new data doesn't fit the original data")
    end
    MF.X=vcat(X,X_extend)
    MF.Y=vcat(Y,Y_extend)
    Trees=MF.Trees
    Features = MF.Features
    for i=1:MF.n_trees
        T = expand!(Trees[i], X_extend[:,Features[i]],Y_extend,λ)
        Trees[i]=T
    end
     MF.Trees=Trees

end

function UpdatePosteriorCounts(j::Mondrian_Node,Y::Int64)
    j.c[Y] = j.c[Y]+1
    j_clone = j
    while true
            if j_clone.tab[Y]==1
                 return
            else
                if j_clone.node_type != [false,true,false]
                    j_clone.c[Y]=get(j_clone.left).tab[Y]+get(j_clone.right).tab[Y]
                    j_clone.tab[Y] = min(j_clone.c[Y],1)
                end
                if j_clone.node_type == [false,false,true]
                    return
                else
                    j_clone = get(j_clone.parent)
                end
            end
    end
end

#Utils for extending Mondrian Trees

"""
`function Extended_dimension(Θ::Axis_Aligned_Box,X::Array{Float64})`

This function samples the rate ∑ᵢ₌₁ᴰmax(lⱼ-x,0)+max(x-uⱼ,0) with which we draw the parameter E from the exponential distribution in the function Extend_Mondrian_Block!.

`Input`:  Axis Aligned Box Θ (Abstract type Axis_Aligned_Box), One dimensional array of features X (Array of Float64)

`Output`: Sampling rate s

`Files needed to run this function`: "Mondrian_extention.jl"

This function is one of the utility functions to run "Extend_Mondrian_Block!".


"""

function Extended_dimension(Θ::Axis_Aligned_Box,X::Array{Float64})
    # rate = ∑ᵢ₌₁ᴰmax(lⱼ-x,0)+max(x-uⱼ,0)

    s = 0
    for i in 1:Θ.D
        s += max(Θ.Intervals[i,1]-X[i],0)+max(X[i]-Θ.Intervals[i,2],0)  #compute the rate
    end
    return s
end

"""
`function sample_extended_split_dimension(Θ::Axis_Aligned_Box,X::Array{Float64})`

This function samples the split dimension and direction associated to the new node j_wave in the function "Extend_Mondrian_Block!"

`Input`: Axis Aligned Box Θ (abstract type Axis_Aligned_Box), One dimensional array of features X (Array of Float64)

`Output`: Split dimension d, split direction s

`Files needed to run this function`: "Mondrian_extention.jl"

This function is one of the utility functions to run "Extend_Mondrian_Block!".
"""

function sample_extended_split_dimension(Θ::Axis_Aligned_Box,X::Array{Float64})
    p_k = zeros(Θ.D)
    for i in 1:Θ.D
        p_k[i] = max(Θ.Intervals[i,1]-X[i],0)+max(X[i]-Θ.Intervals[i,2],0)  #compute max(lⱼ-x,0)+max(x-uⱼ,0)
    end
    p_k = p_k ./ Extended_dimension(Θ,X)                                #create the probability array
    d = rand(Categorical(p_k))                                          # sample from it (d with probability proportional to max(lⱼ-x,0)+max(x-uⱼ,0) )

    if X[d] > Θ.Intervals[d,2]  # choice of interval dependent on where the new datapoint is
    x = rand(Uniform(Θ.Intervals[d,2][1],X[d]))  #sample the split loc. uniformly from that interval
    else
    x = rand(Uniform(X[d],Θ.Intervals[d,1][1])) #sample the split loc. uniformly from that interval
    end
    return d,x   #return split dim., split loc.
end

"""
`function update_intervals(Θ::Axis_Aligned_Box,X::Array{Float64})`

This function computes the boxes associated to the new node j_wave.

`Input`: Axis Aligned Box Θ (abstract type Axis_Aligned_Box), One dimensional array of features X (Array of Float64)

`Output`: Axis Aligned Box Θ (updated Intervals)

`Files needed to run this function`: "Mondrian_extention.jl"

This function is one of the utility functions to run "Extend_Mondrian_Block!".

"""
function update_intervals(Θ::Axis_Aligned_Box,X::Array{Float64})
    Intervals = Θ.Intervals                         # extract the intervals from Θ
   for i=1:length(Θ.Intervals[:,1])                 # update the intervals successively
       l=min(Intervals[i,1],X[i])
       u=max(Intervals[i,2],X[i])
       Intervals[i,:]=[l,u]
    end
    Θ = Axis_Aligned_Box(Intervals,Θ.D)             # return updated intervals
    return Θ
end
"""
`function update_counts_extended(j_wave::Mondrian_Node,j::Mondrian_Node,Y::Int64)`

This assigns the correct number of nodes of each class to the node j_wave.

`Input`: Mondrian Node j_wave (abstract type Mondrian_Node, node that is added above j), Mondrian Node j (abstract type Mondrian_Node, node the function currently considers), Int64 Y (Label of new datapoint)

`Output`: updated counts j_wave.c, updated j_wave.tab

This function is one of the utility functions to run "Extend_Mondrian_Block!".
"""

function update_counts_extended(j_wave::Mondrian_Node,j::Mondrian_Node,Y::Int64)
    j_wave.tab = copy(j.tab)
    j_wave.c = copy(j.c)
    j_wave.Gₚ = zeros(size(j.Gₚ))
    if j_wave.tab[Y] == 0
    j_wave.tab[Y] = 1
    end
    j_wave.c[Y] = j_wave.c[Y]+1
end

"""
`function backpropergate_c_tab(j_wave::Mondrian_Node,Y::Int64)`

This function backpropergates through the tree and updates the counts of each class at each node if a new datapoint was inserted.

`Input`: Mondrian Node j_wave (abstract type Mondrian_Node, node that is added above j), Int64 Y (label of the new datapoint)

`Output`: updated counts for all nodes that are on the same branch as j_wave and above j_wave


This function is one of the utility functions to run "Extend_Mondrian_Block!".
"""
function backpropergate_c_tab(j_wave::Mondrian_Node,Y::Int64)
    if j_wave.node_type != [false, false, true]
   get(j_wave.parent).c[Y] =  get(j_wave.parent).c[Y]+1
    if get(j_wave.parent).tab[Y] == 0
    get(j_wave.parent).tab[Y] = 1
    end
    if get(j_wave.parent).node_type != [false,false,true]
        backpropergate_c_tab(get(j_wave.parent),Y)
    end
    end
end
"""
function paused_train_data(j::Mondrian_Node,X::Array{Float64},Y::Int64)

This function assembles the right dataset to sample the tree from given that j is a paused leaf.

`Input`: Mondrian Node j (abstract type Mondrian Node, paused leaf), datapoint X,Y

`Output`: p_data (training data), p_labels (training labels) for paused leaf (merge of data that was there before+new DP)

This function is one of the utility functions to run "Extend_Mondrian_Block!".
"""

function paused_train_data(j::Mondrian_Node,X::Array{Float64},Y::Int64)
   Data = j.Paused_Data[1]
   Labels = j.Paused_Data[2]
   A=zeros(1,length(X))
   A[:,:]=X
   p_data = vcat(Data,A)
   p_labels = vcat(Labels, Y)
   return p_data, p_labels

end

"""
function init_j_wave(j::Mondrian_Node,E::Float64,d::Int64,x::Float64,Θ::Axis_Aligned_Box)

This function initialises the new node j_wave that is inserted in the tree just above j.

`Input`: Mondrian Node j (abstract type Mondrian Node, node the algoritm currently looks at), E (Float 64, value sampled in the algorithm), d (Int64, split dimension), x (Float64, split direction), Θ (Axis_Aligned_Box, 'Box' associated with j_wave)

`Output`: New node j_wave in tree, above j

This function is one of the utility functions to run "Extend_Mondrian_Block!".

"""

function init_j_wave(T::Mondrian_Tree,j::Mondrian_Node,E::Float64,d::Int64,x::Float64,Θ::Axis_Aligned_Box)
   if j.node_type[3]==true                              # check if we replace the root
            j_wave=Mondrian_Node(E,[false,false,true])       # replace the root by the new node j_wave
            j.node_type = [true,false,false]
            T.root = j_wave
        else
            j_wave=Mondrian_Node(get(j.parent).τ+E,[true,false,false])  #if we don't replace the root, introcue a new node j_wave, parent to j
            j_wave.parent = j.parent
            if j == get(j.parent).left             # check if j was left or right child of j_parent
                get(j.parent).left = j_wave
            else
                get(j.parent).right = j_wave
            end
        end
        j_wave.δ = d
        j_wave.ζ = x
        j_wave.Θ = Θ
        j_wave.tab = copy(j.tab)
    j_wave.c = copy(j.c)
    j_wave.Gₚ = zeros(size(j.Gₚ))
    return j_wave

end

"""
`function init_j_prime(j_wave::Mondrian_Node,Y::Int64)`

This function initialises a new node j_prime, child of j_wave and sibling to j

`Input`: Mondrian Node j_wave (Abstract type Mondrian Node, parent to the node j_prime), Y (Int64, label of the new datapoint)

`Output`: New node j_prime, child of j_wave, sibling to j, leaf that only has the new datapoint

This function is one of the utility functions to run "Extend_Mondrian_Block!".

"""
function init_j_prime(j_wave::Mondrian_Node,Y::Int64)
   j_prime = Mondrian_Node(0.0, [false,true,false])   #initialise new sibling to j
        j_prime.parent = j_wave
        j_prime.tab = zeros(size(j_wave.tab))
        j_prime.c = zeros(size(j_wave.c))
        j_prime.c[Y] = 1
        j_prime.Gₚ=zeros(size(j_wave.Gₚ))
    return j_prime
end
"""
`function check_left_right!(j_wave::Mondrian_Node,j::Mondrian_Node,j_prime::Mondrian_Node,X::Array{Float64},d::Int64,x::Int64)`

This function figures out whether j_prime should be the left or the right child of j_wave. It is always chosen such that j_prime contains the new datapoint (dependent on the split). It also makes sure all the data is in the right boxes.

`Input`: Mondrian Node j_wave (Abstract type Mondrian Node, parent j and j_prime), Mondrian Node j (Abstract type Mondrian Node, child to j_wave, sibling to j_prime), Mondrian Node j_prime (Abstract type Mondrian Node, child to j_wave, sibling to j), X (Array of Float64, features of the new datapoint), d (Int64, split dimension), x (Float64, split direction)

`Output`: A (transformed feature data for next function), j_wave (updated children), j_prime (updated AxisAligned Box)


This function is one of the utility functions to run "Extend_Mondrian_Block!".

"""

function check_left_right!(j_wave::Mondrian_Node,j::Mondrian_Node,j_prime::Mondrian_Node,X::Array{Float64},d::Int64,x::Float64)
            A=zeros(1,length(X))
                A[:,:]=X
            if X[d] > x     # check where the new datapoint lies
                j_wave.left = j
                j_wave.right = j_prime
                    if get(j.Θ).Intervals[d,2]> x
                        get(j.Θ).Intervals[d,2] = x  #adapt box of j according to new split
                    end
                j_prime.Θ = j_wave.Θ
                get(j_wave.Θ).Intervals[d,1] = x

            else
                j_wave.left = j_prime
                j_wave.right = j
                if get(j.Θ).Intervals[d,1]< x
                    get(j.Θ).Intervals[d,1]= x      #adapt box of j according to new split
                end
                j_prime.Θ = j_wave.Θ
                get(j_wave.Θ).Intervals[d,2] = x
            end
    return A, j_wave, j_prime
end



function j_print(j)
   if get(j).node_type != [false, true, false]
        println(get(j).c)
        println(get(get(j).left).c)
        println(get(get(j).right).c)
        println("node done")
        j_print(get(j).left)
        j_print(get(j).right)
    end

end


function leaf_check(j)
   if get(j).node_type == [false, true, false]
        println(get(j).c)
    else
        leaf_check(get(j).left)
        leaf_check(get(j).right)
    end
end
