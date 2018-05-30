include("Mondrian_extention_utils.jl");
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
function Extend_Mondrian_Tree!(T::Mondrian_Tree,λ::Float64,X::Array{Float64},Y::Int64) 
    ϵ=get(T.root)
    Extend_Mondrian_Block!(T,λ,ϵ,X,Y)
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
function Extend_Mondrian_Block!(T::Mondrian_Tree,λ::Float64,j::Mondrian_Node,X::Array{Float64},Y::Int64)
    
if sum(j.c .> 0) == 1  #check if all labels are identical
     Θ = update_intervals(get(j.Θ),X)        # update extent
     j.Θ=Θ 
         if findmax(j.c)[2] == Y
             i = findmax(j.c)[2]
             j.c[i] = j.c[i]+1
             backpropergate_c_tab(j,Y)
             return
         else
            j.node_type = [true,false,false]
            p_data,p_labels = paused_train_data(j,X,Y)
            Sample_Mondrian_Block!(j,get(j.Θ),λ,T,p_data,p_labels)
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
        j_wave = init_j_wave(j,E,d,x,Θ) 
        update_counts_extended(j_wave,j,Y)
        j.parent=j_wave
        backpropergate_c_tab(j_wave,Y) 
            
        j_prime = init_j_prime(j_wave,Y)
        A,j_wave,j_prime = check_left_right!(j_wave,j,j_prime,X,d,x)
        Sample_Mondrian_Block!(j_prime,get(j_prime.Θ),λ,T,A,[Y])


    else                                 # if the split didn't occur in time
        Θ = update_intervals(get(j.Θ),X)        # update the boxes of j
        j.Θ=Θ
        if j.node_type != [false,true,false]    # check if j is a leaf
            if X[get(j.δ)] < get(j.ζ)           # if the new datapoint is in the boxes associated with the left child of j -> Extend towards the left child, else the right
                     
                Extend_Mondrian_Block!(T,λ,get(j.left),X,Y)
            else
                    
                Extend_Mondrian_Block!(T,λ,get(j.right),X,Y)
                
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
        T.Tree=Extend_Mondrian_Tree!(T.Tree,λ,X[i,:],Y[i]);
    end
    T.X = vcat(T.X,X)
    T.Y = vcat(T.Y,Y)
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
    Trees=MF.Trees
    Features = MF.Features
    @parallel for i=1:MF.n_trees
        T = expand!(Trees[i], X_extend[:,Feature[i]],Y_extend,λ)
        Trees[i]=T
    end
     MF.Trees=Trees
     MF.X=vcat(X,X_extend)
     MF.Y=vcat(Y,Y_extend)
    
end
