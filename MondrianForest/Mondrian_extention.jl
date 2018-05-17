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

## Algorithm 4 in the Paper "Mondrian Forests: Efficient Online Random Forests"
function Extend_Mondrian_Block!(T::Mondrian_Tree,λ::Float64,j::Mondrian_Node,X::Array{Float64},Y::Int64)
    
    E = rand(Exponential(1/Extended_dimension(get(j.Θ),X)))  #sample value E
    if j.node_type[3]==true                                  # check if the node we're looking at is the root (if yes the split time is assumed to be 0)
        τₚ = 0
    else
        τₚ = (get(j.parent)).τ                               # if it's not the root get the split time of the node above j
    end
    if τₚ + E < j.τ                                          # check if our split happens in time
        d,x= sample_extended_split_dimension(get(j.Θ),X)     # sample new split dimension / split direction
        Θ = update_intervals(get(j.Θ),X)                     # get the boxes for the new node
        if j.node_type[3]==true                              # check if we replace the root
            j_wave=Mondrian_Node(E,[true,false,false])       # replace the root by the new node j_wave
	    j_wave.δ = d
        j_wave.ζ = x
        j_wave.Θ = Θ
        j_wave.tab = zeros(size(get(j.tab)))
        j_wave.c = zeros(size(get(j.c)))
        j_wave.Gₚ = zeros(size(get(j.Gₚ)))
        else
            j_wave=Mondrian_Node(get(j.parent).τ+E,[true,false,false])  #if we don't replace the root, introcue a new node j_wave, parent to j
	    j_wave.parent = j.parent
	    j_wave.δ = d
        j_wave.ζ = x
        j_wave.Θ = Θ
        j_wave.tab = get(j.parent).tab
        j_wave.c = get(j.parent).c
        j_wave.Gₚ = get(j.parent).Gₚ
            if j == get(j.parent).left             # check if j was left or right child of j_parent
                get(j.parent).left = j_wave
            else
                get(j.parent).right = j_wave
            end
        end

        j.parent=j_wave
        j_prime = Mondrian_Node(0.0, [true,false,false])   #initialise new sibling to j
        j_prime.parent = j_wave
        j_prime.tab = zeros(size(j_wave.tab))
        j_prime.c = zeros(size(j_wave.tab))
        j_prime.Gₚ=zeros(size(j_wave.c,1))
        
        if X[d] > x     # check where the new datapoint lies
            j_wave.left = j
            j_wave.right = j_prime 
                if get(j.Θ).Intervals[d,2]> x    
                    get(j.Θ).Intervals[d,2] = x  #adapt box of j according to new split
                end
            j_prime.Θ = j_wave.Θ
            get(j_wave.Θ).Intervals[d,1] = x
            A=zeros(1,length(X))
            A[:,:]=X
            Sample_Mondrian_Block!(j_prime,get(j_prime.Θ),λ,T,A,[Y])  #sample a mondrian block at the node whose associated boxes contain the new datapoint
        else
            j_wave.left = j_prime
            j_wave.right = j
            if get(j.Θ).Intervals[d,1]< x 
                get(j.Θ).Intervals[d,1]= x      #adapt box of j according to new split
            end
            j_prime.Θ = j_wave.Θ
            get(j_wave.Θ).Intervals[d,2] = x
            A=zeros(1,length(X))
            A[:,:]=X
            Sample_Mondrian_Block!(j_prime,get(j_prime.Θ),λ,T,A,[Y])  #sample a mondrian block at the node whose associated boxes contain the new datapoint
        end


        else                                    # if the split didn't occur in time
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

"""
`function expand!(T::Mondrian_Tree,X::Array{Float64,N} where N,Y::Array{Int64},λ::Float64)`

This function expands an already sampled Mondrian Tree by a desired number of datapoints. 

`Input`: Mondrian Tree T (abstract type Mondrian_Tree), array of features X (Array of Float64), array of class labels (1dim of Float 64), Lifetime parameter λ (Float 64)

Each row in the array X represents one set of features, the corresponding row in Y represents the class label. 

`Output`: Mondrian Tree with incoporated new datapoints

`Files needed to run this function`: Mondrian_Forest_Classifier.jl", "Mondrian_extention.jl"

This function calls the function Extend_Mondrian_Tree. 
"""

function expand!(T::Mondrian_Tree,X::Array{Float64,N} where N,Y::Array{Int64},λ::Float64)    
    
    # puts the extention in a nice framework, allows to extend by multiple datapoints
    
    for i=1:length(X[:,1])
        T=Extend_Mondrian_Tree!(T,λ,X[i,:],Y[i]);
    end
    return T    
end
