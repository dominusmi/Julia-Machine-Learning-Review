include("Mondrian_Tree.jl")
include("Mondrian_extention.jl")
# for nice repl output
using Base.show

### Mondrian Tree Classifier Definitions

mutable struct Mondrian_Tree_Classifier
    Tree::Mondrian_Tree
    X::AbstractArray{Float64,N} where N             # training data
    Y::AbstractArray{Int64}                           # training labels
end

"""
`function Mondrian_Tree_Classifier()`

Initialise an empty mondrian tree classifier
"""
function Mondrian_Tree_Classifier()
    return Mondrian_Tree_Classifier(Mondrian_Tree(),[],[])
end

"""
`function Mondrian_Tree_Classifier()`

Initialise a mondrian tree classifier with a given mondrian tree Tree.
"""
function Mondrian_Tree_Classifier(Tree::Mondrian_Tree)
    return Mondrian_Tree_Classifier(Tree,[],[])
end

"""
`function Mondrian_Tree_Classifier()`

Initialise a mondrian tree classifier with a given tree Tree, data X, and labels Y.
"""
function Mondrian_Tree_Classifier{X<:AbstractArray{Float64,N} where N,
                                  Y<:AbstractArray{Int64, N} where N}(
                                  Tree::Mondrian_Tree,
                                  Data::X,
                                  Labels::Y)
    return Mondrian_Tree_Classifier(Tree,Data,Labels)
end

function Base.show(io::IO, MT::Mondrian_Tree_Classifier)
    print("Mondrian Tree Classifier:\n")
    print("    Mondrian Tree with ", length(MT.Tree.leaves), " leaves")
end

### Mondrian Forest Classifier Definitions

mutable struct Mondrian_Forest_Classifier   # currently trees in the forest just use default params same as in the paper
    n_trees::Int64                          # number of trees in the forest
    Trees::Array{Mondrian_Tree_Classifier,1}
    X::AbstractArray{Float64,N} where N
    Y::AbstractArray{Int64}
end

"""
`function Mondrian_Forest_Classifier(n_trees::Int64=10)`

Initialise an empty mondrian forest classifier with default 10 trees
"""
function Mondrian_Forest_Classifier(n_trees::Int64=10)
    MF = Mondrian_Forest_Classifier(n_trees,
                                    Array{Mondrian_Tree_Classifier,1}(),
                                    [],
                                    [])
    MF.Trees = Array{Mondrian_Tree_Classifier,1}()
    return MF
end

function Base.show(io::IO, MF::Mondrian_Forest_Classifier)
    print("Mondrian Forest Classifier with ",MF.n_trees," Mondrian trees:\n")
    for tree in MF.Trees
        print("    Mondrian Tree with ", length(tree.Tree.leaves), " leaves\n")
    end
end

### Mondrian Tree Training and Prediction

"""
`function train!(MT::Mondrian_Tree_Classifier,
                X::AbstractArray{Float64,N} where N,
                Y::AbstractArray{Int64},
                λ=1e9)`

Trains (samples) a mondrian tree on data X with labels Y and a lifetime parameter
λ (cross-validate this).
"""
function train!{X <:AbstractArray{Float64,N} where N,
                Y <:AbstractArray{Int64, 1},}(
                MT::Mondrian_Tree_Classifier,
                Data::X,
                Labels::Y,
                λ::Float64=1e9)
    Sample_Mondrian_Tree!(MT.Tree,λ,Data,Labels)
    compute_predictive_posterior_distribution!(MT.Tree,10*size(Data,2))   # TODO get rid of this override
    MT.X = Data
    MT.Y = Labels
    return MT
end

"""
`function predict!(MT::Mondrian_Tree_Classifier,
                  X::AbstractArray{Float64,N} where N)`

Predict the classes for the new data instances X with a trained tree
classifier.
"""
function predict!{X <: AbstractArray{Float64,N} where N}(
                  MT::Mondrian_Tree_Classifier,      # batch prediction NB supposedly can change tree structure!
                  Data::X)
    pred = []
    for i in 1:size(Data,1)
        p = predict!(MT.Tree, Data[i,:], 10*size(Data,2))
        if p[1] > p[2]
            push!(pred,1)
        else
            push!(pred,2)
        end
    end
    return pred
end


"""
`function predict_proba!(MT::Mondrian_Tree_Classifier,
                        X::AbstractArray{Float64,2})`

Return the probabilities for predicting each class for the new data instances
X.
"""
function predict_proba!{X <: AbstractArray{Float64,N} where N}(
                        MT::Mondrian_Tree_Classifier,      # batch prediction NB supposedly can change tree structure!
                        Data::X)
    pred = []
    for i in 1:size(Data,1)
        push!(pred,predict!(MT.Tree,Data[i,:],10*size(Data,2)))
    end
    return pred
end

## Mondrian Forest Training and Prediction
"""
`function train!(MF::Mondrian_Forest_Classifier,
                X::AbstractArray{Float64,2},
                Y::AbstractArray{Int64},
                λ::AbstractFloat=1e9)`

Trains (samples) a mondrian forest on the data X with labels Y.
"""
function train!{X<:AbstractArray{Float64, N} where N,
                Y<:AbstractArray{Int64, N} where N}(
                MF::Mondrian_Forest_Classifier,
                Data::X,
                Labels::Y,
                λ::Float64=1e9)
    @parallel for i in 1:MF.n_trees
        push!(MF.Trees,Mondrian_Tree_Classifier())
        train!(MF.Trees[end], Data, Labels, λ)
    end
    MF.X = Data
    MF.Y = Labels
    return MF
end

"""
`function predict!(MF::Mondrian_Forest_Classifier,
                  X::AbstractArray{Float64,2})`

Predicts the classes of the new data instances X.
"""
function predict!{X<:AbstractArray{Float64,N} where N}(
                  MF::Mondrian_Forest_Classifier,
                  Data::X)
    pred = zeros(MF.n_trees,size(Data,1))
    for item in enumerate(MF.Trees)
        pred[item[1],:] = predict!(item[2], Data)
    end
    p = Array{Int64,1}(size(Data,1))
    for i in 1:size(Data,1)
        p[i] = mode(pred[:,i])
    end
    return p
end

"""
`function predict_proba!(MF::Mondrian_Forest_Classifier,
                        X::AbstractArray{Float64,2})`

Returns the probabilities for each class of the new data instances X.
"""
function predict_proba!{X<:AbstractArray{Float64,N} where N}(
                        MF::Mondrian_Forest_Classifier,
                        Data::X)
    pred = []
    for i in 1:size(Data,1)
        push!(pred,[0.0,0.0])
    end
    for trees in enumerate(MF.Trees)
        p=predict_proba!(trees[2], Data)
        for item in enumerate(p)
            pred[item[1]] += item[2]
        end
    end
    return pred/MF.n_trees
end

## For testing

function FakedataClassif(N,d,N_test=0)
    X = randn((N,d))
    param1 = randn(d)
    param2 = randn(d)
    Y = ( sum(X*param1,2) .> mean(sum(X*param2,2)) )
    if (N_test > 0)
        x = randn((N_test,d))
        y = ( sum(x*param1,2) .> mean(sum(x*param2,2)) )
        return X,Y,x,y
    end
    return X,Y
end

## print the tree with text

"""
`function print_mondrian_tree(node::Mondrian_Node, depth=-1, indent=0)`

Prints a mondrian tree like DecisionTree.jl [1] input node as the root
node *MT.root* of a mondrian tree MT (Mondrian_Tree_Classifier.Tree).

[1] https://github.com/bensadeghi/DecisionTree.jl/blob/5f0adc5d6d0280f995ccc364e5d00a72f6387368/src/DecisionTree.jl#L89
"""
function print_mondrian_tree(node::Mondrian_Node, depth=-1, indent=0)
    ## Adapted function from DecisionTree.jl
    if (node.node_type != [0,0,1]) & (indent==0)
        println("Not starting from root node..")
    end
    if (node.node_type == [0,1,0])
        println("Prediction: ", round.((node.Gₚ),3))
        return
    end
    println("τ: ",round((node.τ),3))
    print("    " ^ indent * "L-> ")
    print_mondrian_tree(get(node.left), depth, indent + 1)
    print("    " ^ indent * "R-> ")
    print_mondrian_tree(get(node.right), depth, indent + 1)
end


"""
`function show_mondrian_split_2d(MT::Mondrian_Tree_Classifier;
                                times=false,
                                Data=false,
                                animation=false)`
For a two dimensional problem prints the splits of a trained Mondrian_Tree_Classifier

Returns the plotted figure and an Animation object (1 frame if `animation=false`)

-`time=true`        will label each split with the split time (messy)
-`Data=true`        will plot the data points and classes
-`animation=true`   will return an Animation object of the plotting
"""
function show_mondrian_split_2d(MT::Mondrian_Tree_Classifier;
                                times=false,
                                Data=false,
                                animation=false)

    fig = plot()
    τ = []
    ζ = []
    δ = []
    nodes = []
    push!(τ,get(MT.Tree.root).τ)
    push!(ζ,get(get(MT.Tree.root).ζ))
    push!(δ,get(get(MT.Tree.root).δ))
    push!(nodes, get(MT.Tree.root))
    for l in MT.Tree.leaves
        j = get(l.parent)
        while j.node_type[3] != true
            if (j in nodes)
                break
            end
            push!(τ, j.τ)
            push!(ζ, get(j.ζ))
            push!(δ, get(j.δ))
            push!(nodes,j)
            j = get(j.parent)
        end
    end
    indices = sortperm(τ)
    τ = τ[indices]
    ζ = ζ[indices]
    δ = δ[indices]
    nodes = nodes[indices]

    anim = Animation()
    if animation
        frame_numbers = collect(1:size(ζ,1))
    else
        frame_numbers = size(ζ)
    end
    for i in 1:size(ζ,1)
        if (i==1)
            if (Data)
                for c in unique(MT.Y)
                    scatter!(fig,MT.X[MT.Y.==c,1],MT.X[MT.Y.==c,2], label="Class $c")
                end
            end
            int = get(nodes[1].Θ).Intervals
            xlims!(fig,int[1,1],int[1,2])
            ylims!(fig,int[2,1],int[2,2])
            title!(fig,"Mondrian Tree Partitions")
        end
        int = get(nodes[i].Θ).Intervals
        if times
            label = round(τ[i],3)
        else
            label = ""
        end
        if (δ[i] == 1)
            x = linspace(int[2,1],int[2,2],20)
            plot!(fig,fill(ζ[i],length(x)),x, label="")
            annotate!([(ζ[i], median(x), text(label,6))])
        else
            x = linspace(int[1,1],int[1,2],20)
            plot!(fig,x,fill(ζ[i],length(x)), label="")
            annotate!([(median(x),ζ[i], text(label,6))])
        end
        if i in frame_numbers
            frame(anim)
        end
    end
    return fig,anim
end
