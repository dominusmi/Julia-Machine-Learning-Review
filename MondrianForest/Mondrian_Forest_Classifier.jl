include("Mondrian_Tree.jl")

mutable struct Mondrian_Tree_Classifier
    Tree::Mondrian_Tree
    λ::Float64                              # lifetime parameter set to inf in paper and 1e9 in implementations (e.g pythons)
    γ::Real                                 # Hierachy of normailized stable processes discount parameter 10*Dimensionality of data in the paper
    X::Array{Float64,2}                     # training data
    Y::Array{Int}                           # training labels
end

function Mondrian_Tree_Classifier(Tree::Mondrian_Tree)
    return Mondrian_Tree_Classifier(Tree,1e9,0,[],[])
end

function Mondrian_Tree_Classifier(Tree::Mondrian_Tree,λ::Float64)
    return Mondrian_Tree_Classifier(Tree,λ,0,[],[])
end

function Mondrian_Tree_Classifier(Tree::Mondrian_Tree,
                                  λ::Float64,
                                  X::Array{Float64,2},
                                  Y::Array{Int64})
    return Mondrian_Tree_Classifier(Tree,λ,0,X,Y)
end

mutable struct Mondrian_Forest_Classifier   # currently trees in the forest just use default params same as in the paper
    n_trees::Int64                          # number of trees in the forest
    Trees::Array{Mondrian_Tree}
    X::Array{Float64,2}
    Y::Array{Int}
end

function Mondrian_Forest_Classifier(n_trees::Int64)
    return Mondrian_Forest_Classifier(n_trees,
                                    Array{Mondrian_Tree,1}(),
                                    Array{Float64,2}(0,0),
                                    Array{Int}(0))
end

function train!(Tree::Mondrian_Tree,
                X::Array{Float64,2},
                Y::Array{Int64},λ=1e9)
    Sample_Mondrian_Tree!(Tree,λ,X,Y)
    compute_predictive_posterior_distribution!(Tree,10*size(X,2))   # TODO get rid of this override
end

function predict!(Tree::Mondrian_Tree,      # batch prediction NB supposedly can change tree structure!
                  X::Array{Float64,2})
    pred = []
    for i in 1:size(X,1)
        p = predict!(Tree,X[i,:],10*size(X,2))
        if p[1] > p[2]
            push!(pred,1)
        else
            push!(pred,2)
        end
    end
    return pred
end

function train!(MF::Mondrian_Forest_Classifier,
                X::Array{Float64,2},
                Y::Array{Int64},
                λ::Real)
    for i in 1:MF.n_trees
        Tree = Mondrian_Tree()
        train!(Tree, X, Y, λ)
        push!(MF.Trees,Tree)
    end
end

function predict!(MF::Mondrian_Forest_Classifier,
                  X::Array{Float64,2})
    pred = zeros(MF.n_trees,size(X,1))
    for i in 1:length(MF.Trees)
        pred[i,:] = predict!(MF.Trees[i], X)
    end
    p = Array{Int,1}(size(X,1))
    for i in 1:size(X,1)
        p[i] = mode(pred[:,i])
    end
    return p
end
