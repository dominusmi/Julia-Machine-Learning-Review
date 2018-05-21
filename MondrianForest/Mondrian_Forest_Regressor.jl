include("Mondrian_Tree_Regression.jl")

## same as classifier by for regression
mutable struct Mondrian_Tree_Regressor
    Tree::Mondrian_Tree
    X::Array{Float64, N} where N
    Y::Array{Float64, N} where N
end

function Mondrian_Tree_Regressor()
    return Mondrian_Tree_Regressor(Mondrian_Tree(),[],[])
end

function Mondrian_Tree_Regressor(Tree::Mondrian_Tree,
                                 X::Array{Float64, N} where N,
                                 Y::Array{Float64, N} where N)
    return Mondrian_Tree_Regressor(Tree,X,Y)
end

mutable struct Mondrian_Forest_Regressor
    n_trees::Int64                          # number of trees in the forest
    Trees::Array{Mondrian_Tree_Regressor,1}
    X::Array{Float64,N} where N
    Y::Array{Float64, N} where N
end

function Mondrian_Forest_Regressor(n_trees::Int64=10)
    MF = Mondrian_Forest_Regressor(n_trees,
                                    Array{Mondrian_Tree_Regressor,1}(),
                                    [],
                                    [])
    MF.Trees = Array{Mondrian_Tree_Regressor,1}()
    return MF
end

function train!(MT::Mondrian_Tree_Regressor,
                X::Array{Float64,N} where N,
                Y::Array{Float64, N} where N,
                λ=1e9,
                min_samples_split=10)
    Sample_Mondrian_Tree!(MT.Tree,λ,X,Y,min_samples_split)
    MT.X = X
    MT.Y = Y
end

function predict!(MT::Mondrian_Tree_Regressor,
                  X)
    return predict_reg_batch(MT.Tree,X)
end

function train!(MF::Mondrian_Forest_Regressor,
                X::Array{Float64, N} where N,
                Y::Array{Float64, N} where N,
                λ::Float64=1e9,
                min_samples_split=10)
    @parallel for i in 1:MF.n_trees
        push!(MF.Trees,Mondrian_Tree_Regressor())
        train!(MF.Trees[end], X, Y, λ, min_samples_split)
    end
    MF.X = X
    MF.Y = Y
end

function predict!(MF::Mondrian_Forest_Regressor,
                  X::Array{Float64,N} where N)
    pred = zeros(MF.n_trees,size(X,1))
    # if this print is not here
    # the regressor predicts
    # all zeros !!!
    print("")
    for item in enumerate(MF.Trees)
        pred[item[1],:] = predict!(item[2], X)
    end
    p = []
    for i in 1:size(X,1)
        push!(p, mean(pred[:,i]))
    end
    return p
end
