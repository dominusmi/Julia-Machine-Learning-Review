include("Mondrian_Tree_Regression.jl")

## same as classifier by for regression
mutable struct Mondrian_Tree_Regressor
    Tree::Mondrian_Tree
    X::AbstractArray{Float64, N} where N
    Y::AbstractArray{Float64, N} where N
end

function Mondrian_Tree_Regressor()
    return Mondrian_Tree_Regressor(Mondrian_Tree(),[],[])
end

function Mondrian_Tree_Regressor{X<:AbstractArray{Float64,N} where N,
                                 Y<:AbstractArray{Float64,N} where N}(
                                 Tree::Mondrian_Tree,
                                 Data::X,
                                 Labels::Y)
    return Mondrian_Tree_Regressor(Tree,Data,Labels)
end

mutable struct Mondrian_Forest_Regressor
    n_trees::Int64                          # number of trees in the forest
    Trees::Array{Mondrian_Tree_Regressor,1}
    X::AbstractArray{Float64,N} where N
    Y::AbstractArray{Float64,N} where N
end

function Mondrian_Forest_Regressor(n_trees::Int64=10)
    MF = Mondrian_Forest_Regressor(n_trees,
                                    Array{Mondrian_Tree_Regressor,1}(),
                                    [],
                                    [])
    MF.Trees = Array{Mondrian_Tree_Regressor,1}()
    return MF
end

function train!{X<:AbstractArray{Float64,N} where N,
                Y<:AbstractArray{Float64,N} where N}(
                MT::Mondrian_Tree_Regressor,
                Data::X,
                Labels::Y,
                λ=1e9,
                min_samples_split=10)
    Sample_Mondrian_Tree!(MT.Tree,λ,Data,Labels,min_samples_split)
    MT.X = Data
    MT.Y = Labels
end

function predict!(MT::Mondrian_Tree_Regressor,
                  X)
    return predict_reg_batch(MT.Tree,X)
end

function train!{X<:AbstractArray{Float64,N} where N,
                Y<:AbstractArray{Float64,N} where N}(
                MF::Mondrian_Forest_Regressor,
                Data::X,
                Labels::Y,
                λ::Float64=1e9,
                min_samples_split=10)
    @parallel for i in 1:MF.n_trees
        push!(MF.Trees,Mondrian_Tree_Regressor())
        train!(MF.Trees[end], Data, Labels, λ, min_samples_split)
    end
    MF.X = Data
    MF.Y = Labels
end

function predict!{X<:AbstractArray{Float64,N} where N,}(
                  MF::Mondrian_Forest_Regressor,
                  Data::X)
    pred = zeros(MF.n_trees,size(Data,1))
    # if this print is not here
    # the regressor predicts
    # all zeros !!!
    print("")
    for item in enumerate(MF.Trees)
        pred[item[1],:] = predict!(item[2], Data)
    end
    p = []
    for i in 1:size(Data,1)
        push!(p, mean(pred[:,i]))
    end
    return p
end
