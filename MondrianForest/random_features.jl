function train!{X<:Array{Float64, N} where N,
                Y<:Array{<: Integer,N} where N}(
                MF::Mondrian_Forest_Classifier,
                Data::X,
                Labels::Y,
                λ::Float64=1e9,
                random_features=ceil(size(Data,2)))
    @parallel for i in 1:MF.n_trees
        features = randperm(size(Data,2))[1:random_features]
        push!(MF.Trees,Mondrian_Tree_Classifier())
        push!(MF.Features,features)
        train!(MF.Trees[end], Data[:,features], Labels, λ)
    end
    MF.X = Data
    MF.Y = Labels
    return MF
end

function predict!{X<:Array{<: AbstractFloat,N} where N}(
                  MF::Mondrian_Forest_Classifier,
                  Data::X)
    pred=zeros(MF.n_trees,size(Data,1))
    println("")
    for i in 1:MF.n_trees
        pred[i,:] = predict!(MF.Trees[i], Data[:,MF.Features[i]])
    end
    p = zeros(size(Data,1))
    for i in 1:size(Data,1)
        p[i] = mode(pred[:,i])
    end
    return p
end
