function makeMondrianForest(learner::Learner, task::Task, data)
    parameters = []
    possible_names = ["n_trees"]
    possible_parameters = Dict(
        "n_trees" => Integer
    )
    for (i, (name,value)) in enumerate(learner.parameters)
        if possible_names[i] == name
            if typeof(learner.parameters[name]) <: possible_parameters[name]
                push!(parameters, learner.parameters[name])
            end
        end
    end
    MF=Mondrian_Forest_Classifier()
    MLRModel(MF, parameters, inplace=false)
end
function learnᵧ(modelᵧ::MLRModel{<:Mondrian_Forest_Classifier};
                learner=nothing::Learner,
                data=nothing::Matrix{Real},
                task=nothing::Task)
    train = data[:,task.features]
    target = convert(Array{Int64,1},data[:,task.targets[1]])
    MF = Mondrian_Forest_Classifier(modelᵧ.parameters...)
    train!(MF, train, target)
    MLRModel(MF, modelᵧ.parameters)
end

function predictᵧ(modelᵧ::MLRModel{<:Mondrian_Forest_Classifier};
                    data_features=nothing::Matrix, task=nothing::Task)
    preds = predict!(modelᵧ.model, data_features)
    ## TODO get predict_proba! working for mondrian forest and tree
    # println(probs)
    # preds = [p>0.5?1:0 for p in probs]
    preds, probs
end
