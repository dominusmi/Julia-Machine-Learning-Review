function makeMondrianForest(learner::ModelLearner, task::Task)
    parameters = []
    possible_names = ["n_trees"]
    possible_parameters = Dict(
        "n_trees" => Integer,
        "λ" => Float64
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
function learnᵧ(modelᵧ::MLRModel{<:Mondrian_Forest_Classifier},
                learner::Learner,
                task::Task)
    train = data[:,task.features]
    target = data[:,task.targets[1]]
    MF = Mondrian_Forest_Classifier(modelᵧ.parameters...)
    train!(MF, train, convert(Array{Int64},target))
    MLRModel(MF, modelᵧ.parameters)
end

function predictᵧ(modelᵧ::MLRModel{<:Mondrian_Forest_Classifier},
                    data_features, task::Task)
    # required otherwise predictᵧ returns 0,0 probs!    
    println()
    probs = predict_proba!(modelᵧ.model, data_features)
    preds = zeros(size(probs,1))
    for i in 1:size(probs,1)
        preds[i] = indmax(probs[i])
    end
    preds, probs
end
