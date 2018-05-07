using DecisionTree



function makeDecisiontree(learner::Learner, task::Task, data)
    parameters = []
    possible_names = ["maxlabels", "nsubfeatures", "maxdepth"]
    possible_parameters = Dict(
        "maxlabels"=>Integer,
        "nsubfeatures"=>Integer,
        "maxdepth"=>Integer
    )

    for (i, (name, value)) in enumerate(learner.parameters)
        if possible_names[i] == name
            if typeof(learner.parameters[name]) <: possible_parameters[name]
                push!(parameters, learner.parameters[name])
            end
        else
            if i !== length(Learner.parameters)
                warn("DT requires that you provide maxlabels to be to set nsubfeatures"*
                      "and that you provide nsubfeatures to be able to set maxdepth."*
                      "parameter $(name) was therefore not set")
            end
        end
    end


    node = Node(0, nothing, Leaf(nothing,[nothing]), Leaf(nothing,[nothing]))
    MLRModel(node, parameters, inplace=false)
end


immutable DecisionForestᵧ end

function makeForest(lrn::Learner, task::Task, data)
    lprms = copy(lrn.parameters)

    parameters = []

    possible_names = ["maxlabels", "partialsampling", "maxdepth"]
    possible_parameters = Dict(
        "maxlabels"=>Integer,
        "partialsampling"=>Float64,
        "maxdepth"=>Integer
    )

    if get(lprms, "nsubfeatures", false ) == false || get(lprms, "ntrees", false ) == false
        throw("nsubfeatures and ntrees must be set")
    end

    push!(parameters, lprms["nsubfeatures"], lprms["ntrees"])
    delete!(lprms, "nsubfeatures")
    delete!(lprms, "ntrees")


    for (i, (name, value)) in enumerate(lprms)
        if possible_names[i] == name
            if typeof(lprms) <: possible_parameters[name]
                push!(parameters, lrn.parameters[name])
            end
        else
            if i !== length(lprms)
                warn("DT requires that you provide maxlabels to be to set partialsampling"*
                      "and that you provide nsubfeatures to be able to set maxdepth."*
                      "parameter $(name) was therefore not set")
            end
        end
    end
    MLRModel(DecisionForestᵧ(), parameters, inplace=false)
end



function learnᵧ(modelᵧ::MLRModel{<:Node}; learner=nothing::Learner, data=nothing::Matrix{Real}, task=nothing::Task)
    # TODO: add pruning

    train = data[:,task.features]
    target = data[:,task.target[1]]

    tree = build_tree(target, train, modelᵧ.parameters...)

    MLRModel(tree, modelᵧ.parameters)
end

function predictᵧ(modelᵧ::MLRModel{<:DecisionTree.Node};
                     data_features=nothing::Matrix, task=nothing::Task)
    probs = apply_tree(modelᵧ.model, data[:,task.features])
    preds = [p>0.5?1:0 for p in probs]
    preds, probs
end


function learnᵧ(modelᵧ::MLRModel{<:DecisionForestᵧ}; learner=nothing::Learner, data=nothing::Matrix{Real}, task=nothing::Task)
    # TODO: add pruning

    train = data[:,task.features]
    targets = data[:,task.targets]

    forest = build_forest(targets, train, modelᵧ.parameters...)

    MLRModel(forest, modelᵧ.parameters)
end

function predictᵧ(modelᵧ::MLRModel{<:DecisionTree.Ensemble};
                    data_features=nothing::Matrix, task=nothing::Task)
    probs = apply_forest(modelᵧ.model, data_features)
    preds = [p>0.5?1:0 for p in probs]
    preds, probs
end
