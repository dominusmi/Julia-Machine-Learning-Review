using SparseRegression

############# MODEL SPECIFIC CONSTRUCTORS #################

"""
    Functions specifying how model should be constructed given parameters
    These function are completely model dependent and are the only ones
    that need to be written to add a new model to the list.

    TODO:
    - Add more models
    - Add functions specifying which parameter per model and what form they should take
    - Make checks that parameters are of the type/form that they should
"""
function makeRidge(learner::Learner, task::Task, data)
    if isempty(learner.parameters)
        model = SModel(data[:, task.features], data[:, task.target])
    else
        parameters = []
        push!(parameters, get_λ(learner.parameters, data))
        model = SModel(data[:, task.features], data[:, task.target], L2DistLoss(), L2Penalty(), parameters...)
    end
    MLRModel(model, copy(learner.parameters))
end

function makeLasso(learner::Learner, task::Task, data)
    if isempty(learner.parameters)
        model = SModel(data[:, task.features], data[:, task.target])
    else
        parameters = []
        push!(parameters, get_λ(learner.parameters, data))
        model = SModel(data[:, task.features], data[:, task.target], L2DistLoss(), L1Penalty(), parameters...)
    end
    MLRModel(model, copy(learner.parameters))
end

function makeElasticnet(learner::Learner, task::Task, data)
    if isempty(learner.parameters)
        model = SModel(data[:, task.features], data[:, task.target])
    else
        parameters = []
        push!(parameters, get_λ(learner.parameters, data))
        model = SModel(data[:, task.features], data[:, task.target], L2DistLoss(), ElasticNetPenalty(α=0.5), parameters...)
    end
    MLRModel(model, copy(learner.parameters))
end

function makeGlm(learner::Learner, task::Task, data)
    if isempty(learner.parameters)
        model = SModel(data[:, task.features], data[:, task.target])
    else
        parameters = []
        if get(learner.parameters, "λ", false) !== false
            # Add λ
            push!(parameters, get_λ(learner.parameters, task))
        end
        if get(learner.parameters, "penalty", false) !== false
            # Add penalty
            push!(parameters, learner.parameters["penalty"])
        end
        if get(learner.parameters, "loss", false) !== false
            # Add penalty
            push!(parameters, learner.parameters["loss"])
        end
        model = SModel(data[:, task.features], data[:, task.target], parameters...)
    end
    MLRModel(model, copy(learner.parameters))
end

# Utiliy function #
function get_λ(parameters, task)
    if get(parameters, "λ", false) == false
        lambda = fill(0.0, task.features)
    elseif typeof(parameters["λ"]) <: Real
        lambda = fill(parameters["λ"], length(task.features) )
    elseif typeof(parameters["λ"]) <: Vector{Float64}
        lambda = copy(parameters["λ"])
    end
    lambda
end

################## MODEL SPECIFIC ALGORITHMS ####################

"""
    How to predict using a specific model
"""
function predictᵧ(modelᵧ::MLRModel{<:SModel}; data=data, task=task)
    predict(modelᵧ.model, data[:, task.features])
end

"""
    How to learn using a specific model
"""
function learnᵧ!(modelᵧ::MLRModel{<:SModel}; learner=nothing::Learner, data=nothing::Matrix{Real}, task=nothing::Task)
    learn!(modelᵧ.model)
end
