using LIBSVM

# model = svmtrain(train,train_targets,probability=true)   #train the classifier (default: radial basis kernel)
# (predicted_labels, decision_values) = svmpredict(model, test);


function makeLibsvm(learner::Learner, task::Task, data)
    parameters = Dict()
    possible_parameters = Dict(
        :svmtype=>LIBSVM.AbstractSVC,
        :kernel=>Kernel.KERNEL,
        :degree=>Integer,
        :cost=>Float64,
        :coef0=>Float64,
        :nu=>Float64,
        :epsilon=>Float64,
        :tolerance=>Float64,
        :probability=>Bool,
        :verbose=>Bool
        # Many more to add but this will do for now
    )
    for (p_name, p_value) in learner.parameters
        p_symbol = Symbol(p_name)
        if get(possible_parameters, p_symbol, false) != false
            if !(typeof(p_value) <: possible_parameters[p_symbol])
                throw("Parameter $p_name is not of the correct type: $p_value
                        ($(typeof(p_value)) instead of $(possible_parameters[p_symbol]))")
            end
            parameters[p_symbol] = p_value
        else
            println("Parameter $p_name was not found and is therefore ignored")
        end
    end
    parameters[:svmtype] = typeof(parameters[:svmtype])
    MLRModel(learner.parameters["svmtype"], parameters, inplace=false)
end

function predictᵧ(modelᵧ::MLRModel{<:LIBSVM.SVM{Float64}}; data=data, task=task)
    (labels, decision_values) = svmpredict(modelᵧ.model, data[:,task.features]')
    labels, decision_values
end

function learnᵧ(modelᵧ::MLRModel{<:LIBSVM.AbstractSVC}; learner=nothing::Learner, data=nothing::Matrix{Real}, task=nothing::Task)
    train = data[:, task.features]'
    targets = data[:,task.target]

    model = svmtrain(train,targets; modelᵧ.parameters...)
    modelᵧ = MLRModel(model, modelᵧ.parameters)
end