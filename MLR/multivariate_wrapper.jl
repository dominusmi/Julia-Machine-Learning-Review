using MultivariateStats


### TODO: Types need to be re-thought

abstract type  MultivariateModel end

mutable struct MultivariateLlsq <: MultivariateModel
    sol::Matrix{<:Float64}
    MultivariateLlsq() = new(zeros(0,0))
end

mutable struct MultivariateRidge <: MultivariateModel
    λ::Float64
    sol::Matrix{<:Float64}
    MultivariateRidge(λ) = new(λ, zeros(0,0))
end

function getParamsMultivariate()
    possible_parameters = Dict(
        "regType"=>["ridge", "llsq"]
    )
    possible_parameters
end

function makeMultivariate(learner::Learner, task::Task, data)
    prms = learner.parameters
    possible_parameters = getParamsMultivariate()

    if prms["regType"] in possible_parameters["regType"]
        if prms["regType"] == "ridge"
            λ = get(prms, "λ", false)
            if λ == false λ=0.1 end
            MLRModel(MultivariateRidge(λ), copy(prms))
        else
            MLRModel(MultivariateLlsq(), copy(prms))
        end
    else
        throw("regType must be either \"ridge\" or \"llsq\"")
    end
end


function learnᵧ!(modelᵧ::MLRModel{<:MultivariateRidge}; learner=nothing::Learner,
                data=nothing::Matrix{Real}, task=nothing::Task)

        modelᵧ.model.sol = ridge(data[:,task.features], data[:,task.targets], modelᵧ.model.λ)
end


function learnᵧ!(modelᵧ::MLRModel{<:MultivariateLlsq}; learner=nothing::Learner,
                data=nothing::Matrix{Real}, task=nothing::Task)

        modelᵧ.model.sol = llsq(data[:,task.features], data[:,task.targets])
end

function predictᵧ(modelᵧ::MLRModel{<:MultivariateModel};
                    data_features=nothing::Matrix{Real}, task=nothing::Task)

    sol = modelᵧ.model.sol
    A, b = sol[1:end-1,:], sol[end,:][:,:]
    preds = data_features * A .+ b'
    preds, nothing
end
