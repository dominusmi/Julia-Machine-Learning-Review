import StatsBase: predict
import Base: getindex, show
import MLBase: Kfold, fit!
using MLMetrics
using SparseRegression

"""
    Contains task type (regression,classification,..)
    and the columns to use as target and as features.
    TODO: accept multiple targets
"""
immutable Task
    task_type::String
    target::Int
    features::Array{Int}
end

function Task(;task_type="regression", target=nothing, data=nothing)
    if target == nothing || data == nothing
        throw("Requires target and data to be set")
    end

    features = size(data,2)
    features = deleteat!( collect(1:features), target)

    Task(task_type, target, features)
end

"""
    Contains the name and the parameters of the model to train.
"""
immutable Learner
    name::String
    parameters::Union{Void,Dict{Any}}
    Learner(learner::String) = new(learner, Dict())
    Learner(learner::String, parameters::Dict{Any}) = new(learner, parameters)
end

function show(io::IO,l::Learner)
    println("Learner: $(l.name)")
    for (key, value) in l.parameters
       println(" ▁ ▂ ▃ $key: $value")
    end
end

"""
    Allows resampling for cross validation
    TODO: add more methods (only accepts k-fold)
"""
immutable Resampling
    method::String
    iterations::Int
    Resampling() = new("KFold", 3)
end


"""
 A parameter set allows a user to add multiple parameters to tune
 It must include a name. Constructor only accepts key arguments
 TODO: parameters cross-checked by learner to see whether they are valid
"""
abstract type Parameter end

"""
    Discrete parameter requires a name and an array of value to check
    TODO: check whether values are correct for specific learner
"""
immutable DiscreteParameter <: Parameter
    name::String
    values::Array{Any}
    DiscreteParameter(;name=nothing,values=nothing) = new(name, values)
end

"""
    Tuning of a parameter. Must provide name, lower&upper bound, and transform
    that iterates through values in lower:upper and gives te actual parameter to test

    e.g.
    ```julia
        # Will check λ={1,4,9,16}
        ContinuousParameter("λ", 1, 4, x->x²)
    ```
"""
immutable ContinuousParameter <: Parameter
    name::String
    lower::Real
    upper::Real
    transform::Function
    ContinuousParameter(;name=nothing, lower=nothing, upper=nothing, transform=nothing) = new(name, lower, upper, transform)
end

"""
    Set of parameters.
    Will be used to implement checks on validity of parameters
"""
immutable ParametersSet
   parameters::Array{Parameter}
end
getindex(p::ParametersSet, i::Int64) = p.parameters[i]

"""
    Abstraction layer for model
"""
mutable struct MLRModel{T}
    model::T
    parameters
    inplace::Bool
end
MLRModel(model, parameters::Dict; inplace=true) = MLRModel(model, parameters, inplace)

#### ABSTRACT FUNCTIONS ####
"""
    Constructor for any model. Will call the function makeModelname, where
    modelname is stored in learner.name
    Function makeModelname should be defined separately for each model
"""
function MLRModel(learner::Learner, task::Task, data)
    # Calls function with name "makeModelname"
    f_name = learner.name
    f_name = "make" * titlecase(f_name)

    f = getfield(Main, Symbol(f_name))
    f(learner, task, data)
end

"""
    Function which sets up model given by learner, and then calls model-based
    learning function, which must be defined separately for each model.
"""
function learnᵧ(learner::Learner, task::Task, data)
    modelᵧ = MLRModel(learner, task, data)
    if modelᵧ.inplace
        learnᵧ!(modelᵧ, learner=learner, task=task, data=data)
    else
        modelᵧ = learnᵧ(modelᵧ, learner=learner, task=task, data=data)
    end
    modelᵧ
end

#### UTILITIES ####
function Fakedata(N,d)
    n_obs = 100
    x = randn((n_obs,d))
    y = sum(x*randn(d),2)

    hcat(x,y)
end

function FakedataClassif(N,d)
    n_obs = 100
    x = randn((n_obs,d))
    y = ( sum(x*randn(d),2) .> mean(sum(x*randn(d),2)) )

    hcat(x,y)
end
