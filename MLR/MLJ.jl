import StatsBase: predict
import Base: getindex, show
import MLBase: Kfold
using MLMetrics
using SparseRegression

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

immutable Resampling
    method::String
    iterations::Int
    Resampling() = new("KFold", 3)
end

abstract type Parameter end

immutable DiscreteParameter <: Parameter
    name::String
    values::Array{Any}
    DiscreteParameter(;name=nothing,values=nothing) = new(name, values)
end

immutable ContinuousParameter <: Parameter
    name::String
    lower::Real
    upper::Real
    transform::Function
    ContinuousParameter(;name=nothing, lower=nothing, upper=nothing, transform=nothing) = new(name, lower, upper, transform)
end


immutable ParametersSet
   parameters::Array{Parameter}
end
getindex(p::ParametersSet, i::Int64) = p.parameters[i]

immutable MLRModel{T}
    model::T
    parameters
end


#### ABSTRACT FUNCTIONS ####

function MLRModel(learner::Learner, task::Task, data)
    # Calls function with name "makeModelname"
    f_name = learner.name
    f_name = "make" * titlecase(f_name)

    f = getfield(Main, Symbol(f_name))
    f(learner, task, data)
end

function learnᵧ(learner::Learner, task::Task, data)
    modelᵧ = MLRModel(learner, task, data)
    learnᵧ!(modelᵧ, learner=learner, task=task, data=data)
    modelᵧ
end

#### UTILITIES ####

function Fakedata(N,d)
    n_obs = 100
    x = randn((n_obs,d))
    y = sum(x*randn(d),2)

    hcat(x,y)
end
