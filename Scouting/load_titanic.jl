# Usage:
# include("load_titanic.jl")
# train, test = load()

using MLLabelUtils
using CSV

function convert_embarked(loc)
    if loc == "S"
        return 0
    elseif loc == "C"
        return 1
    else
        return 2
    end
end

function load()
    # Load dataset
    train = CSV.read("titanic_train.csv")

    # Remove columns and missing rows from train set
    to_remove = ( Missings.ismissing.( train[6] ) )
    to_remove = to_remove .| (Missings.ismissing.(train[12]))
    train = train[ .!to_remove, : ]
    train = train[:, [:PassengerId, :Survived, :Pclass, :Sex, :Age, :SibSp, :Parch, :Fare, :Embarked] ]

    # Convert categorical to integers
    train[:Sex] = convert.(Int32, train[:Sex] .== "male" )
    train[:Embarked] = convert_embarked.(train[:Embarked])

    # Divide into training-testing sets (fix seed)
    srand(1234)
    X = collect(1:length(train[1]))
    X = X[randperm(length(X))]
    test_indeces = X[1:80]
    train_indeces = X[81:end]

    test = train[test_indeces, :]
    train = train[train_indeces, :]

    return convert(Array{Float64}, train), convert(Array{Float64}, test)
end
