# Usage:
# include("load_titanic.jl")
# train, test = load()

using JuliaDB

function convert_embarked(loc)
    if loc == "S"
        return 0
    elseif loc == "C"
        return 1
    else
        return 2
    end
end

# converts JuliaDB table to
# a regular array
function table_to_array(tab)
    arr = zeros(length(tab),length(tab[1]))
    for i = 1:length(tab[1])
        arr[:,i] = select(tab,i)
    end
    return arr
end

function load()
    # Load dataset
    titanic = loadtable("resources/titanic_train.csv")
    # remove any rows with NA values
    titanic_clean = dropna(titanic)

    # codify categorics
    titanic_clean = setcol(titanic_clean, :Sex, convert.(Int32, select(titanic_clean,:Sex) .== "male" ))
    titanic_clean = setcol(titanic_clean, :Embarked, convert_embarked.(select(titanic_clean,:Embarked)))

    # get rid of some columns
    titanic_clean = popcol(titanic_clean, :Ticket)
    titanic_clean = popcol(titanic_clean, :Cabin)
    titanic_clean =popcol(titanic_clean, :Name)

    # Divide into training-testing sets (fix seed)
    srand(1234)
    X = collect(1:length(titanic_clean))
    X = X[randperm(length(X))]
    test_indeces = X[1:80]
    train_indeces = X[81:end]

    # assemble the random samples
    test = titanic_clean[sort(test_indeces)]
    train = titanic_clean[sort(train_indeces)]

    train_targets = copy( select(train, :Survived) )
    train = popcol(train, :Survived)

    test_targets = copy( select(test, :Survived) )
    test = popcol(test, :Survived)

    return table_to_array(train), train_targets, table_to_array(test), test_targets
end
