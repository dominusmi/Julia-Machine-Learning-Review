include("MLJ.jl")

# Decision trees example

include("decisiontree_wrapper.jl")
data = FakedataClassif(1000,3)

task = Task(task_type="classification", targets=[4], data=data)
lrn = Learner("forest", Dict("nsubfeatures"=>2, "ntrees"=>10))

modelᵧ = learnᵧ(lrn, task)
predictᵧ(modelᵧ, data[:,task.features], task)

# Multivariate example

include("multivariate_wrapper.jl")
data = Fakedata(1000,4)

task = Task(task_type="regression", targets=[3], data=data)
lrn = Learner("multivariate", Dict("regType"=>"llsq"))

modelᵧ = learnᵧ(lrn, task)
predictᵧ(modelᵧ, data[:, task.features], task)



## Example regression using GLM with penalty and λ tuning

include("glm_wrapper.jl")

ps = ParametersSet([
    ContinuousParameter(
        name = "cost",
        lower = -4,
        upper = 1,
        transform = x->10^x
    ),
    DiscreteParameter(
        name = "penalty",
        values = [L2Penalty(), L1Penalty()]
    )
])

data = Fakedata(1000,3)

task = Task(task_type="classification", targets=[4], data=data)
lrn = Learner("glm")

storage = MLRStorage()

tune(lrn, task, ps,
    measure=mean_squared_error, storage=storage)
#
include("Visualisation.jl")

plot_storage(storage)
# Example classification using SVM with type and cost tuning

include("libsvm_wrapper.jl")
ps = ParametersSet([
    ContinuousParameter(
        name = "cost",
        lower = -4,
        upper = 1,
        transform = x->10^x
    ),
    DiscreteParameter(
        name = "svmtype",
        values = [SVC()]
    ),
    DiscreteParameter(
        name = "kernel",
        values = [Kernel.Polynomial]
    ),
    ContinuousParameter(
        name = "coef0",
        lower = -4,
        upper = 1,
        transform = x->10^x
    ),
])

data = FakedataClassif(1000,3)

task = Task(task_type="classification", targets=[4], data=data)
lrn = Learner("libsvm")

tune(lrn, task, ps,
    measure=accuracy)


# Multiplex example
include("glm_wrapper.jl")
include("multivariate_wrapper.jl")

data = Fakedata(1000,4)
task = Task(task_type="regression", targets=[3], data=data)

lrns = Array{Learner}(0)
psSet = Array{ParametersSet}(0)

lrn = Learner("glm")
ps = ParametersSet([
    ContinuousParameter(
        name = "cost",
        lower = -4,
        upper = 1,
        transform = x->10^x
    ),
    DiscreteParameter(
        name = "penalty",
        values = [L2Penalty(), L1Penalty()]
    )
])
push!(lrns, lrn)
push!(psSet, ps)

lrn = Learner("multivariate")
ps = ParametersSet([
    DiscreteParameter(
        name="regType",
        values = ["llsq", "ridge"]
    ),
    ContinuousParameter(
        name = "λ",
        lower = -4,
        upper = 1,
        transform = x->10^x
    )
])
push!(lrns, lrn)
push!(psSet, ps)

storage = MLRStorage()
mp = MLRMultiplex(lrns, psSet)
include("Tuning.jl")
include("multivariate_wrapper.jl")
include("Visualisation.jl")
tune(mp, task, storage=storage, measure=mean_squared_error)
plot_storage(storage, plotting_args=Dict(:scale=>:log10))
# modelᵧ = learnᵧ(lrn, task, data)
# predictᵧ(modelᵧ, data_features=data, task=task)

# Multivariate example

#
# modelᵧ = learnᵧ(lrn, task, data)
# predictᵧ(modelᵧ, data=data, task=task)
