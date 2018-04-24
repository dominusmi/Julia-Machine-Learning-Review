include("MLJ.jl")
include("Tuning.jl")
include("glm_wrapper.jl")
include("libsvm_wrapper.jl")
include("decisiontree_wrapper.jl")


# Decision trees example

data = FakedataClassif(1000,3)

task = Task(task_type="classification", target=4, data=data)
lrn = Learner("forest", Dict("nsubfeatures"=>2, "ntrees"=>10))

modelᵧ = makeForest(lrn, task, data)
modelᵧ = learnᵧ(modelᵧ, learner=lrn, data=data, task=task)
predictᵧ(modelᵧ, data=data, task=task)


## Example regression using GLM with penalty and λ tuning

# ps = ParametersSet([
#     ContinuousParameter(
#         name = "cost",
#         lower = -4,
#         upper = 1,
#         transform = x->10^x
#     ),
#     DiscreteParameter(
#         name = "penalty",
#         values = [L2Penalty(), L1Penalty()]
#     )
# ])
#
# data = Fakedata(1000,3)
#
# task = Task(task_type="classification", target=4, data=data)
# lrn = Learner("glm")
#
# tune(learner=lrn, task=task, data=data, parameters_set=ps,
#     measure=mean_squared_error)

# variable_select_forward(learner=lrn, task=task, data=data, measure=mean_squared_error)


## Example classification using SVM with type and cost tuning


# ps = ParametersSet([
#     ContinuousParameter(
#         name = "cost",
#         lower = -4,
#         upper = 1,
#         transform = x->10^x
#     ),
#     DiscreteParameter(
#         name = "svmtype",
#         values = [SVC()]
#     ),
#     DiscreteParameter(
#         name = "kernel",
#         values = [Kernel.Polynomial]
#     ),
#     ContinuousParameter(
#         name = "coef0",
#         lower = -4,
#         upper = 1,
#         transform = x->10^x
#     ),
# ])
#
# data = FakedataClassif(1000,3)
#
# task = Task(task_type="classification", target=4, data=data)
# lrn = Learner("libsvm")
#
# tune(learner=lrn, task=task, data=data, parameters_set=ps,
#     measure=accuracy)
