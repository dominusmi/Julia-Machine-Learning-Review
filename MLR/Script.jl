include("MLJ.jl")
include("Tuning.jl")

# Decision trees example

# include("decisiontree_wrapper.jl")
# data = FakedataClassif(1000,3)
#
# task = Task(task_type="classification", target=4, data=data)
# lrn = Learner("forest", Dict("nsubfeatures"=>2, "ntrees"=>10))
#
# modelᵧ = learnᵧ(lrn, task, data)
# predictᵧ(modelᵧ, data_features=data, task=task)

# Multivariate example

# include("multivariate_wrapper.jl")
# data = Fakedata(1000,4)
#
# task = Task(task_type="regression", target=[3], data=data)
# lrn = Learner("multivariate", Dict("regType"=>"llsq"))
#
# modelᵧ = learnᵧ(lrn, task, data)
# predictᵧ(modelᵧ, data=data, task=task)



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

task = Task(task_type="classification", target=4, data=data)
lrn = Learner("glm")

storage = MLRStorage()

tune(learner=lrn, task=task, data=data, parameters_set=ps,
    measure=mean_squared_error, storage=storage)




## Example classification using SVM with type and cost tuning

# include("libsvm_wrapper.jl")
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
