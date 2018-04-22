include("MLJ.jl")
include("Tuning.jl")
include("glm_wrapper.jl")
include("libsvm_wrapper.jl")


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


ps = ParametersSet([
    ContinuousParameter(
        name = "cost",
        lower = -4,
        upper = 1,
        transform = x->10^x
    ),
    DiscreteParameter(
        name = "svmtype",
        values = [SVC(), NuSVC()]
    )
])

data = FakedataClassif(1000,3)

task = Task(task_type="classification", target=4, data=data)
lrn = Learner("libsvm")

tune(learner=lrn, task=task, data=data, parameters_set=ps,
    measure=mean_squared_error)

svmpredict()
