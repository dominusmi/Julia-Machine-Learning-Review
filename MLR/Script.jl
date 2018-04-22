include("MLJ.jl")
include("Tuning.jl")
include("glm_wrapper.jl")

ps = ParametersSet([
    ContinuousParameter(
        name = "λ",
        lower = -4,
        upper = 1,
        transform = x->10^x
    )
        ,
    DiscreteParameter(
        name = "penalty",
        values = [L1Penalty(), L2Penalty(), ElasticNetPenalty()]
    )
])

data = Fakedata(1000,3)

task = Task(task_type="regression", target=4, data=data)
lrn = Learner("glm")

tune(learner=lrn, task=task, data=data, parameters_set=ps,
measure=mean_squared_error)
