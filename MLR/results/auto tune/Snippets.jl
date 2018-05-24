using Plots, DiffEqDiffTools, Calculus

function svm_loss(x)
    srand(1)
    x=2^x
    svm = svmtrain(X[:,1:2]', X[:,3], svmtype=SVC, cost=x, probability=true)
    preds = svmpredict(svm, X[:,1:2]')
    p_labels = convert(Array{Int}, preds[1])
    t_labels = convert(Array{Int}, X[:,3])
    # scatter(X[:,1], X[:,2], markercolor=colour.(preds))
    # MLMetrics.accuracy(preds, X[:,3])
    proba = [preds[2][t_labels[i]+1,i] for i in 1:length(X[:,3])]
    categorical_cross_entropy(proba)
end

# DiffEqDiffTools

xs = collect(-6.0:0.01:5.0)
_l = [svm_loss(x) for x in xs]
plot(xs, _l, title="SVC Categorical Cross-Entropy Loss", xlabel="C", ylabel="Loss", label="loss")
savefig("SVC loss")

dx = DiffEqDiffTools.finite_difference_derivative.(svm_loss, xs)
plot(xs, dx, title="SVC CCE Loss Finite Difference Derivative",
    xlabel="C", ylabel=L"\frac{\partial L}{\partial C}", label="loss")
savefig("SVC derivative of loss")


# Calculus

xs = collect(-6.0:0.1:5.0)
_l = [svm_loss(x) for x in xs]
plot(xs, _l, title="SVC Categorical Cross-Entropy Loss", xlabel="C", ylabel="Loss", label="loss")
savefig("SVC loss")

g = Calculus.derivative(svm_loss)
dx = g.(xs)
plot(xs, dx, title="SVC CCE Loss Finite Difference Derivative",
    xlabel="C", ylabel=L"\frac{\partial L}{\partial C}", label="loss")
savefig("SVC derivative of loss")

