using DiffEqDiffTools
using Calculus
using Plots
using LaTeXStrings
include("MLJ.jl")
include("Utilities.jl")

pyplot()

function categorical_cross_entropy(predictions)
    H = 0
    N⁻¹ = 1/length(predictions)
    for p in predictions
        H += - N⁻¹ * log2(p)
    end
    H
end

function _loss(λ)
    if λ == 0.0
        sol = llsq(dd[:,1:end-1], dd[:,end])
    else
        sol = ridge(dd[:,1:end-1], dd[:,end], λ)
    end
    println(sol)

    A, b = sol[1:end-1,:], sol[end,:][:,:]
    preds = dd[:,1:end-1] * A .+ b
    MLMetrics.mean_absolute_error(dd[:,end], preds)

    x = collect(minimum(dd[:,1]):0.1:maximum(dd[:,1]))
    plot!(x, A.*x.+b, show=true)
end


λ = [10.0^x for x in -5:1]
push!(λ,0.0)
scatter(dd[:,1], dd[:,2])
for i in λ
    _loss(i)
end


DiffEqDiffTools.finite_difference_derivative(_loss, λ)
dd[5,:]+[0.5,0.5]

dd = Fakedata(50,1)
vcat(dd, [dd[5,1] dd[5,2]+2])


sol = ridge(dd[:,1:end-1], dd[:,end], 0.0000001)
A,b = sol[1:end-1,:], sol[end,:][:,:]

x = collect(minimum(dd[:,1]):0.1:maximum(dd[:,1]))
preds = dd[:,1:end-1] * A .+ b'
mse = MLMetrics.mean_absolute_error(dd[:,end], preds)
println("MSE: $mse")
scatter(dd[:,1], dd[:,2])
plot!(x, A.*x.+b)


 X = FakedataClassifClusters(3000,2)
 colour(x) = Int64(round(x)) == 0 ? :red : :green
 scatter(X[:,1], X[:,2], markercolor=colour.(X[:,3]))

include("libsvm_wrapper.jl")
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

xs = collect(-6.0:0.1:5.0)
_l = [svm_loss(x) for x in xs]
plot(xs, _l, title="SVC Categorical Cross-Entropy Loss", xlabel="C", ylabel="Loss", label="loss")
savefig("SVC loss")

g = Calculus.derivative(svm_loss)
dx = g.(xs)
plot(xs, dx, title="SVC CCE Loss Finite Difference Derivative",
    xlabel="C", ylabel=L"\frac{\partial L}{\partial C}", label="loss")
savefig("SVC derivative of loss")
