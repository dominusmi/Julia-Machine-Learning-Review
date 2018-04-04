using SparseRegression, MultivariateStats, OnlineStats, Plots

function _SparseRegression(x,y, loss, penalty, λ)
    s = SModel(x, y, loss, penalty, fill(λ, size(x, 2)));

   tic();
   SparseRegression.learn!(s);
   time = toq();

   return time
end

# MultivariateStats can only do ridge and least square
function _MultivariateStats(x,y, regression, λ)
    if regression == "ols"
        tic()
        llsq(x,y)
        time = toq()
    else
        tic()
        ridge(x,y, λ)
        time = toq()
    end
    return time
end

function _OnlineStats(x,y, loss, penalty, λ)
    o = StatLearn(size(x,2), loss, penalty, fill(λ, size(x,2)), SGD())
    s = Series(o);

    tic();
    fit!(s, (x, y));
    time = toq();

    return time
end


# Note: the first repetition is used as compilation, we only count repetitions-1
repetitions = 16
n_points = 10_000
n_dims = [32, 64, 128, 256, 512, 1024]

avg_times = []
std_times = []

### OLS ###
for n_dim in n_dims
    times = zeros((repetitions,3))
    for i in 1:repetitions
        t = []
        x = randn(n_points, n_dim);
        y = x * linspace(-1, 1, n_dim) + randn(n_points);

        push!(t, _SparseRegression(x,y, L2DistLoss(), L2Penalty(), 0.0))
        push!(t, _MultivariateStats(x,y, "ols", 0.0))
        push!(t, _OnlineStats(x,y, L2DistLoss(), L2Penalty(), 0.0))

        times[i,:] = t
    end
    avg_times = vcat(avg_times, mean(times[2:end,:],1));
    std_times = vcat(std_times, std(times[2:end,:],1));
    println("OLS: Average time $(avg_times) for $(n_dim) dimensions")
end

avg_times = convert(Array{Float64,2}, avg_times)
std_times = convert(Array{Float64,2}, std_times)

plot(n_dims, avg_times[:,1], label="SparseRegression", yerr=std_times[:,1], color="blue", msc="blue")
plot!(n_dims, avg_times[:,2], label="MultivariateStats", yerr=std_times[:,2], color="red", msc="red")
plot!(n_dims, avg_times[:,3], label="OnlineStats", yerr=std_times[:,3], color="green", msc="green")
plot!(title="OLS time comparison", ylabel="execution time", xlabel="number of dimensions", xscale=:log2, xlims=(2^5.5, 2^10.5))
savefig("LR_comparison_ols.png")

### Ridge ###
repetitions = 16
n_points = 10_000
n_dims = [32, 64, 128, 256, 512, 1024]

avg_times = []
std_times = []

for n_dim in n_dims
    times = zeros((repetitions,3))
    for i in 1:repetitions
        t = []
        x = randn(n_points, n_dim);
        y = x * linspace(-1, 1, n_dim) + randn(n_points);

        push!(t, _SparseRegression(x,y, L2DistLoss(), L2Penalty(), 0.1))
        push!(t, _MultivariateStats(x,y, "ols", 0.1))
        push!(t, _OnlineStats(x,y, L2DistLoss(), L2Penalty(), 0.1))

        times[i,:] = t
    end
    avg_times = vcat(avg_times, mean(times[2:end,:],1));
    std_times = vcat(std_times, std(times[2:end,:],1));
    println("Ridge: Average time $(avg_times) for $(n_dim) dimensions")
end

avg_times = convert(Array{Float64,2}, avg_times)
std_times = convert(Array{Float64,2}, std_times)

plot(n_dims, avg_times[:,1], label="SparseRegression", yerr=std_times[:,1], color="blue", msc="blue")
plot!(n_dims, avg_times[:,2], label="MultivariateStats", yerr=std_times[:,2], color="red", msc="red")
plot!(n_dims, avg_times[:,3], label="OnlineStats", yerr=std_times[:,3], color="green", msc="green")
plot!(title="Ridge time comparison", ylabel="execution time", xlabel="number of dimensions", xscale=:log2, xlims=(2^5.5, 2^10.5))
savefig("LR_comparison_ridge.png")
