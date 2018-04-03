function _SparseRegression(x,y, loss, penalty)
    s = SModel(x, y, loss, penalty, fill(1.0, size(x, 2)));

   tic();
   learn!(s);
   time = toq();

   return time
end

# MultivariateStats can only do ridge and least square
function _MultivariateStats(x,y, regression)
    if regression == "ols"
        tic()
        llsq(x,y)
        time = toq()
    else
        tic()
        ridge(x,y, 0.1)
        time = toq()
    end
    return time
end


repetitions = 5
n_points = 10_000
n_dims = [32, 64, 128, 256, 512, 1024, 2048, 5096]

avg_times = []

# OLS
for n_dim in n_dims
    times = zeros((repetitions,3))
    for i in 1:repetitions
        t = []
        x = randn(n_points, n_dims);
        y = x * linspace(-1, 1, n_dims) + randn(n_points);

        push!(t, _SparseRegression(x,y, L2DistLoss() ))
        push!(t, _MultivariateStats(x,y, "ridge"))
        push!(t, _OnlineStats(x,y, L2DistLoss(), L2Penalty() ))

        times[i,:] = t
    end
    avg_times = mean(times,1);
    std_times = std(times,1);
    println("Average time $(avg_times) for $(n_dim) dimensions")
end


# Ridge
for n_dim in n_dims
    times = zeros((repetitions,3))
    for i in 1:repetitions
        t = []
        x = randn(n_points, n_dims);
        y = x * linspace(-1, 1, n_dims) + randn(n_points);

        push!(t, _SparseRegression(x,y, L2DistLoss(), L2Penalty() ))
        push!(t, _MultivariateStats(x,y, "ridge"))
        push!(t, _OnlineStats(x,y, L2DistLoss(), L2Penalty() ))

        times[i,:] = t
    end
    avg_times = mean(times,1);
    std_times = std(times,1);
    println("Average time $(avg_times) for $(n_dim) dimensions")
end
