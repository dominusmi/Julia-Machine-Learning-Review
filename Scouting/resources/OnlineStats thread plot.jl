using OnlineStats
using Plots

# Setup variables
repetitions = 5;
sizes = [900_000, 3_000_000, 9_000_000, 27_000_000];

end_times = zeros(4,2)
end_stds  = zeros(4,2)

for (size_index, n_obs) in enumerate(sizes)

    # Prepare data, with σ=0.1 normal noise
    x_train = randn(n_obs, 6)
    noise = 0.1*randn(n_obs)
    y_train = [f(x_train[i,:])+noise[i] for i=1:size(x_train,1) ]

    x_test = randn(50, 6)
    y_test = [f(x_test[i,:])+0.1*randn() for i=1:size(x_test,1) ];

    repetition_times = zeros(repetitions,2)

    for i in 1:repetitions
        # Specify model: Ridge with 6 input dimensions, using λ=0.1 for all features, and SGD optimiser
        o = StatLearn(6, L2Penalty(), L2DistLoss(), fill(0.1, 6), SGD())

        # Make it into a series and fit it
        s = Series(o);

        ### Single thread ###
        tic()
        fit!(s, (x_train, y_train));
        _end = toq();

        repetition_times[i,1] = _end;

        ### In parallel ###
        s1 = Series(StatLearn(6, L2Penalty(), L2DistLoss(), fill(0.1, 6), SGD()))
        s2 = Series(StatLearn(6, L2Penalty(), L2DistLoss(), fill(0.1, 6), SGD()))
        s3 = Series(StatLearn(6, L2Penalty(), L2DistLoss(), fill(0.1, 6), SGD()))

        set_size = convert(Int32, n_obs/3)    

        tic()
        # Divide task into three
        @spawn fit!(s1, (x_train[1:set_size,:], y_train[1:set_size]))
        @spawn fit!(s2, (x_train[set_size+1:2*set_size,:], y_train[set_size+1:2*set_size]))
        @spawn fit!(s3, (x_train[2*set_size+1:3*set_size,:], y_train[2*set_size+1:3*set_size]))

        merge!(s1, s2)  # merge information from s2 into s1
        merge!(s1, s3)  # merge information from s3 into s1
        _end = toq();

        repetition_times[i,2] = _end;

        # Check that coefs do actually agree
#         println("### Sanity check ###\nCoefficients comparison:\n")
#         n_coefs = coef(o)
#         p_coefs = value(s1)[1]
#         for i in 1:6
#             println("Normal: $(round(n_coefs[i],3)) \tParallel: $(round(p_coefs[i],3))")
#         end
    end
    end_times[size_index, :] = mean(repetition_times,1)
    end_stds[size_index, :]  = std(repetition_times,1)
end