function Fakedata(N,d)
    n_obs = N
    x = randn((n_obs,d))
    y = sum(x*randn(d)+randn(size(x))*0.1,2)

    hcat(x,y)

end

function FakedataClassif(N,d)
    n_obs = N
    x = randn((n_obs,d))
    y = ( sum(x*randn(d),2) .> mean(sum(x*randn(d),2)) )

    hcat(x,y)
end


function FakedataClassifClusters(N,d)

    X = zeros(N,d)
    y = zeros(Int64, N,1)

    μ = 0.0
    σ = 0.5
    for i in 1:Int64(round(N/2))
        X[i,:] =  randn((1,d))*σ+μ
        y[i] = 0
    end

    μ = 1.0
    σ = 0.5
    for i in Int64(round(N/2))+1:N
        X[i,:] = randn((1,d))*σ+μ
        y[i] = 1
    end

    hcat(X,y)
end
