include("MLJ.jl")
include("multivariate_wrapper.jl")
using Plots

data = Fakedata(20,1)

dd = zeros(100,2)
for (i,x) in enumerate(rand(100))
    dd[i,:] = [x, 2*x+1]
end


ridge(reshape(data[:,1], 100, 1), data[:,2], 0.01)

scatter(data[:,1], data[:,2])

_loss(100.0)
function _loss(λ)
       sol = ridge(reshape(data[:,1], 100, 1), data[:,2], λ)
       print(sol)
       A, b = sol[1,:][:,:], sol[end,:][:,:]
       preds = dd[:,1] * A .+ b'
       mean_squared_error(dd[:,2], preds)
end
