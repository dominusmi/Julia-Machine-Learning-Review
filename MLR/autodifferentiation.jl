function _loss(λ)
       sol = ridge(dd[:,1:3], dd[:,4], λ)
       A, b = sol[1:end-1,:], sol[end,:][:,:]
       preds = dd[:,1:3] * A .+ b'
       mean_squared_error(dd[:,4], preds)
end


DiffEqDiffTools.finite_difference_derivative(_loss, [10.0^x for x in -5:1])


dd = Fakedata(50,3)

_loss(1)

sol = ridge(dd[:,1:3], dd[:,4], 0.1)
