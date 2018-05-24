include("../Mondrian_Forest_Regressor.jl")

# simple regression task
X = collect(linspace(1,16,10))
Y = X+sin.(X)

# Mondrian Forest Regression object
MF = Mondrian_Forest_Regressor()

# training uses @parallel
# (not sure if the macro call
# is relevant to the problem)
train!(MF, X, Y, 1e6, 10);

# not so good
y =  predict!(MF,X);
# zeros..
println("All zero without the print: ", y)

# y = @enter predict!(MF,X);
# # zeros..
# println("All zero without the print: ", y)

# all good now with the added print("") (not all zero prediction)
function predict_with_print!{X<:Array{<: AbstractFloat} where N,}(
                  MF::Mondrian_Forest_Regressor,
                  Data::X)
    pred = zeros(MF.n_trees,size(Data,1))
    # if this print is not here
    # the regressor predicts
    # all zeros !!!
    print("")
    for item in enumerate(MF.Trees)
        pred[item[1],:] = predict_reg_batch(item[2].Tree, Data)
    end
    return reshape(mean(pred,1),size(Data,1))
end

y =  predict_with_print!(MF,X);
print("All good with print: ", y)
