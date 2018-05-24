### Error

Mondrian_Forest_Regressor predicts all zeros [0,0,0,...] in a regression task unless predict has a print statement.

It seems that without the print function the for loop is not entered. 

The error producing function is

```Julia
function predict!{X<:Array{<: AbstractFloat} where N,}(
                  MF::Mondrian_Forest_Regressor,
                  Data::X)
    pred = zeros(MF.n_trees,size(Data,1))
    for item in enumerate(MF.Trees)
        pred[item[1],:] = predict_reg_batch(item[2].Tree, Data)
    end
    return reshape(mean(pred,1),size(Data,1))
end
```

And the 'fix' is the following

```Julia
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
```
