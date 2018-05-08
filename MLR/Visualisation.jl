using Plots
plotlyjs()


x = [1,2,3]
y = [2,3,1]

text = ["ergerg", "ergerg", "ggre"]

plot(x,y, hover=(text))


function plot_storage(storage::MLRStorage)
    models = Set(storage.models)
    markers = []

    for model in models
        indeces = []
        for i, p_models in enumerate(models)
            if p_models == model
                push!(indeces, model)
            end
        end
        y = storage.averageCV[indeces]
        for key, value in storage.params[indeces]
            push!(markers, "$key: $value")
        end
    end

    plot()
end
