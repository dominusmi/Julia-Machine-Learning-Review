using Plots
plotlyjs()

# x = 1:3
# y = 1:3
# text = [["ergerg", "a"], ["ergerg", "a"], "ggre"]
#
# plot(x,y, hover=(text))


function plot_storage(storage::MLRStorage)
    models = Set(storage.models)
    markers = []
    measures = []

    for model in models
        indeces = []
        for (i, p_models) in enumerate(storage.models)
            if p_models == model
                push!(indeces, i)
            end
        end
        measures = storage.averageCV[indeces]
        for dict in storage.parameters[indeces]
            _marker = ""
            for (key, value) in dict
                _marker = _marker * "$key: $value, "
            end
            push!(markers, _marker)
        end
    end

    plot(measures, hover=markers, label=models)
end
