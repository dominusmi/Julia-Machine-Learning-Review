using Plots
plotlyjs()

# x = 1:3
# y = 1:3
# text = [["ergerg", "a"], ["ergerg", "a"], "ggre"]
#
# plot(x,y, hover=(text))


function plot_storage(storage::MLRStorage; plotting_args=[])
    models = Set(storage.models)
    fig = plot(;plotting_args...)
    for model in models
        markers = []
        measures = []
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
        plot!(measures; hover=markers, label=model)
    end
    fig
end
