

store_results!(no_storage::Void, measure, laerner) = nothing

function store_results!(storage::MLRStorage, measure::Any, learner::Learner)
    push!(storage.models, learner.name)
    push!(storage.measures, measure)
    push!(storage.parameters, learner.parameters)
end
