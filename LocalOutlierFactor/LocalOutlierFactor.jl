#Unsupervised Outlier Detection using Local Outlier Factor (LOF)
#The anomaly score of each sample is called Local Outlier Factor.
#It measures the local deviation of density of a given sample with
#respect to its neighbors.
#It is local in that the anomaly score depends on how isolated the object
#is with respect to the surrounding neighborhood.
#More precisely, locality is given by k-nearest neighbors, whose distance
#is used to estimate the local density.
#By comparing the local density of a sample to the local densities of
#its neighbors, one can identify samples that have a substantially lower
#density than their neighbors. These are considered outliers.
# ============================================================
include("NearestNeighborsWrapper.jl")
using StatsBase.percentile

""""
    fit_predict(X,
                n_neighbors,
                algorithm,
                leaf_size,
                metric,
                contamination)
Fits the model to the training set X and returns the labels
(1 inlier, -1 outlier) on the training set according to the LOF score
and the contamination parameter.
"""
function fit_predict(X::AbstractArray,
                     n_neighbors::Int = 20,
                     algorithm::AbstractString = "kd_tree",
                     leaf_size::Int = 30,
                     metric = Euclidean(),
                     contamination::Float64 = 0.05)
    lof, threshold = fit(X, n_neighbors, algorithm, leaf_size, metric, contamination)
    predict(lof, threshold)
end

"""
    fit(X,
        n_neighbors,
        algorithm,
        leaf_size,
        metric,
        contamination)
Fits the model using X as training data and returns the LOF values
and the threshold value.
"""
function fit(X::AbstractArray,
             n_neighbors::Int = 20,
             algorithm::AbstractString = "kd_tree",
             leaf_size::Int = 30,
             metric = Euclidean(),
             contamination::Float64 = 0.05)
    n_samples = size(X)[1]
    lof = zeros(Float64, n_samples)
    if n_neighbors > n_samples
        warn("n_neighbors (", n_neighbors,") is greater than
              the total number of samples (",n_samples,").
              n_neighbors will be set to n_samples - 1 for estimation.")
    end
    n_neighbors_ = max(1, min(n_neighbors, n_samples - 1))
    neighbors_indices,  distances = kneighbors(X, n_neighbors, algorithm,
                                               leaf_size, metric)
    lrd = local_reachability_density(distances, neighbors_indices, n_neighbors)
    for i=1:n_samples
        lof[i] = mean(lrd[neighbors_indices[i]]./lrd[i])
    end
    threshold = percentile(lof, 100. * (1. - contamination))
    lof, threshold
end

"""
    predict(local_outlier_factor,
            threthreshold)
predict the labels using local outlier factors and the threshold value
"""
function predict(local_outlier_factor::AbstractVector,
                 threthreshold::Float64)
    is_inlier = ones(Int,size(local_outlier_factor)[1])
    is_inlier[local_outlier_factor.>threthreshold] = -1
    is_inlier
end

"""
    local_reachability_density(dists,
                               idxs,
                               n_neighbors)
local_reachability_density returns local reachability density of a sample
which is the inverse of the averagereachability distance of its k-nearest
neighbors.
"""
function local_reachability_density(dists::AbstractVector,
                                    idxs::AbstractVector,
                                    n_neighbors::Int)
    n_samples = size(dists)[1]
    ird = zeros(Float64, n_samples)
    dist_m = (hcat(dists...))'
    dist_k = dist_m[:, n_neighbors]
    for i=1:n_samples
        ird[i] = 1/mean(max.(dist_k[idxs[i]], dist_k[i]))
    end
    ird
end
