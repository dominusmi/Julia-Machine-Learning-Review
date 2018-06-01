using NearestNeighbors

"""
    kneighbors(X,
               n_neighbors,
               algorithm,
               leaf_size,
               metric = Euclidean())
kneighbors returns the k Nearest Neighbors of X based
on input parameters
"""
function kneighbors(X::AbstractArray,
                    n_neighbors::Int = 20,
                    algorithm::AbstractString = "kd_tree",
                    leaf_size::Int = 30,
                    metric = Euclidean())
    if algorithm == "kd_tree"
        tree = KDTree( X', metric; leafsize = leaf_size)
        idxs, dists = knn(tree, X', n_neighbors, true)
    elseif algorithm == "ball_tree"
        tree = BallTree( X', metric; leafsize = leaf_size)
        idxs, dists = knn(tree, X', n_neighbors , true)
    elseif algorithm == "brute_tree"
        tree = BruteTree( X', metric; leafsize = leaf_size)
        idxs, dists = knn(tree, X', n_neighbors, true)
    else
        throw(DomainError())
    end
    idxs, dists
end
