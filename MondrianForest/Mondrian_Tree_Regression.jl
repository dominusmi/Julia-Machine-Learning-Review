
include("Mondrian_Tree.jl")

# The regression requires each node have a
# mean, variance and weight so for now I
# have written these two modified functions

# I think the mean and variance of of the targets
# they just call it predictive mean and variance
function Sample_Mondrian_Tree!(Tree::Mondrian_Tree,
                               λ::Float64,
                               X::Array{Float64,N} where N,
                               Y::Array{Float64, N} where N,
                               min_samples_split=10)
    # initialise the tree
    e = Mondrian_Node(0.0,[false,false,true])
    Tree.root = e
    Θ = Axis_Aligned_Box(get_intervals(X))
    e.Θ = Θ
    # regression data
    e.m = mean(Y)
    e.v = var(Y)
    Sample_Mondrian_Block!(e, Θ, λ, Tree, X, Y, min_samples_split)
    return Tree
end

function Sample_Mondrian_Block!(j::Mondrian_Node,
                                Θ::Axis_Aligned_Box,
                                λ::Float64,
                                Tree::Mondrian_Tree,
                                X::Array{Float64,N} where N,
                                Y::Array{Float64,N} where N,
                                min_samples_split=10)
    # check if minimum samples reached
    if size(Y,1) < min_samples_split
        j.τ = λ
    else
        # sample time
        E = rand(Exponential(1/Linear_dimension(Θ)))
        if j.node_type[3]==true
            τₚ = 0
        else
            τₚ = (get(j.parent)).τ
        end
        j.τ = τₚ+E
    end
    # if splits ocurred in time
    if j.τ < λ
        # get split dimension and cut position
        # A2 -> lines 6,7
        d,x = sample_split_dimension(Θ)
        # update node j's data
        j.δ = d
        j.ζ = x
        Θᴸ = copy(Θ)
        # look at this copy
        Θᴿ = copy(Θ)
        # Left and Right children have constricted boxes
        Θᴸ.Intervals[d,2]=x
        Θᴿ.Intervals[d,1]=x
        # check there is actually data here
        Xᴿ = get_data_indices(Θᴿ,X,d)
        Xᴸ = get_data_indices(Θᴸ,X,d)
        # strictly binary tree
        if (size(Xᴿ,1)>0 && size(Xᴸ,1)>0)
            right = Mondrian_Node(0.0, [true,false,false])
            right.parent = j
            # data changes A2 -> lines 8,9,10
            right.Θ = Θᴿ
            right.m = mean(Y[Xᴿ])
            right.v = var(Y[Xᴿ])
            if isnan(right.v)
                right.v = j.v
            end
            j.right = right

            left = Mondrian_Node(0.0, [true,false,false])
            left.parent = j
            left.Θ = Θᴸ
            left.m = mean(Y[Xᴸ])
            left.v = var(Y[Xᴸ])
            if isnan(left.v)
                left.v = j.v
            end
            j.left = left

            # recurse
            Sample_Mondrian_Block!(left, get(left.Θ), λ, Tree, X[Xᴸ,:], Y[Xᴸ],min_samples_split)
            Sample_Mondrian_Block!(right, get(right.Θ),λ, Tree, X[Xᴿ,:], Y[Xᴿ],min_samples_split)
        # set j as a leaf for no data/ not binary
        else
            j.τ = λ
            j.node_type = [false,true,false]
            push!(Tree.leaves,j)
            return
        end
    # set j as leaf for time out
    else
        j.τ = λ
        # this is is to handle the case of a single
        # data point, so the root is a leaf!
        if j.node_type == [false,false,true]
            j.node_type = [false,true,true]
        # normal stuff
        else
            j.node_type = [false,true,false]
        end
        push!(Tree.leaves,j)
        return
    end
end

function predict_reg(MT::Mondrian_Tree,x)
    j = get(MT.root)
    p_not_sep = 1
    # Gaussian posterior mixture sum
    G = 0
    while true
        # if root
        if j.node_type == [0,0,1]
            τₚ = 0
        else
            τₚ=get(j.parent).τ
        end
        Δⱼ=j.τ-τₚ
        nⱼ=0
        for d in size(x,1)
            nⱼ += max(x[d]-get(j.Θ).Intervals[d,2],0) + max(get(j.Θ).Intervals[d,1]-x[d],0)
        end
        pⱼ = 1-exp(-Δⱼ*nⱼ)
        if pⱼ>0
            j.w = p_not_sep*pⱼ
            G += (j.w)*(rand(Normal(j.m,j.v)))
        end
        # if leaf
        if j.node_type==[0,1,0]
            j.w = p_not_sep*(1-pⱼ)
            G += (j.w)*(rand(Normal(j.m,j.v)))
            return G
        else
            p_not_sep = p_not_sep*(1-pⱼ)
            if x[get(j.δ)] <= get(j.ζ)
                j = get(j.left)
            else
                j = get(j.right)
            end
        end
    end
end

function predict_reg_batch(MT::Mondrian_Tree, X::Array{Float64,N} where N)
    P = zeros(size(X,1))
    for i in 1:size(X,1)
        P[i] = predict_reg(MT,X[i])
    end
    return P
end
