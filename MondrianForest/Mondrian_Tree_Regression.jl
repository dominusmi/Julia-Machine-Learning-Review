
include("Mondrian_Tree.jl")

# The regression requires each node have a
# mean, variance and weight so for now I
# have written these two modified functions

# I think the mean and variance of of the targets
# they just call it predictive mean and variance
function Sample_Mondrian_Tree!{X<:AbstractArray{Float64,N} where N,
                               Y<:AbstractArray{Float64, N} where N}(
                               Tree::Mondrian_Tree,
                               λ::AbstractFloat,
                               Data::X,
                               Labels::Y,
                               min_samples_split=10)
    # initialise the tree
    e = Mondrian_Node(0.0,[false,false,true])
    Tree.root = e
    Θ = Axis_Aligned_Box(get_intervals(Data))
    e.Θ = Θ
    # regression data
    e.m = mean(Labels)
    e.v = var(Labels)
    Sample_Mondrian_Block!(e, Θ, λ, Tree, Data, Labels, min_samples_split)
    return Tree
end

function Sample_Mondrian_Block!{X<:AbstractArray{Float64,N} where N,
                                Y<:AbstractArray{Float64, N} where N}(
                                j::Mondrian_Node,
                                Θ::Axis_Aligned_Box,
                                λ::AbstractFloat,
                                Tree::Mondrian_Tree,
                                Data::X,
                                Labels::Y,
                                min_samples_split=10)
    # check if minimum samples reached
    if size(Labels,1) < min_samples_split
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
        Dataᴿ = get_data_indices(Θᴿ,Data,d)
        Dataᴸ = get_data_indices(Θᴸ,Data,d)
        # strictly binary tree
        if (size(Dataᴿ,1)>0 && size(Dataᴸ,1)>0)
            right = Mondrian_Node(0.0, [true,false,false])
            right.parent = j
            # data changes A2 -> lines 8,9,10
            right.Θ = Θᴿ
            right.m = mean(Labels[Dataᴿ])
            right.v = var(Labels[Dataᴿ])
            if isnan(right.v)
                right.v = j.v
            end
            j.right = right

            left = Mondrian_Node(0.0, [true,false,false])
            left.parent = j
            left.Θ = Θᴸ
            left.m = mean(Labels[Dataᴸ])
            left.v = var(Labels[Dataᴸ])
            if isnan(left.v)
                left.v = j.v
            end
            j.left = left

            # recurse
            Sample_Mondrian_Block!(left, get(left.Θ), λ, Tree, Data[Dataᴸ,:], Labels[Dataᴸ],min_samples_split)
            Sample_Mondrian_Block!(right, get(right.Θ),λ, Tree, Data[Dataᴿ,:], Labels[Dataᴿ],min_samples_split)
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

function predict_reg_batch{X<:AbstractArray{Float64,N} where N}(
                           MT::Mondrian_Tree,
                           Data::X)
    P = zeros(size(Data,1))
    for i in 1:size(Data,1)
        P[i] = predict_reg(MT,Data[i])
    end
    return P
end
