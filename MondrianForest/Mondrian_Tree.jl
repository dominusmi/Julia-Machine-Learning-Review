#### Mondrian Tree as in https://arxiv.org/abs/1406.2673
#### Lakshminarayanan, B., Roy, D.M. and Teh, Y.W., 2014. Mondrian forests: Efficient online random forests. In Advances in neural information processing systems (pp. 3140-3148).
#### Alogrithm numbers (e.g A1) are as in the above paper
using Distributions
# for mondrian process
include("Axis_Aligned_Box.jl");
mutable struct Mondrian_Node
    parent::Nullable{Mondrian_Node}
    left::Nullable{Mondrian_Node}
    right::Nullable{Mondrian_Node}
    τ::Float64                      # split time
    node_type::Array{Bool,1}        # node,leaf,root
    δ::Nullable{Int}                # split dimension
    ζ::Nullable{Float64}            # split position
    Θ::Nullable{Axis_Aligned_Box}   # data boxes B
    tab::Array{Int}                 # tables serving dish k Chinese restaurant process (CRP)
    c::Array{Int}                   # customers eating dish k, tab[k] = min(c[k],1) IKN approx
    Gₚ::Array{Float64}              # posterior mean (predictive probability)
    indices::Array{Int64}           # stores relevant data points dependent on Θ
end

# construction

function Mondrian_Node(τ::Float64, node_type::Array{Bool,1})
    N = Mondrian_Node(Nullable{Mondrian_Node}(),
                      Nullable{Mondrian_Node}(),
                      Nullable{Mondrian_Node}(),
                      τ,
                      node_type,
                      Nullable{Int}(),
                      Nullable{Float64}(),
                      Nullable{Axis_Aligned_Box}(),
                      Array{Int}(),
                      Array{Int}(),
                      Array{Float64}(),
                      Array{Int64}())
    return N
end

# only really need leaves + root directly
mutable struct Mondrian_Tree
    root::Nullable{Mondrian_Node}
    leaves::Array{Mondrian_Node,1}
end

function Mondrian_Tree()
    return Mondrian_Tree(Nullable{Mondrian_Node}(),Array{Mondrian_Node,1}())
end

function Mondrian_Tree(N::Mondrian_Node)
    return Mondrian_Tree(N,Array{Mondrian_Node,1}())
end

function get_count(j::Mondrian_Node, Y::Array{Int64}, class_num::Int64)
    j.tab = zeros(class_num)
    j.c = zeros(class_num)
    for i in 1:class_num
        j.c[i] = length(Y[Y.==i])
        j.tab[i] = min(j.c[i],1)
    end
end

function Sample_Mondrian_Tree!(Tree::Mondrian_Tree,
                               λ::Float64,
                               X::Array{Float64,N} where N,
                               Y::Array{Int64})
    # initialise the tree
    e = Mondrian_Node(0.0,[false,false,true])
    Tree.root = e
    Θ = Axis_Aligned_Box(get_intervals(X))
    e.Θ = Θ
    get_count(e,Y, 2)
    e.Gₚ = zeros(2)
    Sample_Mondrian_Block!(e, Θ, λ, Tree, X, Y)
    return Tree
end

function Sample_Mondrian_Block!(j::Mondrian_Node,
                                Θ::Axis_Aligned_Box,
                                λ::Float64,
                                Tree::Mondrian_Tree,
                                X::Array{Float64,N} where N,
                                Y::Array{Int64})
    # paused mondrian check
    # should be one for pure targets
    if sum(j.c .> 0) == 1
        j.τ = λ
    else
        # not paused, sample the time
        E = rand(Exponential(1/Linear_dimension(Θ)))
        if j.node_type[3]==true
            τₚ = 0
        else
            τₚ = (get(j.parent)).τ
        end
        j.τ = τₚ+E
    end
    # if pausing should fall to the next else, other wise split is valid
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
            get_count(right, Y[Xᴿ], length(j.c))
            right.Gₚ=zeros(size(j.c,1))
            j.right = right

            left = Mondrian_Node(0.0, [true,false,false])
            left.parent = j
            left.Θ = Θᴸ
            get_count(left, Y[Xᴸ], length(j.c))
            left.Gₚ = zeros(size(j.c,1))
            j.left = left

            # recurse
            Sample_Mondrian_Block!(left, get(left.Θ), λ, Tree, X[Xᴸ,:], Y[Xᴸ])
            Sample_Mondrian_Block!(right, get(right.Θ),λ, Tree, X[Xᴿ,:], Y[Xᴿ])
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

# only check indices against the changed dimension CF lines 93-97
function get_data_indices(Θ::Axis_Aligned_Box, X::Array{Float64,N} where N, dim::Int64)
    # this function cause large memory allocation according
    # to @time but the system does not record any
    # large memory allocation -> ram does not get increased
    # at all!
    indices = []
    for i in 1:size(X,1)
        if !(X[i,dim] < Θ.Intervals[dim,1] || X[i,dim] > Θ.Intervals[dim,2])
            push!(indices,i)
        end
    end
    return indices
end
# returns any data from D contained in the boxes of Θ
function get_data_indices(Θ::Axis_Aligned_Box, X::Array{Float64,N} where N)
    # this function cause large memory allocation according
    # to @time but the system does not record any
    # large memory allocation -> ram does not get increased
    # at all!
    indices = []
    include = false
    for i in 1:size(X,1)
        for j in 1:size(Θ.Intervals,1)
            if (X[i,j] < Θ.Intervals[j,1] || X[i,j] > Θ.Intervals[j,2])
                include = false
                break
            end
            include = true
        end
        if (include)
            push!(indices, i)
        end
    end
    return indices
end

# gamma is usually set to 10*dimensionality
# which is done above somewhere
function compute_predictive_posterior_distribution!(Tree::Mondrian_Tree,
                                                    γ::Real)
    J = [get(Tree.root)]
    while (size(J,1) != 0)
        j = shift!(J)
        if (j.node_type[3]==true)
            p = ones(length(j.c))./length(j.c)
            d = exp(-γ*(j.τ))
        else
            d = exp(-γ*(j.τ-get(j.parent).τ))
            p = get(j.parent).Gₚ
        end
        for k in 1:length(j.c)
            j.Gₚ[k] = (1/(sum(j.c)))*(j.c[k]-d*j.tab[k]+d*sum(j.tab)*p[k])
        end
        if sum(j.node_type)==0
            break
        end
        if j.node_type[2] == false
            push!(J, get(j.left))
            push!(J, get(j.right))
        end
    end
end

# predict te class probs
function predict!(Tree::Mondrian_Tree,
                  x::Array{Float64},
                  γ::Real)
    # the algorithm requires computing an expectation
    # empirically
    function expected_discount(nⱼ, Δⱼ,γ=1)
        Δ = rand(Truncated(Exponential(1/nⱼ),0,Δⱼ),10000)
        return mean(exp.(-γ*Δ))
    end
    j = get(Tree.root)
    not_sep = 1
    s = zeros(size(j.c,1))
    while true
        if (sum(j.node_type)==0)
            break
        end
        if (j.node_type[3] == true)
            Δⱼ = j.τ
        else
            Δⱼ = j.τ - get(j.parent).τ
        end
        nⱼ=0
        for d in size(x,1)
            nⱼ += max(x[d]-get(j.Θ).Intervals[d,2],0) + max(get(j.Θ).Intervals[d,1]-x[d],0)
        end
        pⱼ = 1-exp(Δⱼ*nⱼ)
        # yes this part does add nodes to the tree!
        # although i've never seen it called...
        if pⱼ > 0
            # x branches
            jₓ = Mondrian_Node()
            if (j == get(j.parent).left)
                get(j.parent).left = jₓ
            else
                get(j.parent).right = jₓ
            end
            jₓ.parent = get(j.parent)
            j.parent = jₓ
            jₓ.left = j
            jₓ.right = Mondrian_Node()
            d = expected_discount(nⱼ, Δⱼ, γ)
            for k in 1:length(j.c)
                jₓ.c[k] = min(j.c[k],1)
            end
            jₓ.tab = jₓ.c
            for k in 1:length(jₓ.c)
                jₓ.Gₚ[k] = 1/(sum(jₓ.c))*(jₓ.c[k] - d*jₓ.tab[k]+d*sum(jₓ.tab)*get(jₓ.parent).Gₚ[k])
            end
            for k in 1:length(s)
                s[k] += not_sep*(1-pⱼ)*jₓ.Gₚ[k]
            end
        end
        if j.node_type[2] == true
            for k in 1:length(s)
                s[k] += not_sep*(1-pⱼ)*j.Gₚ[k]
            end
            return s
        else
            not_sep = not_sep*(1-pⱼ)
            if x[get(j.δ)] <= get(j.ζ)
                j = get(j.left)
            else
                j = get(j.right)
            end
        end
    end
end
