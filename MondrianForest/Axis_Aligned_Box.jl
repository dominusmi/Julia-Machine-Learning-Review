using Distributions

# stores a series of D intervals
# representing
# [a₁,b₁]x[a₂,b₂]x⋯x[a_D, b_D]
mutable struct Axis_Aligned_Box
    Intervals::AbstractArray{Float64,N} where N
    D::Int
end

# constructor with intervals
function Axis_Aligned_Box{X<:AbstractArray{Float64,N} where N}(Intervals::X)
    Θ = Axis_Aligned_Box(Intervals, size(Intervals,1))
    return Θ
end

# copy constructor
function copy(Θ::Axis_Aligned_Box)
    return Axis_Aligned_Box(Base.copy(Θ.Intervals))
end

# ∑ᵢ₌₁ᴰ[lᵢ,uᵢ]
function Linear_dimension(Θ::Axis_Aligned_Box)
    s = 0
    for i in 1:Θ.D
        s += Θ.Intervals[i,2]-Θ.Intervals[i,1]
    end
    return s
end

# for a data matrix NxD, of N
# instances each with D dimensions
# this creates the Intervals
# [l₁,u₁],…,[l_D, u_D]
# of the dimesion wise
# min and max of the data
function get_intervals{X<:AbstractArray{Float64,N} where N}(Data::X)
    intervals = zeros(size(Data,2),2)
    for i in 1:size(Data,2)
        l = minimum(Data[:,i])
        u = maximum(Data[:,i])
        intervals[i,:] = [l,u]
    end
    return intervals
end

# samples a dimension d and
# location of split in d from
# a set of upper-lower intervals
function sample_split_dimension(Θ::Axis_Aligned_Box)
    p_k = zeros(Θ.D)
    for i in 1:Θ.D
        p_k[i] = Θ.Intervals[i,2]-Θ.Intervals[i,1]
    end
    p_k = p_k ./ Linear_dimension(Θ)
    d = rand(Categorical(p_k))
    x = rand(Uniform(Θ.Intervals[d,1][1],Θ.Intervals[d,2][1]))
    return d,x
end
