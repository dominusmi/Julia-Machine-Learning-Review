#utils for Alg 3 and 4 (need to go in a file)

# ∑ᵢ₌₁ᴰmax(lⱼ-x,0)+max(x-uⱼ,0)
function Extended_dimension(Θ::Axis_Aligned_Box,datapoint) #input: boxes, datapoint (no label)
    s = 0
    for i in 1:Θ.D
        s += max(Θ.Intervals[i,1]-datapoint[i],0)+max(datapoint[i]-Θ.Intervals[i,2],0)
    end
    return s
end

function sample_extended_split_dimension(Θ,datapoint) #input: boxes, datapoint (no label)
    p_k = zeros(Θ.D)
    for i in 1:Θ.D
        p_k[i] = max(Θ.Intervals[i,1]-datapoint[i],0)+max(datapoint[i]-Θ.Intervals[i,2],0)
    end
    p_k = p_k ./ Extended_dimension(Θ,datapoint)  #create the probability array
    d = rand(Categorical(p_k))        # sample from it

    if datapoint[d] > Θ.Intervals[d,2]
    x = rand(Uniform(Θ.Intervals[d,2][1],datapoint[d]))#sample the split loc. uniformly from that interval
    else
    x = rand(Uniform(datapoint[d],Θ.Intervals[d,1][1]))
    end
    return d,x
end

function Mondrian_Node(parent::Nullable{Mondrian_Node},τ::Float64,node_type::Array{Bool,1},δ::Nullable{Int},ζ::Nullable{Float64},Θ::Nullable{Axis_Aligned_Box},tab::Array{Int}, c::Array{Int}, Gₚ::Array{Float64})  #constructor used in Alg4 (Extended_Mond)
    N = Mondrian_Node(parent,
                      left::Nullable{Mondrian_Node},
                      right::Nullable{Mondrian_Node},
                      τ,node_type,
                      δ,
                      ζ,
                      Θ,
                      tab,
                      c,
                      Gₚ,
                      Array{Int}())
    return N
end


function Mondrian_Node(τ::Float64,node_type::Array{Bool,1},δ::Nullable{Int},ζ::Nullable{Float64},Θ::Nullable{Axis_Aligned_Box},tab::Array{Int},c::Array{Int},Gₚ::Array{Float64})  #constructor used in Alg4
    N = Mondrian_Node(Nullable{Mondrian_Node}(),
                      left::Nullable{Mondrian_Node},
                      right::Nullable{Mondrian_Node},
                      left,
                      right,
                      τ,node_type,
                      δ,
                      ζ,
                      Θ,
                      tab,
                      c,
                      Gₚ,
                      Array{Int}())
    return N
end

function update_intervals(Θ::Axis_Aligned_Box,D) #take θ and Datapoint as a tuple
    Intervals = Θ.Intervals
   for i=1:length(Θ.Intervals[:,1])
       l=min(Intervals[i,1],D[1][i])
       u=max(Intervals[i,2],D[1][i])
       Intervals[i,:]=[l,u]
    end
    Θ = Axis_Aligned_Box(Intervals,Θ.D)
    return Θ
end
