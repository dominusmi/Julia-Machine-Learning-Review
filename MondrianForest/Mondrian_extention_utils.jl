#Utils for extending Mondrian Trees 

"""
`function Extended_dimension(Θ::Axis_Aligned_Box,X::Array{Float64})`

This function samples the rate ∑ᵢ₌₁ᴰmax(lⱼ-x,0)+max(x-uⱼ,0) with which we draw the parameter E from the exponential distribution in the function Extend_Mondrian_Block!. 

`Input`:  Axis Aligned Box Θ (Abstract type Axis_Aligned_Box), One dimensional array of features X (Array of Float64)

`Output`: Sampling rate s

`Files needed to run this function`: "Mondrian_extention.jl"

This function is one of the utility functions to run "Extend_Mondrian_Block!".


"""

function Extended_dimension(Θ::Axis_Aligned_Box,X::Array{Float64}) 
    # rate = ∑ᵢ₌₁ᴰmax(lⱼ-x,0)+max(x-uⱼ,0)
    
    s = 0
    for i in 1:Θ.D
        s += max(Θ.Intervals[i,1]-X[i],0)+max(X[i]-Θ.Intervals[i,2],0)  #compute the rate
    end
    return s
end

"""
`function sample_extended_split_dimension(Θ::Axis_Aligned_Box,X::Array{Float64})`

This function samples the split dimension and direction associated to the new node j_wave in the function "Extend_Mondrian_Block!"

`Input`: Axis Aligned Box Θ (abstract type Axis_Aligned_Box), One dimensional array of features X (Array of Float64)

`Output`: Split dimension d, split direction s

`Files needed to run this function`: "Mondrian_extention.jl"

This function is one of the utility functions to run "Extend_Mondrian_Block!".
"""

function sample_extended_split_dimension(Θ::Axis_Aligned_Box,X::Array{Float64}) 
    p_k = zeros(Θ.D)
    for i in 1:Θ.D
        p_k[i] = max(Θ.Intervals[i,1]-X[i],0)+max(X[i]-Θ.Intervals[i,2],0)  #compute max(lⱼ-x,0)+max(x-uⱼ,0)
    end
    p_k = p_k ./ Extended_dimension(Θ,X)                                #create the probability array
    d = rand(Categorical(p_k))                                          # sample from it (d with probability proportional to max(lⱼ-x,0)+max(x-uⱼ,0) )

    if X[d] > Θ.Intervals[d,2]  # choice of interval dependent on where the new datapoint is                                    
    x = rand(Uniform(Θ.Intervals[d,2][1],X[d]))  #sample the split loc. uniformly from that interval
    else
    x = rand(Uniform(X[d],Θ.Intervals[d,1][1])) #sample the split loc. uniformly from that interval
    end
    return d,x   #return split dim., split loc. 
end

"""
`function update_intervals(Θ::Axis_Aligned_Box,X::Array{Float64})`

This function computes the boxes associated to the new node j_wave. 

`Input`: Axis Aligned Box Θ (abstract type Axis_Aligned_Box), One dimensional array of features X (Array of Float64)

`Output`: Axis Aligned Box Θ (updated Intervals)

`Files needed to run this function`: "Mondrian_extention.jl"

This function is one of the utility functions to run "Extend_Mondrian_Block!".

"""
function update_intervals(Θ::Axis_Aligned_Box,X::Array{Float64})
    Intervals = Θ.Intervals                         # extract the intervals from Θ
   for i=1:length(Θ.Intervals[:,1])                 # update the intervals successively 
       l=min(Intervals[i,1],X[i])
       u=max(Intervals[i,2],X[i])
       Intervals[i,:]=[l,u]
    end
    Θ = Axis_Aligned_Box(Intervals,Θ.D)             # return updated intervals
    return Θ
end
