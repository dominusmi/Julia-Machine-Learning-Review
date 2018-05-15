#Go at implementing Alg 3

function Extend_Mondrian_Tree!(T,λ,D) #Mondrian Tree, λ, D -> new training instance as a tuple
    ϵ=get(T.root)
    Extend_Mondrian_Block!(T,λ,ϵ,D) #Input: Mondrian Tree, lifetime, rootnode, new training instance
    return T
end

function Extend_Mondrian_Block!(T,λ,j,D) # Input: Mondrian Tree, lifetime, node, new training instance as a tuple
    E = rand(Exponential(1/Extended_dimension(get(j.Θ),D[1])))
    if j.node_type[3]==true #if its a root
        τₚ = 0
    else
        τₚ = (get(j.parent)).τ
    end
    # if split occured in time
    if τₚ + E < j.τ
        d,x= sample_extended_split_dimension(get(j.Θ),D[1])
        Θ = update_intervals(get(j.Θ),D)
        if j.node_type[3]==true  ## check if we replace the root
            j_wave=Mondrian_Node(E,[true,false,false])
	    j_wave.δ = d
        j_wave.ζ = x
        j_wave.Θ = Θ
        j_wave.tab = zeros(size(get(j.tab)))
        j_wave.c = zeros(size(get(j.c)))
        j_wave.Gₚ = zeros(size(get(j.Gₚ)))
        else
            j_wave=Mondrian_Node(get(j.parent).τ+E,[true,false,false])
	    j_wave.parent = j.parent
	    j_wave.δ = d
        j_wave.ζ = x
        j_wave.Θ = Θ
        j_wave.tab = get(j.parent).tab
        j_wave.c = get(j.parent).c
        j_wave.Gₚ = get(j.parent).Gₚ
            if j == get(j.parent).left
                get(j.parent).left = j_wave
            else
                get(j.parent).right = j_wave
            end
        end

        j.parent=j_wave
        j_prime = Mondrian_Node(0.0, [true,false,false])   #initialise new sibling to j
        j_prime.parent = j_wave
        j_prime.tab = zeros(size(j_wave.tab))
        j_prime.c = zeros(size(j_wave.tab))
        j_prime.Gₚ=zeros(size(j_wave.c,1))
        
        if D[1][d] > x
            j_wave.left = j
            j_wave.right = j_prime 
                if get(j.Θ).Intervals[d,2]> x    #adapt box of j
                    get(j.Θ).Intervals[d,2] = x
                end
            j_prime.Θ = j_wave.Θ
            get(j_wave.Θ).Intervals[d,1] = x
            A=zeros(1,length(D[1]))
            A[:,:]=D[1]
            Sample_Mondrian_Block!(j_prime,get(j_prime.Θ),λ,T,A,[D[2]])
        else
            j_wave.left = j_prime
            j_wave.right = j
            if get(j.Θ).Intervals[d,1]< x
                get(j.Θ).Intervals[d,1]= x
            end
            j_prime.Θ = j_wave.Θ
            get(j_wave.Θ).Intervals[d,2] = x
            A=zeros(1,length(D[1]))
            A[:,:]=D[1]
            Sample_Mondrian_Block!(j_prime,get(j_prime.Θ),λ,T,A,[D[2]])
        end


    else
        Θ = update_intervals(get(j.Θ),D)
        j.Θ=Θ
        if j.node_type != [false,true,false]
            if D[1][get(j.δ)] < get(j.ζ)
                Extend_Mondrian_Block!(T,λ,get(j.left),D)
            else
                Extend_Mondrian_Block!(T,λ,get(j.right),D)
            end
        end
    end
end
