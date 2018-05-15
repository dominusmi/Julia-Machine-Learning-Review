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
        if j.node_type[3]==true                                             ## check if we replace the root
            j_wave=Mondrian_Node(E,[false,false,true],d,x,Θ,zeros(size(get(j.tab))) ,zeros(size(get(j.c))),zeros(size(get(j.Gₚ))))
            j.node_type=[true,false,false]
        else
            j_wave=Mondrian_Node(parent=get(j.parent),get(j.parent).τ+E,[true,false,false],d,x,Θ,get(j.parent).tab,get(j.parent).c,get(j.parent).Gₚ)
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

                if get(j.Θ.Intervals[2][d])> ζ    #adapt box of j
                    j.Θ.Intervals[2][d] = ζ
                end
            j_prime.Θ = j_wave.Θ
            j_wave.Θ.Intervals[1][d] = ζ
            Sample_Mondrian_Block!(j_prime,get(j_prime.Θ),λ,T,D[1],D[2])
        else
            j_wave.left = j_prime
            j_wave.right = j

            if get(j.Θ.Intervals[1][d])< ζ
                j.Θ.Intervals[1][d] = ζ
            end
            j_prime.Θ = j_wave.Θ
            j_wave.Θ.Intervals[2][d] = ζ
            Sample_Mondrian_Block!(j_prime,get(j_prime.Θ),λ,T,D[1],D[2])
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
