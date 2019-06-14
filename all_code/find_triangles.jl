function find_triangles(A::SparseMatrixCSC{T,Int64}) where T
    return find_triangles(MatrixNetwork(A));
end

function find_triangles(A::MatrixNetwork)
    return find_triangles(A, true, true);
end

struct graph_triangle
    node1::Int
    node2::Int
    node3::Int
end

function find_triangles(A::MatrixNetwork,weighted::Bool,normalized::Bool)
    donorm = true
    usew = true
    if !normalized
        donorm = false
    end
    if !weighted
        usew = false
    end

    if is_undirected(A) == false
        error("Only undirected (symmetric) inputs are allowed")
    end
    
    rp,ci,ai = (A.rp,A.ci,A.vals)
    if typeof(findfirst(ai.<0)) != Nothing
        error("only positive edge weights allowed")
    end
    return find_triangles_phase2(donorm,rp,ci,ai,usew)
end

function find_triangles_phase2(donorm::Bool,rp::Vector{Int64},ci::Vector{Int64},
                                       ai::Vector{T}, usew::Bool) where T
    n = length(rp) - 1
    cc = Vector{Float64}(undef,n)
    curccvec = Vector{Float64}(undef,n)
    ind = zeros(Bool,n)
    cache = zeros(Float64,usew ? n : 0)

    all_triangles = Array{Int}(undef,0)

    @inbounds for v = 1:n
        for rpi = rp[v]:rp[v+1]-1
            w = ci[rpi]
            if w > v
                ind[w] = 1
            end
        end
        d = rp[v+1]-rp[v]
        # run two steps of bfs to try and find triangles. 
        for rpi = rp[v]:rp[v+1]-1
            w = ci[rpi]
            if v == w
                d = d-1
                continue
            end #discount self-loop
            istart = rp[w]
            iend = rp[w+1]-1
            if usew # as of 2016-10-04, this arrangement with outer if was better
                for rpi2 = istart:iend
                    x = ci[rpi2]
                    if ind[x] && x > w && w > v
                        append!(all_triangles,[v,w,x])
                    end
                end
            else
                for rpi2 = istart:iend
                    x = ci[rpi2]
                    if ind[x] && x > w && w > v # 
                        append!(all_triangles,[v,w,x])
                    end
                end
            end                 
        end
        
        for rpi = rp[v]:rp[v+1]-1
            ind[ci[rpi]] = 0
        end # reset indicator
    end
    all_triangles = reshape(all_triangles,3,:)
    return all_triangles
end 
