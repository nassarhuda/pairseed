# methods.jl
# this file includes the methods used for pairwise predictions

# 1. Pairseed with heatkernel/pagerank options
# 2. Single seed with heatkernel/pagerank options
# 3. Collapsed network with heatkernel/pagerank options
# 4. Alternating diffusion method

# hk == heatkernel
# pr == pagerank
# er == erdos renyi
# pa == preferential attachement

#= 1. and 2. are combined =#
#= 1. Pairseed with heatkernel/pagerank options =#
#= 2. Single seed with heatkernel/pagerank options =#
# general_double_seed_idea(M,community_sizes,triangles,percentage_of_experiments,myalpha,method)
# double_seed(A,v1,v2,myalpha,method)
# double_seed(A,v1,v2,myalpha) # defaults to pagerank

function general_double_seed_idea(M,
        community_sizes::Vector{Int64},
        triangles::String, #can be "triangle","wedge", or "edge"
        percentage_of_experiments::Float64, #0.5 to run a random set of half of the experiments, 1.0 for all
        myalpha::Float64, #PageRank's alpha
        method::String)
    i1,j1,TR = find_edges_of_tris(M)
    total_experiments = length(i1)
    experiments_to_run = floor(Int,percentage_of_experiments*total_experiments)
    experimentsids = sample(1:total_experiments,experiments_to_run,replace=false)
    tensor_ids_sample = hcat(i1[experimentsids],j1[experimentsids])
    TR = TR[experimentsids,:]
    all_aucs_new = zeros(experiments_to_run,5)
    for ii = 1:experiments_to_run
        #@show ii, experiments_to_run
        v = spzeros(size(A,2))
        vi = copy(v)
        vj = copy(v)
        rw = tensor_ids_sample[ii,:]
        i = rw[1]
        j = rw[2]
        jm = n0[1] + j
        
        v[i] = 0.5
        v[jm] = 0.5
        vi[i] = 1
        vj[jm] = 1
        
        Mc = copy(A)
        ism = TR[ii,:]
        k_ids = ism.nzind
        xrefreal = zeros(Int64,community_sizes[3])
        xrefreal[k_ids] .= 1
        k_ids_mapped = k_ids .+ n0[1] .+ n0[2]
        
        if triangles == "triangle"
            #skip
        elseif triangles == "wedge"
            #flip a coin
            randval = rand()
            if randval>=0.5    
                Mc[i,k_ids_mapped] .= 0
                Mc[k_ids_mapped,i] .= 0
            else
                Mc[jm,k_ids_mapped] .= 0
                Mc[k_ids_mapped,jm] .= 0
            end
        elseif triangles == "edge"
            Mc[i,k_ids_mapped] .= 0
            Mc[k_ids_mapped,i] .= 0
            Mc[jm,k_ids_mapped] .= 0
            Mc[k_ids_mapped,jm] .= 0    
        else
            error("The only options for triangles is \"triangle\",\"wedge\",or \"edge\"")
        end
        # xsol = seeded_pagerank(Mc,myalpha,v)
        # xsol1 = seeded_pagerank(Mc,myalpha,vi)
        # xsol2 = seeded_pagerank(Mc,myalpha,vj)

        xsol,xsol1,xsol2 = double_seed(Mc,i,jm,myalpha,method)
        
        xsol3 = xsol1.*xsol2 
        xsol4 = xsol1.+xsol2
    
        xk = xsol[community_sizes[1]+community_sizes[2]+1:end]#DS
        xk1 = xsol1[community_sizes[1]+community_sizes[2]+1:end]#SS
        xk2 = xsol2[community_sizes[1]+community_sizes[2]+1:end]
        xk3 = xsol3[community_sizes[1]+community_sizes[2]+1:end]#SS-A
        xk4 = xsol4[community_sizes[1]+community_sizes[2]+1:end]#SS-O

        fpr,tpr,auc = calc_AUC_new(xrefreal,xk)
        fpr,tpr,auc1 = calc_AUC_new(xrefreal,xk1)
        fpr,tpr,auc2 = calc_AUC_new(xrefreal,xk2)
        fpr,tpr,auc3 = calc_AUC_new(xrefreal,xk3)
        fpr,tpr,auc4 = calc_AUC_new(xrefreal,xk4)
        all_aucs_new[ii,:] = [auc,auc1,auc2,auc3,auc4]
    end
    return all_aucs_new
end

function double_seed(
  A::SparseMatrixCSC{Int64,Int64},
  v1::Int,
  v2::Int,
  myalpha::Float64, #PageRank's alpha 
  method::String
  )

  v = spzeros(size(A,2))
  vi = copy(v)
  vj = copy(v)

  v[v1] = 0.5
  v[v2] = 0.5
  vi[v1] = 1
  vj[v2] = 1

  if method == "heatkernel" || method == "hk"
    xsol = seeded_stochastic_heat_kernel(MatrixNetwork(A),myalpha,v)
    xsol1 = seeded_stochastic_heat_kernel(MatrixNetwork(A),myalpha,vi)
    xsol2 = seeded_stochastic_heat_kernel(MatrixNetwork(A),myalpha,vj)
  elseif method == "pagerank" || method == "pr"
    xsol = seeded_pagerank(A,myalpha,v)
    xsol1 = seeded_pagerank(A,myalpha,vi)
    xsol2 = seeded_pagerank(A,myalpha,vj)
  else
    error("method should be heatkernel or pagerank")
  end

  return xsol,xsol1,xsol2
end

double_seed(A::SparseMatrixCSC{Int64,Int64},v1::Int,v2::Int,myalpha::Float64) = double_seed(A,v1,v2,myalpha,"pagerank")


#= 3. Collapsed network with heatkernel/pagerank options =#
function collapse_network(A)
  # i1,j1,TR = find_edges_of_tris(M);
  ei = Array{Int64}(undef,0)
  ej = Array{Int64}(undef,0)
  n = size(A,1)
  # new dimension is n^2 + n
  mytriangles = triangles(A;symmetries = true)
  for tri in mytriangles
    r,c,k = tri.v1,tri.v2,tri.v3
    node1 = n*(c-1) + r
    node2 = n^2 + k
    push!(ei,node1)
    push!(ej,node2)
  end
  triangles_collapsed = sparse(ei,ej,1,n^2+n,n^2+n)
  triangles_collapsed = max.(triangles_collapsed,triangles_collapsed')
  # triangles_collapsed[n^2+1:n^2+n,n^2+1:n^2+n] .= A
  return triangles_collapsed
end

function pairseed_on_collapsed_network(C,n,v1,v2,method,p)
	s1 = n*(v1-1) + v2
	s2 = n*(v2-1) + v1
	if method == "heatkernel" || method == "hk"
		xsol1 = seeded_stochastic_heat_kernel(MatrixNetwork(C),p,s1)
		xsol2 = seeded_stochastic_heat_kernel(MatrixNetwork(C),p,s2)
    elseif method == "pagerank" || method == "pr"
		  xsol1 = seeded_pagerank(C,p,s1)
    	xsol2 = seeded_pagerank(C,p,s2)
    else
    	error("method should be heatkernel or pagerank")
    end
    return xsol1[end-n+1:end],xsol2[end-n+1:end]
end

pairseed_on_collapsed_network(C,n,v1,v2) = pairseed_on_collapsed_network(C,n,v1,v2,"pr",0.85)

#=  4. Alternating diffusion method =#
function impTV_old_naive(A,x)
    rp,ci,ai = A.colptr,A.rowval,A.nzval # this should be removed after error in triangles.jl is resolved
    n = size(A,1)
    X = spzeros(n,n)
    for i = 1:n
      if length(MatrixNetworks.find_first_triangle(i,rp,ci,A.n))>2
      i_triangles = triangles(A,i)
      if !isempty(i_triangles)
        for itri in i_triangles
            i = itri.v1
            j1 = itri.v2
            j2 = itri.v3
            X[i,j1] += x[j2]
            X[i,j2] += x[j1]
        end
      end
      end
    end
    return X
end

function impTV(A,x)
    rp,ci,ai = A.colptr,A.rowval,A.nzval # this should be removed after error in triangles.jl is resolved
    n = size(A,1)
    X = spzeros(n,n)
    nonzero_x = x.nzind
    for i = nonzero_x #1:n
      if length(MatrixNetworks.find_first_triangle(i,rp,ci,A.n))>2
      i_triangles = triangles(A,i)
      if !isempty(i_triangles)
        for itri in i_triangles
            v1 = itri.v1
            v2 = itri.v2
            v3 = itri.v3
            X[v2,v3] += x[v1]
            X[v3,v2] += x[v1]
            # X[i,j1] += x[j2]
            # X[i,j2] += x[j1]
        end
      end
      end
    end
    return X
end

function edgeRank(A,x,alpha,maxiter)
    xref = copy(x)
    n = size(A,1)
    # first iteration
    X = impTV(A,x)
    X = X+A # the weights have changed now with higher weights for edges involved in "important" triangles
    X = _normout_rowstochastic(X)
    x = alpha*X'*x + (1-alpha)*xref
    for i = 2:maxiter
        X = impTV(A,x)
        X = X+A
        X = _normout_rowstochastic(X)
        x = alpha*X'*x + (1-alpha)*xref
    end
    X = impTV(A,x)
    return X,x
end

function pairseed_alternate(A,v1,v2,myalpha,maxiter)
	nA = size(A,1)
	v = spzeros(nA)
	v[v1] = 0.5
	v[v2] = 0.5
	X,x = edgeRank(A,v,myalpha,maxiter)
	return x
end
function pair_prediction_alternate(A,vi,myalpha,maxiter)
    nA = size(A,1)
    v = zeros(nA)
    v[vi] = 1
    X,x = edgeRank(A,v,myalpha,maxiter)
    return X
end

###########
function triproducts(T,x,y)
  z = similar(x)
  z .= 0
  for (ei,ej,ek) in T
    z[ei] += x[ej]*y[ek]
  end
  return z
end

function triproducts_iterative(T,x,y)
  z = similar(x)
  z .= 0
  for ti in T
    ei,ej,ek = ti.v1,ti.v2,ti.v3
    z[ei] += x[ej]*y[ek]
  end
  return z
end

function trpr(A, T, x0::Vector{Float64}, α, iter)
  # assumes A is a symmetric sparse matrix
  # T is a list of triangles,and fully symmetric.
  @assert(issymmetric(A))
  x = copy(x0)
  n = size(A,1)
  dg = vec(sum(A;dims=1))
  for i=1:iter
    dx = triproducts(T,x,ones(n))
    d = dx + dg
    id = map!(x -> 1.0/x, d, d)
    y = x.*id
    xn = α.*triproducts(T,x,y) + α.*(A*y) + (1-α).*x0
    x = xn 
  end
  return x
end

function trpr(A,x0,α,maxit)
    T = collect(triangles(A;symmetries=true))
    return trpr(A,T,x0,α,maxit)
end

function triproducts_iterator(A,x,y)
  z = similar(x)
  z .= 0
  T = triangles(A;symmetries=true)
  for tri in T
    z[tri.v1] += x[tri.v2]*y[tri.v3]
  end
  return z
end
function trpr_iterator(A, x0::Vector{Float64}, α, iter)
  # assumes A is a symmetric sparse matrix
  # T is a list of triangles,and fully symmetric.
  @assert(issymmetric(A))
  x = copy(x0)
  n = size(A,1)
  dg = vec(sum(A;dims=1))
  for i=1:iter
    dx = triproducts_iterator(A,x,ones(n))
    d = dx + dg
    id = map!(x -> 1.0/x, d, d)
    y = x.*id
    xn = α.*triproducts_iterator(A,x,y) + α.*(A*y) + (1-α).*x0
    x = xn 
  end
  return x
end

function pairseed_trpr(A,T,v1,v2,myalpha,maxiter)
    nA = size(A,1)
    v = zeros(nA)
    v[v1] = 0.5
    v[v2] = 0.5
    x = trpr(A,T,v,myalpha,maxiter)
    return x
end

function pairseed_trpr(A,v1,v2,myalpha,maxiter)
    nA = size(A,1)
    v = zeros(nA)
    v[v1] = 0.5
    v[v2] = 0.5
    x = trpr_iterator(A,v,myalpha,maxiter)
    return x
end
##########################



#######################################################################################
# some ground truth - sanity predictions
function neighbors((u,v)::Tuple{Int64,Int64},A)
    return intersect(A[:,u].nzind,A[:,v].nzind)
end
function neighbors(u::Int64,A)
    return A[:,u].nzind
end
function AdamicAdar((u,v),w,A,degs)
    
    commonset = (A[:,u].*A[:,v].*A[:,w]).nzind #intersect(neighbors((u,v),A),neighbors(w,A))
    # @show commonset
    neighbors_sets = degs[commonset]
    # @show neighbors_sets
    return sum(1 ./ (log10.(neighbors_sets)))
end
function AdamicAdar_single((u,v),w,A,degs)
    
    commonset = (A[:,u].*A[:,w]).nzind #intersect(neighbors((u,v),A),neighbors(w,A))
    neighbors_sets = degs[commonset]
    A1 = sum(1 ./ (log10.(neighbors_sets)))

    commonset = (A[:,v].*A[:,w]).nzind #intersect(neighbors((u,v),A),neighbors(w,A))
    neighbors_sets = degs[commonset]
    A2 = sum(1 ./ (log10.(neighbors_sets)))

    # @show neighbors_sets
    return max(A1,A2),A1*A2
end
function AdamicAdar_union((u,v),w,A,degs)
    # @show u,v
    commonset = ((A[:,u].+A[:,v]).*A[:,w]).nzind #intersect(neighbors((u,v),A),neighbors(w,A))
    # @show commonset
    neighbors_sets = degs[commonset]
    # @show neighbors_sets
    return sum(1 ./ (log10.(neighbors_sets)))
end

function AdamicAdar_union_neighbors(allnodes,w,A,degs)
    # @show u,v
    commonset = ((sum(A[allnodes,:],dims=1)).*A[w,:]).nzind
    # ((A[:,u].+A[:,v]).*A[:,w]).nzind #intersect(neighbors((u,v),A),neighbors(w,A))
    # @show commonset
    neighbors_sets = degs[commonset]
    # @show neighbors_sets
    return sum(1 ./ (log10.(neighbors_sets)))
end

function pref_attach((u,v),w,A,degs)
    n1 = nnz(A[:,u].*A[:,v]) #jlength(neighbors((u,v),A))
    n2 = degs[w]
    # return n1*n2
    return n2
end
function pref_attach_union((u,v),w,A,degs)
    n1 = nnz(A[:,u].+A[:,v]) #jlength(neighbors((u,v),A))
    n2 = degs[w]
    # return n1*
    return n2
end
function Jaccard_Similarity((u,v),w,A)
    # n1 = neighbors((u,v),A)
    # n2 = neighbors(w,A)
    commonset = (A[:,u].*A[:,v].*A[:,w]).nzind #intersect(n1,n2)
    unionset = ((A[:,u].*A[:,v]).+A[:,w]).nzind #union(n1,n2)
    if length(unionset) != 0
        J = length(commonset)/length(unionset)
    else
        J = 0
    end
    return J
end

function Jaccard_Similarity_single((u,v),w,A)
    # n1 = neighbors((u,v),A)
    # n2 = neighbors(w,A)
    commonset = (A[:,u].*A[:,w]).nzind #intersect(n1,n2)
    unionset = (A[:,u].+A[:,w]).nzind #union(n1,n2)
    if length(unionset) != 0
        J = length(commonset)/length(unionset)
    else
        J = 0
    end

    commonset = (A[:,v].*A[:,w]).nzind #intersect(n1,n2)
    unionset = (A[:,v].+A[:,w]).nzind #union(n1,n2)
    if length(unionset) != 0
        F = length(commonset)/length(unionset)
    else
        F = 0
    end
    return max(F,J),F*J
end

function Jaccard_Similarity_with_union((u,v),w,A)
    # n1 = neighbors((u,v),A)
    # n2 = neighbors(w,A)
    commonset = ((A[:,u].+A[:,v]).*A[:,w]).nzind #intersect(n1,n2)
    unionset = ((A[:,u].+A[:,v]).+A[:,w]).nzind #union(n1,n2)
    if length(unionset) != 0
        J = length(commonset)/length(unionset)
    else
        @show "union was empty"
        J = 0
    end
    return J
end

function similarity_predict(
  A::SparseMatrixCSC{Int64,Int64},
  v1::Int,
  v2::Int,
  w_compare_to)
degs = sum(A,dims=2)[:]
    aa_distance = map(i->AdamicAdar((v1,v2),i,A,degs),w_compare_to)
    # aa_distance = aa_distance./sum(aa_distance)
    pf_distance = map(i->pref_attach((v1,v2),i,A,degs),w_compare_to)
    if maximum(pf_distance) != 0
        pf_distance = pf_distance./sum(pf_distance) # normalize
    end
    js_distance = map(i->Jaccard_Similarity((v1,v2),i,A),w_compare_to)
    # js_distance = js_distance./sum(js_distance)
    return aa_distance,pf_distance,js_distance
end

function similarity_predict_singles(
  A::SparseMatrixCSC{Int64,Int64},
  v1::Int,
  v2::Int,
  w_compare_to)
degs = sum(A,dims=2)[:]
    aa_distance_max = zeros(length(w_compare_to))
    aa_distance_mul = zeros(length(w_compare_to))
    js_distance_max = zeros(length(w_compare_to))
    js_distance_mul = zeros(length(w_compare_to))

    map(i->(aa_distance_max[i],aa_distance_mul[i]) = AdamicAdar_single((v1,v2),w_compare_to[i],A,degs),1:length(w_compare_to))
    map(i->(js_distance_max[i],js_distance_mul[i]) = Jaccard_Similarity_single((v1,v2),w_compare_to[i],A),1:length(w_compare_to))

    return aa_distance_max,aa_distance_mul,js_distance_max,js_distance_mul
end

function similarity_predict_withJS_union(
  A::SparseMatrixCSC{Int64,Int64},
  v1::Int,
  v2::Int,
  w_compare_to)
degs = sum(A,dims=2)[:]
    aa_distance = map(i->AdamicAdar((v1,v2),i,A,degs),w_compare_to)
    aa_distance_union = map(i->AdamicAdar_union((v1,v2),i,A,degs),w_compare_to)
    # aa_distance = aa_distance./sum(aa_distance)
    pf_distance = map(i->pref_attach((v1,v2),i,A,degs),w_compare_to)
    if maximum(pf_distance) != 0
        pf_distance = pf_distance./sum(pf_distance) # normalize
    end
    pf_distance_union = map(i->pref_attach_union((v1,v2),i,A,degs),w_compare_to)
    if maximum(pf_distance_union) != 0
        pf_distance_union = pf_distance_union./sum(pf_distance_union) # normalize
    end
    js_distance = map(i->Jaccard_Similarity((v1,v2),i,A),w_compare_to)
    js_with_union = map(i->Jaccard_Similarity_with_union((v1,v2),i,A),w_compare_to) 
    # js_distance = js_distance./sum(js_distance)
    return aa_distance,pf_distance,js_distance,aa_distance_union,pf_distance_union,js_with_union
end
#######################################################################################

#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################