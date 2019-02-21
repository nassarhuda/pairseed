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
  myalpha::Float64, #PageRank's alpha or HK's t (set alpha = 0.8 or t = 15.0)
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
  # tao = 0.8
  # myalpha = 0.8
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

pairseed_on_collapsed_network(C,n,v1,v2) = pairseed_on_collapsed_network(C,n,v1,v2,"pr",0.8)


#=  4. Alternating diffusion method =#
function impTV(A,x)
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
	v = zeros(nA)
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

#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
# some standard functions
function general_double_seed_idea(M,
        community_sizes::Vector{Int64},
        triangles::String, #can be "triangle","wedge", or "edge"
        percentage_of_experiments::Float64, #0.5 to run a random set of half of the experiments, 1.0 for all
        myalpha::Float64, #PageRank's alpha
        A
    )
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

        xsol,xsol1,xsol2 = double_seed(Mc,i,jm,myalpha)#,"heatkernel")
        
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

function general_double_seed_idea_all_methods(M,
        community_sizes::Vector{Int64},
        triangles::String, #can be "triangle","wedge", or "edge"
        percentage_of_experiments::Float64, #0.5 to run a random set of half of the experiments, 1.0 for all
        myalpha::Float64, #PageRank's alpha
        A
    )
    n = size(A,1)
    i1,j1,TR = find_edges_of_tris(M)
    total_experiments = length(i1)
    experiments_to_run = floor(Int,percentage_of_experiments*total_experiments)
    experimentsids = sample(1:total_experiments,experiments_to_run,replace=false)
    tensor_ids_sample = hcat(i1[experimentsids],j1[experimentsids])
    TR = TR[experimentsids,:]
    AUC = zeros(experiments_to_run,11)
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

        xd_hk,xs1_hk,xs2_hk = double_seed(Mc,i,jm,15.0,"hk")
        xd_pr,xs1_pr,xs2_pr = double_seed(Mc,i,jm,myalpha,"pr")
        if size(Mc,1) >= 200
          Mc_Collapse = collapse_network(Mc)
          xcollapse1_hk,xcollapse2_hk = pairseed_on_collapsed_network(Mc_Collapse,n,i,jm,"hk",15.0)
          xcollapse1_pr,xcollapse2_pr = pairseed_on_collapsed_network(Mc_Collapse,n,i,jm,"pr",myalpha)
        else
          xcollapse1_hk = xcollapse2_hk = xcollapse1_pr = xcollapse2_pr =  zeros(community_sizes[3])
        end
        xalternate = pairseed_alternate(Mc,i,jm,myalpha,10)
    

        xk1 = xd_hk[community_sizes[1]+community_sizes[2]+1:end]
        xk2 = xs1_hk[community_sizes[1]+community_sizes[2]+1:end]
        xk3 = xs2_hk[community_sizes[1]+community_sizes[2]+1:end]

        xk4 = xd_pr[community_sizes[1]+community_sizes[2]+1:end]
        xk5 = xs1_pr[community_sizes[1]+community_sizes[2]+1:end]
        xk6 = xs2_pr[community_sizes[1]+community_sizes[2]+1:end]

        xk7 = xcollapse1_hk[community_sizes[1]+community_sizes[2]+1:end]
        xk8 = xcollapse2_hk[community_sizes[1]+community_sizes[2]+1:end]
        xk9 = xcollapse1_pr[community_sizes[1]+community_sizes[2]+1:end]
        xk10 = xcollapse2_pr[community_sizes[1]+community_sizes[2]+1:end]

        xk11 = xalternate[community_sizes[1]+community_sizes[2]+1:end]

        tpr,fpr,auc = calc_AUC_new(xrefreal,xk1); AUC[ii,1] = auc
        tpr,fpr,auc = calc_AUC_new(xrefreal,xk2); AUC[ii,2] = auc
        tpr,fpr,auc = calc_AUC_new(xrefreal,xk3); AUC[ii,3] = auc
        tpr,fpr,auc = calc_AUC_new(xrefreal,xk4); AUC[ii,4] = auc
        tpr,fpr,auc = calc_AUC_new(xrefreal,xk5); AUC[ii,5] = auc
        tpr,fpr,auc = calc_AUC_new(xrefreal,xk6); AUC[ii,6] = auc
        sum(xk7) == 0 ? auc = 0 : auc = calc_AUC_new(xrefreal,xk7)[3]; AUC[ii,7] = auc
        sum(xk8) == 0 ? auc = 0 : auc = calc_AUC_new(xrefreal,xk8)[3]; AUC[ii,8] = auc
        sum(xk9) == 0 ? auc = 0 : auc = calc_AUC_new(xrefreal,xk9)[3]; AUC[ii,9] = auc
        sum(xk10) == 0 ? auc = 0 : auc = calc_AUC_new(xrefreal,xk10)[3]; AUC[ii,10] = auc
        tpr,fpr,auc = calc_AUC_new(xrefreal,xk11); AUC[ii,11] = auc

    end
    return AUC
end