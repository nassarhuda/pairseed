# Pkgs that I need

using SparseArrays
using LinearAlgebra
using MatrixNetworks
using MatrixNetworks
using StatsBase
using Plots
using StatsPlots
using Statistics
using UnicodePlots
include("find_triangles.jl")
using PyCall

isnotnan = !isnan
isconnected(A) = scomponents(A).sizes[1] == size(A,1)

import SparseArrays.spzeros

spzeros(A1) = spzeros(size(A1,1),size(A1,2))
function spones(A)
    eei,eej,eev = findnz(A)
    return sparse(eei,eej,1,size(A,1),size(A,2))
end

function findtris(M,n0)
    H1 = (M[4]'*M[5]).*M[2]'
    H3 = (M[2]'*M[5]').*M[4]'
    H2 = (M[2]*M[4]').*M[5]'

    i1,j1,v1 = findnz(H1)
    tris = zeros(Int64,sum(Int,v1),3)
    tricounter = 1
    for i = 1:length(v1)
        tval = v1[i]
        # nval = [i1[i],j1[i]]
        tris[tricounter:tricounter+tval-1,1] .= i1[i]
        tris[tricounter:tricounter+tval-1,2] .= j1[i]
        tricounter = tricounter+tval
    end
    TR = H3[i1,:].*H2[j1,:]
    # See if there is a better way to do this:
    c = copy(TR')[:] # c2 = TR'[:]; @show isequal(c,c2)
    kk = c.nzind 
    kk = mod.(kk.-1,n0[3]).+1
    tris[:,3] = kk
    return tris
end

begin
"""
findin_index(x,y) returns a vector v of the same size as that of x
- where v[i] = index of the element x[i] in the vector y
- v[i] = 0 if x[i] does not exist in y
- assumption: If y does not consist of unique elements, the index returned is the last occurence
```
example:
    julia> x = [1,2,3,10,1,4];
    julia> y = [1,2,5,4,3];
    julia> findin_index(x,y)
    6-element Array{Int64,1}:
     1
     2
     5
     0
     1
     4
```
"""
function findin_index(x::Vector{T},y::Vector{T}) where T
  indices_in_y = zeros(Int64,length(x))
  already_exist = findall((in)(y), x)
  donot_exist = setdiff(1:length(x),already_exist)
  funcmap = i -> indices_in_y[findall(x.==y[i])] .= i
  lookfor_indices = findall((in)(x), y)
  map(funcmap,lookfor_indices)
  return indices_in_y
end
end
function ismember(A::Array{T,2},x::Vector{T},dims::Int) where T
  @assert (dims==1 || dims==2)
  # dims = 1 means we're looking at columns
  # dims = 2 means we're looking at rows
  if dims == 2
    sz = size(A,1)
    A = A'
  else
    sz = size(A,2)
  end
  ret = trues(sz)
  map(i->ret[i] = A[:,i] == vec(x),1:sz)
  return ret
end

@pyimport sklearn.metrics as metrics
function calc_AUC_pycall(Rtest,Xa)
  Rtest = vec(Rtest)
  Xa = vec(Xa)
  minNonZero = minimum(abs.(Xa[findall(Xa.!=0)]))
  Xa = Xa / minNonZero
  fpr,tpr,thresholds = metrics.roc_curve(Rtest,Xa)
  auc = metrics.auc(fpr,tpr)
  return tpr,fpr,auc
end

using MLBase
function calc_AUC_new(xref,xresult) #xref is ground truth
    r = MLBase.roc(xref,xresult)
    fpr = zeros(length(r))
    tpr = zeros(length(r))
    for i = length(r):-1:1
        fpr[length(r)-i+1] = false_positive_rate(r[i])
        tpr[length(r)-i+1] = true_positive_rate(r[i])
    end
    aucv = 0
    for i = 1:length(fpr)-1
        aucv = aucv + (fpr[i+1] - fpr[i])*(tpr[i] + tpr[i+1])/2
    end
    return tpr,fpr,aucv
end

function calc_triangle_closing_percent(x,k,i,j,addednodes,Ar)
    sorted_preds = sortperm(x,rev=true).+addednodes
    ci = zeros(length(x))
    for ki = 1:length(x)
        nki = sorted_preds[ki]
        ci[ki] = close_triangles(i,nki,Ar)
    end

    di = zeros(length(x))
    for ki = 1:length(x)
        nki = sorted_preds[ki]
        di[ki] = close_triangles(j,nki,Ar)
    end
    t1 = sum(ci[1:k])/(2*sum(ci))

    t2 = sum(di[1:k])/(2*sum(di))

    tri_percent = t1+t2
    return t1,t2,tri_percent
end

using Random
function split_train_test(R::SparseMatrixCSC{T,Int64},rho::Float64) where T
    if !(0<=rho<=1)
      error("function split_train_test: rho must be between 0 and 1. Read `? split_train_test` for more")
    end
    
    m,n = size(R)
    ei,ej,ev = findnz(triu(R))
    len = length(ev)
    seed = time()
    r = MersenneTwister(round(Int64,seed))
    a = randperm(r,len)
    nz = floor(Int,rho*len)
    p = a[1:nz]
    cp = setdiff(collect(1:len),p);

    Rtrain = sparse(ei[p],ej[p],ev[p],m,n)
    Rtrain = max.(Rtrain,Rtrain')
    Rtest = sparse(ei[cp],ej[cp],ev[cp],m,n)
    Rtest = max.(Rtest,Rtest')
    return Rtrain,Rtest
end

function split_train_test_keep_many_wedges(R,tao)
    if !(0<=tao<=1)
      error("function split_train_test: rho must be between 0 and 1. Read `? split_train_test` for more")
    end
    n = size(R,1)
    A = copy(R)
    A2 = A*A # length2 network
    A2 = A2-Diagonal(A2)
    T = spones(A2.*A) #every edge that is part of at least one triangle
    F = A-T # 1 if edge is part of no triangle at all
    Ftrain,Ftest = split_train_test(F,tao)
    # Ttrain,Ttest,Edges = split_train_test_keep_connected_wedges_helper(T,tao)
    # return Ttrain+Ftrain,Ttest+Ftest,Edges
    # A = A-Ftest
    Ctrain = spzeros(Int,size(A)...)
    Ctest = spzeros(Int,size(A)...)
    # @show nnz(Ftrain)/nnz(F)

    tris = collect(triangles(A))

    len = length(tris)
    seed = time()
    r = MersenneTwister(round(Int64,seed))
    a = randperm(r,len)
    nz = ceil(Int,tao/4*len)
    tris = tris[a]#[1:nz]
    
    # eitest = zeros(Int,2nz)
    # ejtest = zeros(Int,2nz)

    test_edge_count = 1
    opt = floor(Int,0.2*(nnz(R)/2) - nnz(Ftest)/2)
    @show opt
    for curtri_id = 1:length(tris)
        if test_edge_count > opt
            break
        end
        v1,v2,v3 = tris[curtri_id]

        # make sure it is still a triangle
        # if !(A[v1,v2] !=1 || A[v1,v3] != 1 || A[v2,v3] != 1)
        # ei1 = v1
        # ej1 = v2
        # ek1 = v3
        C = Ctrain+Ctest
        mys = [C[v1,v2],C[v2,v3],C[v1,v3]]
        # @show sum(mys)
        if sum(mys) == 0 # cases 0,1 # none of the three nodes were previously touched
            Ctrain[v1,v2] = 1
            Ctest[v2,v3] = 1; test_edge_count += 1
            Ctest[v1,v3] = 1; test_edge_count += 1
        elseif sum(mys) == 2 # two edges have been touched, only care about the case when both are in test
            mytest = [Ctest[v1,v2],Ctest[v2,v3],Ctest[v1,v3]]
            if sum(mytest) == 2
            cc = findfirst(mys.==0)
            if cc == 1
                Ctrain[v1,v2] = 1
                #v1,v2 train
                # Ctest[v2,v3] = 1
                # Ctest[v1,v3] = 1
            elseif cc == 2
                Ctrain[v2,v3] = 1
                # Ctest[v1,v2] = 1
                # Ctest[v1,v3] = 1
            elseif cc == 3
                Ctrain[v1,v3] = 1
                # Ctest[v1,v2] = 1
                # Ctest[v2,v3] = 1
            end
        end
        elseif sum(mys) == 1 #case 2 # only one edge has been used, 
            mytest = [Ctest[v1,v2],Ctest[v2,v3],Ctest[v1,v3]]
            cc = findfirst(mys.==1)
            if sum(mytest) == 1
            if cc == 1
                Ctrain[v2,v3] = 1
                Ctest[v1,v3] = 1; test_edge_count += 1
            elseif cc == 2
                Ctrain[v1,v2] = 1
                Ctest[v1,v3] = 1; test_edge_count += 1
            elseif cc == 3
                Ctrain[v1,v2] = 1
                Ctest[v2,v3] = 1; test_edge_count += 1
            end
            else
            if cc == 1
                Ctest[v2,v3] = 1; test_edge_count += 1
                Ctest[v1,v3] = 1; test_edge_count += 1
            elseif cc == 2
                Ctest[v1,v2] = 1; test_edge_count += 1
                Ctest[v1,v3] = 1; test_edge_count += 1
            elseif cc == 3
                Ctest[v1,v2] = 1; test_edge_count += 1
                Ctest[v2,v3] = 1; test_edge_count += 1
            end
        end

        end
        # if sum(mys) == 3 # everything is in training
        Ctrain = max.(Ctrain,Ctrain')
        Ctest = max.(Ctest,Ctest')
    end
    @show test_edge_count
    ei,ej,ev = findnz(triu(Ctrain))
    Edges = hcat(ei,ej)
    Mtrain = A-Ftest-Ctest
    Mtest = Ftest+Ctest
    return Mtrain,Mtest,Edges

end

function split_train_test_keep_connected_wedges_helper(R::SparseMatrixCSC{T,Int64},rho::Float64) where T
    if !(0<=rho<=1)
      error("function split_train_test: rho must be between 0 and 1. Read `? split_train_test` for more")
    end
    
    m,n = size(R)
    ei,ej,ev = findnz(triu(R))
    tris = collect(triangles(R))
    len = length(ev)
    seed = time()
    r = MersenneTwister(round(Int64,seed))
    a = randperm(r,len)
    nz = ceil(Int,(1-rho)*len)
    trirand = randperm(r,length(tris))

    if isodd(nz)
        nz -=1
    end
    eitest = zeros(Int,nz)
    ejtest = zeros(Int,nz)

    test_edge_count = 1
    icounter = 1
    A = copy(R)
    @show nz
    while icounter < len && test_edge_count <= nz 
        # add a new edge to test
        v1,v2,v3 = tris[trirand[icounter]]
        # make sure it is still a triangle
        if !(A[v1,v2] !=1 || A[v1,v3] != 1 || A[v2,v3] != 1)
        ei1 = v1
        ej1 = v2
        ek1 = v3

        #drop 4 edges
        A[ei1,ej1] = 0
        A[ej1,ei1] = 0

        A[ei1,ek1] = 0
        A[ek1,ei1] = 0

        dropzeros!(A)
        # if isconnected(A)
            #add to test
            eitest[test_edge_count] = ei1
            ejtest[test_edge_count] = ej1
            test_edge_count += 1

            eitest[test_edge_count] = ei1
            ejtest[test_edge_count] = ek1
            test_edge_count += 1
        # else
            # A[ei1,ej1] = 1
            # A[ej1,ei1] = 1

            # A[ei1,ek1] = 1
            # A[ek1,ei1] = 1
        # end
        end
        icounter += 1
    end
    @show test_edge_count-1
    @show nz
    eitest = eitest[1:test_edge_count-1]
    ejtest = ejtest[1:test_edge_count-1]
    Atest = sparse(eitest,ejtest,1,m,n)
    Atest = max.(Atest,Atest')
    Edges1 = eitest[2:2:end]
    Edges2 = ejtest[2:2:end]
    Edges = hcat(Edges1,Edges2)
    return A,Atest,Edges
end


function split_train_test_keep_connected_wedges(R::SparseMatrixCSC{T,Int64},rho::Float64) where T
    if !(0<=rho<=1)
      error("function split_train_test: rho must be between 0 and 1. Read `? split_train_test` for more")
    end
    
    m,n = size(R)
    ei,ej,ev = findnz(triu(R))
    tris = collect(triangles(R))
    tris = tris[randperm(length(tris))]
    len = length(ev)
    seed = time()
    r = MersenneTwister(round(Int64,seed))
    a = randperm(r,len)
    nz = ceil(Int,(1-rho)*len)

    if isodd(nz)
        nz -=1
    end
    eitest = zeros(Int,nz)
    ejtest = zeros(Int,nz)

    test_edge_count = 1
    icounter = 1
    A = copy(R)
    @show nz
    while icounter < len && test_edge_count <= nz 
        # add a new edge to test
        v1,v2,v3 = tris[icounter]
        # make sure it is still a triangle
        if !(A[v1,v2] !=1 || A[v1,v3] != 1 || A[v2,v3] != 1)
        ei1 = v1
        ej1 = v2
        ek1 = v3

        #drop 4 edges
        A[ei1,ej1] = 0
        A[ej1,ei1] = 0

        A[ei1,ek1] = 0
        A[ek1,ei1] = 0

        dropzeros!(A)
        if isconnected(A)
            #add to test
            eitest[test_edge_count] = ei1
            ejtest[test_edge_count] = ej1
            test_edge_count += 1

            eitest[test_edge_count] = ei1
            ejtest[test_edge_count] = ek1
            test_edge_count += 1
        else
            A[ei1,ej1] = 1
            A[ej1,ei1] = 1

            A[ei1,ek1] = 1
            A[ek1,ei1] = 1
        end
        end
        icounter += 1
    end
    @show test_edge_count-1
    @show nz
    eitest = eitest[1:test_edge_count-1]
    ejtest = ejtest[1:test_edge_count-1]
    Atest = sparse(eitest,ejtest,1,m,n)
    Atest = max.(Atest,Atest')
    return A,Atest
end


function split_train_test_keep_connected(R::SparseMatrixCSC{T,Int64},rho::Float64) where T
    if !(0<=rho<=1)
      error("function split_train_test: rho must be between 0 and 1. Read `? split_train_test` for more")
    end
    
    m,n = size(R)
    ei,ej,ev = findnz(triu(R))
    len = length(ev)
    seed = time()
    r = MersenneTwister(round(Int64,seed))
    a = randperm(r,len)
    nz = ceil(Int,(1-rho)*len)

    eitest = zeros(Int,nz)
    ejtest = zeros(Int,nz)

    test_edge_count = 1
    icounter = 1
    A = copy(R)
    while icounter < len && test_edge_count <= nz 
        # add a new edge to test
        ei1 = ei[a[icounter]]
        ej1 = ej[a[icounter]]
        A[ei1,ej1] = 0
        A[ej1,ei1] = 0
        dropzeros!(A)
        if isconnected(A)
            #add to test
            eitest[test_edge_count] = ei1
            ejtest[test_edge_count] = ej1
            test_edge_count += 1
        else
            A[ei1,ej1] = 1
            A[ej1,ei1] = 1
        end
        icounter += 1
    end
    @show test_edge_count-1
    @show nz
    eitest = eitest[1:test_edge_count-1]
    ejtest = ejtest[1:test_edge_count-1]
    Atest = sparse(eitest,ejtest,1,m,n)
    Atest = max.(Atest,Atest')
    # p = a[1:nz]
    # cp = setdiff(collect(1:len),p);

    # Rtrain = sparse(ei[p],ej[p],ev[p],m,n)
    # Rtrain = max.(Rtrain,Rtrain')
    # Rtest = sparse(ei[cp],ej[cp],ev[cp],m,n)
    # Rtest = max.(Rtest,Rtest')
    return A,Atest
end

function find_edges_of_tris(M)
    H1 = (M[4]'*M[5]).*M[2]'
    H3 = (M[2]'*M[5]').*M[4]'
    H2 = (M[2]*M[4]').*M[5]'
    i1,j1,v1 = findnz(H1)
    TR = H3[i1,:].*H2[j1,:]
    @assert isequal([nnz(TR[i,:]) for i = 1:length(i1)],v1)
    return i1,j1,TR
end
begin
"""
SBM(communities, P) returns a sparse matrix representing the underlying graph under that model
- comms is the vector of communities.
- comms[i] = community number of node i
- P is the matrix of probabilities between and within communities
- P is rxr where r is the number of communities
- i.e. maximum(comms) must be = r (which is asserted in the code)
"""

function SBM(comms::Vector{Int},P::Symmetric{T,Array{T,2}}) where T
    n = length(comms)
    @assert maximum(comms) == size(P,1)
    ei = Vector{Int64}()
    ej = Vector{Int64}()
    for i = 1:n
        for j = i+1:n
            curp = P[comms[i],comms[j]]
            currand = rand()
            if currand <= curp
                push!(ei,i)
                push!(ej,j)
            end
        end
    end
    A = sparse(ei,ej,1,n,n)
    max.(A,A')
end
end

function generate_SBM_data(r::Int,n0::Vector{Int},p::Float64,q::Float64)
    @assert r == length(n0)
    n = sum(n0) #total number of nodes
    communities = ones(Int64,n0[1])
    i = 2
    while i <= r
        communities = vcat(communities, i.*ones(Int64,n0[i]))
        i += 1
    end

    P = Diagonal((p-q).*ones(r)) + q.*ones(r,r) 
    P = Symmetric(P)
    A = SBM(communities,P)
    return A,communities
end

function create_subnetworks(A,r,n0)
nblocks = div(r*(r+1),2)
n0cum = vcat(0,cumsum(n0))
M = Array{SparseMatrixCSC{Int64,Int64},1}(undef,nblocks)
idx = 1
for i = 1:r
    for j = 1:i
        rows = n0cum[i]+1:n0cum[i+1]
        cols = n0cum[j]+1:n0cum[j+1]
        M[idx] = A[rows,cols]
        idx += 1
    end
end
return M
end

# putting it all together
function SBM_double_seed_idea(
        community_sizes::Vector{Int64}, #community sizes for SBM
        p::Float64, #within community connection probability
        q::Float64, #between communities
        triangles::String, #can be "triangle","wedge", or "edge"
        percentage_of_experiments::Float64, #0.5 to run a random set of half of the experiments, 1.0 for all
        myalpha::Float64 #PageRank's alpha
    )
    A,communities = generate_SBM_data(3,community_sizes,p,q)
    M = create_subnetworks(A,3,community_sizes)
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
        xsol = seeded_pagerank(Mc,myalpha,v)
        xsol1 = seeded_pagerank(Mc,myalpha,vi)
        xsol2 = seeded_pagerank(Mc,myalpha,vj)
        
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

sample_experiment_ids(experiments_to_run::Int,total_experiments::Int) = sample(1:total_experiments,experiments_to_run,replace=false)

# function double_seed(
#   A::SparseMatrixCSC{Int64,Int64},
#   v1::Int,
#   v2::Int,
#   myalpha::Float64, #PageRank's alpha
#   method::String
#   )

#   v = spzeros(size(A,2))
#   vi = copy(v)
#   vj = copy(v)

#   v[v1] = 0.5
#   v[v2] = 0.5
#   vi[v1] = 1
#   vj[v2] = 1

#   if method == "heatkernel"
#     xsol = seeded_stochastic_heat_kernel(MatrixNetwork(A),15.,v)
#     xsol1 = seeded_stochastic_heat_kernel(MatrixNetwork(A),15.,vi)
#     xsol2 = seeded_stochastic_heat_kernel(MatrixNetwork(A),15.,vj)
#   elseif method == "pagerank"
#     xsol = seeded_pagerank(A,myalpha,v)
#     xsol1 = seeded_pagerank(A,myalpha,vi)
#     xsol2 = seeded_pagerank(A,myalpha,vj)
#   else
#     error("method should be heatkernel or pagerank")
#   end

#   return xsol,xsol1,xsol2
# end

# double_seed(A::SparseMatrixCSC{Int64,Int64},v1::Int,v2::Int,myalpha::Float64) = double_seed(A,v1,v2,myalpha,"pagerank")

# function collapse_network(A)
#   # tao = 0.8
#   # myalpha = 0.8
#   # i1,j1,TR = find_edges_of_tris(M);
#   ei = Array{Int64}(undef,0)
#   ej = Array{Int64}(undef,0)
#   n = size(A,1)
#   # new dimension is n^2 + n
#   mytriangles = triangles(A;symmetries = true)
#   for tri in mytriangles
#     r,c,k = tri.v,tri.w,tri.x
#     node1 = n*(c-1) + r
#     node2 = n^2 + k
#     push!(ei,node1)
#     push!(ej,node2)
#   end
#   triangles_collapsed = sparse(ei,ej,1,n^2+n,n^2+n)
#   triangles_collapsed = max.(triangles_collapsed,triangles_collapsed')
#   return triangles_collapsed
# end

function _normout_rowstochastic(P::SparseArrays.SparseMatrixCSC{T,Int64}) where T
  n = size(P,1)
  colsums = sum(P,dims=2)
  pi,pj,pv = findnz(P)
  Q = SparseMatrixCSC(P.m,P.n,P.colptr,P.rowval,pv./colsums[pi])
end
function _normout_colstochastic(P::SparseArrays.SparseMatrixCSC{T,Int64}) where T
  n = size(P,1)
  colsums = sum(P,dims=2)
  pi,pj,pv = findnz(P)
  Q = SparseMatrixCSC(P.m,P.n,P.colptr,P.rowval,pv./colsums[pj])
end


# how many triangles does a given edge close
function close_triangles(i,j,Aref)
    A = copy(Aref)
    A[i,j]=1
    A[j,i]=1
    C1 = hcat(unzip_triangles(collect(triangles(A,i)))...)
    C2 = hcat(unzip_triangles(collect(triangles(A,j)))...)
    D1 = map(i->sort(C1[i,:]),1:size(C1,1))
    D2 = map(i->sort(C2[i,:]),1:size(C2,1))
    return length(intersect(D1,D2))
end

function findset(v1,n0)
    if 1 <= v1 <= n0[1]
        return 1:n0[1]
    elseif n0[1]+1 <= v1 <= n0[2]+n0[1]
        return n0[1]+1:n0[1]+n0[2]
    elseif n0[1]+n0[2]+1 <= v1 <= sum(n0)
        return n0[1]+n0[2]+1:sum(n0)
    else
        error("something's wrong")
    end
end

function findremaining(v1,v2,n0)
    f1 = findset(v1,n0)
    f2 = findset(v2,n0)
    ntoadd = setdiff(1:sum(n0),f1,f2)
    ntoadd = ntoadd[1]-1
    return ntoadd
end

function triangle_density(A)
    Am = A^2
    d = sum(Am)-sum(Diagonal(Am))
    Am = A*Am
    n = sum(Diagonal(Am))
    return n/d
end

plot_mm(mm,ylabelval) = plot_mm(mm,ylabelval,:black)

function plot_mm(mm,ylabelval,mylinecolor)
pyplot()
c1 = RGB(153/255,194/255,77/255)#g
c2 = RGB(218/255,98/255,125/255)#r
c3 = RGB(89/255, 60/255, 143/255)
c4 = RGB(56/255, 245/255, 235/255)

aucvals = mm[:,[4,11]]
Plots.boxplot(["Pairseed" "TRPR"],aucvals,legend = false,color=c1,alpha=0.6,xtickfont=font(10),linecolor=mylinecolor)

aucvals = mm[:,[7,8]]
Plots.boxplot!(["MUL" "MAX"],aucvals,color=c2,alpha=0.6,linecolor=mylinecolor)

aucvals = mm[:,[15,16,17]]
Plots.boxplot!(["AA" "PA" "JS"],aucvals,color=c3,alpha=0.6,linecolor=mylinecolor)
        
aucvals = mm[:,[18,19,20,21]]
Plots.boxplot!(["AA-MAX" "AA-MUL" "JS-MAX" "JS-MUL"],aucvals,color=c4,alpha=0.6,linecolor=mylinecolor)
        
aucvals = mm[:,[4,11,7,8,15,16,17,18,19,20,21]]
ma(aucvals)
Plots.plot!(ylim=(-0.2,1),yticks=0:0.2:1)
Plots.ylabel!(ylabelval,size=(900,200),xtickfont=font(10))
end

function plot_mm!(mm,ylabelval,mylinecolor;labelnumber=true,lower_y_label=false)

c1 = RGB(153/255,194/255,77/255)#g
c2 = RGB(218/255,98/255,125/255)#r
c3 = RGB(89/255, 60/255, 143/255)
c4 = RGB(56/255, 245/255, 235/255)

aucvals = mm[:,[4,11]]
Plots.boxplot!(["Pairseed" "TRPR"],aucvals,legend = false,color=c1,alpha=0.6,xtickfont=font(10),linecolor=mylinecolor)

aucvals = mm[:,[7,8]]
Plots.boxplot!(["MUL" "MAX"],aucvals,color=c2,alpha=0.6,linecolor=mylinecolor)

aucvals = mm[:,[15,16,17]]
Plots.boxplot!(["AA" "PA" "JS"],aucvals,color=c3,alpha=0.6,linecolor=mylinecolor)
        
aucvals = mm[:,[18,19,20,21]]
Plots.boxplot!(["AA-MAX" "AA-MUL" "JS-MAX" "JS-MUL"],aucvals,color=c4,alpha=0.6,linecolor=mylinecolor)
        
aucvals = mm[:,[4,11,7,8,15,16,17,18,19,20,21]]
if labelnumber == true
    ma(aucvals)
end

if lower_y_label == true
    ytickslabels = ["LOETO",0,0.2,0.4,0.6,0.8,1]
    Plots.plot!(ylim=(-0.2,1),yticks = ([-.1,0,0.2,0.4,0.6,0.8,1],ytickslabels))
else
    Plots.plot!(ylim=(-0.2,1),yticks=0:0.2:1)
end
Plots.ylabel!(ylabelval,size=(900,200),xtickfont=font(10))
end

function plot_mm2(mm,ylabelval,mylinecolor)
pyplot()
c1 = RGB(153/255,194/255,77/255)#g
c2 = RGB(218/255,98/255,125/255)#r
c3 = RGB(89/255, 60/255, 143/255)
c4 = RGB(56/255, 245/255, 235/255)

aucvals = mm[:,[4,11]]
Plots.boxplot(["Pairseed" "TRPR"],aucvals,legend = false,color=c1,alpha=0.6,xtickfont=font(10),linecolor=mylinecolor)

aucvals = mm[:,[7,8]]
Plots.boxplot!(["MUL" "MAX"],aucvals,color=c2,alpha=0.6,linecolor=mylinecolor)

aucvals = mm[:,[15,16,17]]
Plots.boxplot!(["AA" "PA" "JS"],aucvals,color=c3,alpha=0.6,linecolor=mylinecolor)
        
aucvals = mm[:,[18,19,20,21]]
Plots.boxplot!(["AA-MAX" "AA-MUL" "JS-MAX" "JS-MUL"],aucvals,color=c4,alpha=0.6,linecolor=mylinecolor)
        
aucvals = mm[:,[4,11,7,8,15,16,17,18,19,20,21]]
ma2(aucvals)
Plots.plot!(ylim=(-0.2,1),yticks=0:0.2:1)
Plots.ylabel!(ylabelval,size=(900,200),xtickfont=font(10))
end

function plot_mm2!(mm,ylabelval,mylinecolor)
c1 = RGB(153/255,194/255,77/255)#g
c2 = RGB(218/255,98/255,125/255)#r
c3 = RGB(89/255, 60/255, 143/255)
c4 = RGB(56/255, 245/255, 235/255)

aucvals = mm[:,[4,11]]
Plots.boxplot!(["Pairseed" "TRPR"],aucvals,legend = false,color=c1,alpha=0.6,xtickfont=font(10),linecolor=mylinecolor)

aucvals = mm[:,[7,8]]
Plots.boxplot!(["MUL" "MAX"],aucvals,color=c2,alpha=0.6,linecolor=mylinecolor)

aucvals = mm[:,[15,16,17]]
Plots.boxplot!(["AA" "PA" "JS"],aucvals,color=c3,alpha=0.6,linecolor=mylinecolor)
        
aucvals = mm[:,[18,19,20,21]]
Plots.boxplot!(["AA-MAX" "AA-MUL" "JS-MAX" "JS-MUL"],aucvals,color=c4,alpha=0.6,linecolor=mylinecolor)
        
aucvals = mm[:,[4,11,7,8,15,16,17,18,19,20,21]]
ma2(aucvals)
Plots.plot!(ylim=(-0.2,1),yticks=0:0.2:1)
Plots.ylabel!(ylabelval,size=(900,200),xtickfont=font(10))
end

function ma2(aucvals)
    xstart = 0.5
    m = median(aucvals,dims=1)
    m = round.(m,digits=2)
    for i = 1:length(m)
        StatsPlots.annotate!([(xstart,m[i]+0.05,text(m[i],12,:center))])
        xstart += 1.4
    end
end

function ma(aucvals)
    xstart = 0.5
    m = median(aucvals,dims=1)
    m = round.(m,digits=2)
    for i = 1:length(m)
        StatsPlots.annotate!([(xstart,-.1,text(m[i],12,:center))])
        xstart += 1.4
    end
end