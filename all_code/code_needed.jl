# Pkgs that I need

using SparseArrays
using LinearAlgebra
using MatrixNetworks
using MatrixNetworks
using StatsBase
using Plots
using StatsPlots
include("find_triangles.jl")
using PyCall

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
function calc_AUC_new(Rtest,Xa)
  Rtest = vec(Rtest)
  Xa = vec(Xa)
  minNonZero = minimum(abs.(Xa[findall(Xa.!=0)]))
  Xa = Xa / minNonZero
  fpr,tpr,thresholds = metrics.roc_curve(Rtest,Xa)
  auc = metrics.auc(fpr,tpr)
  return tpr,fpr,auc
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

# ,seedon)
#   for v in seedon
#     xsol = seeded_pagerank(triangles_collapsed,myalpha,v)
#   end



# TR = spones(TR)
# nrows,ncols = size(TR)
# C = vcat(
#         hcat(spzeros(Int64,nrows,nrows),TR),
#         hcat(TR',spzeros(Int64,ncols,ncols))
#         )


# Mtrain,Mtest = split_train_test(C,tao);
# Ctest = Mtest[nrows+1:end,1:nrows]


# seedon = 1:nrows

# X = zeros(Float64,ncols,length(seedon));

# for i = 1:length(seedon)
#   X[:,i] = seeded_pagerank(Mtrain,myalpha,seedon[i])[nrows+1:end]
# end
# tpr2,fpr2,auc2 = calc_AUC_new(Ctest,X); @show auc2
# end