# packages needed
ENV["PYTHON"]=""
using Pkg
Pkg.build("PyCall")
using Plots
pyplot()

# for input files
using MAT
using NumbersFromText
using DelimitedFiles
using CSV

# graphs/linear algebra packages
using MatrixNetworks
using SparseArrays
using LinearAlgebra

# code that I need
include("all_code/code_needed.jl")
include("all_code/methods.jl")
include("all_code/global_functions.jl")
;

#read the data
M = Int.(readdlm("real-world/ChCh-Miner_durgbank-chem-chem.tsv"))
A = sparse(M[:,1],M[:,2],1,maximum(M),maximum(M))
A = max.(A,A')
A = largest_component(A)[1]
A = A-Diagonal(A)
dropzeros!(A)
;

# split the data
tao = 0.7
Atrain,Atest = split_train_test(A,tao);
cc = scomponents(Atrain)
cids = findall(cc.map.==argmax(cc.sizes))

Atrain = Atrain[cids,cids]
Atest = Atest[cids,cids]

n = size(Atrain,1)
;

method_used = "BMV"
topk = [25,10,5];
AUC1,AUC2,AUC3,CWEDGES1 = general_pairseed_different_evaluation_varytopk(
        500,#Edges,#500,
        0.85,
        topk,
        Atrain,
        Atest,
        method_used, #BMV, MRR, recall_topk
        true #"AND" (true) or "OR" (false) experiment
    );
AUCW,cWedgesW = general_wedges(A,500,5,true,"any","all","BMV")
;

pyplot()
include("all_code/plot_values_script.jl")
