function general_pairseed_different_evaluation_varytopk(
        percentage_of_experiments::Union{Float64,Int64,Array{Int64,2}}, #0.5 to run a random set of half of the experiments, 10 for 10 experiments, or specific edges
        myalpha::Float64, #PageRank's alpha
        topkref, #number of top k we are interested in
        Atrain,
        Atest,
        eval_type, #BMV, MRR, recall_topk
        atestval; #"AND" or "OR" experiment
        dropnodes=[]
    )
@show topkref
    A = Atrain+Atest
    n = size(Atrain,1)
    community_sizes = [0,0,n]
    n0 = community_sizes

    if typeof(percentage_of_experiments) == Int64
        ei,ej,ev = findnz(triu(Atrain))
        total_experiments = length(ei)
        experiments_to_run = percentage_of_experiments
        experimentsids = sample(1:total_experiments,experiments_to_run,replace=false)
        newei = ei[experimentsids]
        newej = ej[experimentsids]
    elseif typeof(percentage_of_experiments) == Float64
        ei,ej,ev = findnz(triu(Atrain))
        total_experiments = length(ei)
        experiments_to_run = floor(Int,percentage_of_experiments*total_experiments)
        experimentsids = sample(1:total_experiments,experiments_to_run,replace=false)
        newei = ei[experimentsids]
        newej = ej[experimentsids]
    elseif typeof(percentage_of_experiments) == Array{Int64,2}
        newei = percentage_of_experiments[:,1]
        newej = percentage_of_experiments[:,2]
        experiments_to_run = length(newei)
    end

    ei = newei
    ej = newej

    # AUC = zeros(experiments_to_run,21)

    AUCRECALL = zeros(experiments_to_run,21)
    AUCMRR = zeros(experiments_to_run,21)
    AUCBMV = zeros(experiments_to_run,21)
    count_wedges = zeros(experiments_to_run)
    keepids = []


    eval_fn = alltopk
    
    for i = 1:experiments_to_run
        print(i)
        print(", ")
        v1 = ei[i]
        v2 = ej[i]
        
        xd_pr,xs1_pr,xs2_pr = double_seed(Atrain,v1,v2,myalpha,"pr")
        xalternate = pairseed_trpr(Atrain,v1,v2,myalpha,10)
        #find ground truth
        if atestval==false
            ek_nodes = ((A[:,v1] .* A[:,v2]).>0).nzind # everything that either v1 or v2 connects to
            ek1 = ((Atrain[:,v1] .* Atrain[:,v2]).>0).nzind
            ek_nodes = setdiff(ek_nodes,ek1)

            aset = setdiff(1:n,ek1)
            aset = setdiff(aset,v1,v2)

            if eval_type == "MRR"
                topk = length(aset)
            end
            # aset = 1:n
            # ek_nodes_drop = ((Atrain[:,v1] .* Atrain[:,v2]).>0).nzind # everything that either v1 or v2 connects to
            # ek_nodes = setdiff(ek_nodes,ek_nodes_drop)
        elseif atestval==true
            ek_nodes = ((Atest[:,v1] .* Atest[:,v2]).>0).nzind
            ek1 = ((A[:,v1] .* A[:,v2]).>0).nzind
            ek2 = ((Atrain[:,v1] .* Atrain[:,v2]).>0).nzind
            # ek3 = intersect(Atrain[v1,:].nzind,ek1)
            ek3 = Atrain[v1,:].nzind
            # ek4 = intersect(Atrain[v2,:].nzind,ek1)
            ek4 = Atrain[v2,:].nzind
            cset = union(ek2,ek3,ek4)
            aset = setdiff(1:n,cset)
            aset = setdiff(aset,v1,v2)
            if !isempty(dropnodes)
                aset = setdiff(aset,dropnodes)
            end
            aset = sort(aset) # nodes we are interested in looking at (everything except nodes that are already connected to that edge)
            if eval_type == "MRR"
                topk = length(aset)
            end
        else
            error("atestval can only be true or false")
        end

        if !isempty(ek_nodes)
            topk = length(aset)
            push!(keepids,i)
            count_wedges[i] = length(ek_nodes)

            ek_nodes = findin_index(ek_nodes,aset)
            xrefreal = ek_nodes

            @assert findfirst(xrefreal.==0) == nothing # we should be able to find everything in aset

            xk4 = sortperm(xd_pr[aset],rev=true)[1:topk]
            xk5 = sortperm(xs1_pr[aset],rev=true)[1:topk]
            xk6 = sortperm(xs2_pr[aset],rev=true)[1:topk]

            xk7 = sortperm((xs1_pr.*xs2_pr)[aset],rev=true)[1:topk]
            xk8 = sortperm(max.(xs1_pr,xs2_pr)[aset],rev=true)[1:topk]
            xk11 = sortperm(xalternate[aset],rev=true)[1:topk]

            #random
            xk1 = sortperm(rand(size(Atrain,1))[aset],rev=true)[1:topk]
            
        tpr,fpr,auc = eval_fn(xrefreal,xk1,topkref); AUCRECALL[i,1] = tpr;AUCMRR[i,1] = fpr;AUCBMV[i,1] = auc;
        tpr,fpr,auc = eval_fn(xrefreal,xk4,topkref); idi = 4; AUCRECALL[i,idi] = tpr;AUCMRR[i,idi] = fpr;AUCBMV[i,idi] = auc;
        tpr,fpr,auc = eval_fn(xrefreal,xk5,topkref); idi = 5; AUCRECALL[i,idi] = tpr;AUCMRR[i,idi] = fpr;AUCBMV[i,idi] = auc;
        tpr,fpr,auc = eval_fn(xrefreal,xk6,topkref); idi = 6; AUCRECALL[i,idi] = tpr;AUCMRR[i,idi] = fpr;AUCBMV[i,idi] = auc;
        tpr,fpr,auc = eval_fn(xrefreal,xk7,topkref); idi = 7; AUCRECALL[i,idi] = tpr;AUCMRR[i,idi] = fpr;AUCBMV[i,idi] = auc;
        tpr,fpr,auc = eval_fn(xrefreal,xk8,topkref); idi = 8; AUCRECALL[i,idi] = tpr;AUCMRR[i,idi] = fpr;AUCBMV[i,idi] = auc;
        tpr,fpr,auc = eval_fn(xrefreal,xk11,topkref); idi = 11; AUCRECALL[i,idi] = tpr;AUCMRR[i,idi] = fpr;AUCBMV[i,idi] = auc;

        aa_distance,pf_distance,js_distance,aa_distance_union,pf_distance_union,js_distance_union = similarity_predict_withJS_union(Atrain,v1,v2,1:n)
        aa_distance_max,aa_distance_mul,js_distance_max,js_distance_mul = similarity_predict_singles(Atrain,v1,v2,1:n)
######
        aa_distance = sortperm(aa_distance[aset],rev=true)[1:topk]
        pf_distance = sortperm(pf_distance[aset],rev=true)[1:topk]
        js_distance = sortperm(js_distance[aset],rev=true)[1:topk]
        aa_distance_union = sortperm(aa_distance_union[aset],rev=true)[1:topk]
        pf_distance_union = sortperm(pf_distance_union[aset],rev=true)[1:topk]
        js_distance_union = sortperm(js_distance_union[aset],rev=true)[1:topk]
        aa_distance_max = sortperm(aa_distance_max[aset],rev=true)[1:topk]
        aa_distance_mul = sortperm(aa_distance_mul[aset],rev=true)[1:topk]
        js_distance_max = sortperm(js_distance_max[aset],rev=true)[1:topk]
        js_distance_mul = sortperm(js_distance_mul[aset],rev=true)[1:topk]

        # # similarity predict
        # tpr,fpr,auc = calc_AUC_new(xrefreal,aa_distance); AUC[i,12] = auc
        # tpr,fpr,auc = calc_AUC_new(xrefreal,pf_distance); AUC[i,13] = auc
        # tpr,fpr,auc = calc_AUC_new(xrefreal,js_distance); AUC[i,14] = auc

        tpr,fpr,auc = eval_fn(xrefreal,aa_distance,topkref); idi = 12; AUCRECALL[i,idi] = tpr;AUCMRR[i,idi] = fpr;AUCBMV[i,idi] = auc;
        tpr,fpr,auc = eval_fn(xrefreal,pf_distance,topkref); idi = 13; AUCRECALL[i,idi] = tpr;AUCMRR[i,idi] = fpr;AUCBMV[i,idi] = auc;
        tpr,fpr,auc = eval_fn(xrefreal,js_distance,topkref); idi = 14; AUCRECALL[i,idi] = tpr;AUCMRR[i,idi] = fpr;AUCBMV[i,idi] = auc;

        # settoconsider = setdiff(1:length(aa_distance_union),v1,v2) # no need here because they are removed already
        # @show aa_distance_union
        tpr,fpr,auc = eval_fn(xrefreal,aa_distance_union,topkref); idi = 15; AUCRECALL[i,idi] = tpr;AUCMRR[i,idi] = fpr;AUCBMV[i,idi] = auc;
        tpr,fpr,auc = eval_fn(xrefreal,pf_distance_union,topkref); idi = 16; AUCRECALL[i,idi] = tpr;AUCMRR[i,idi] = fpr;AUCBMV[i,idi] = auc;
        tpr,fpr,auc = eval_fn(xrefreal,js_distance_union,topkref); idi = 17; AUCRECALL[i,idi] = tpr;AUCMRR[i,idi] = fpr;AUCBMV[i,idi] = auc;

        tpr,fpr,auc = eval_fn(xrefreal,aa_distance_max,topkref); idi = 18; AUCRECALL[i,idi] = tpr;AUCMRR[i,idi] = fpr;AUCBMV[i,idi] = auc;
        tpr,fpr,auc = eval_fn(xrefreal,aa_distance_mul,topkref); idi = 19; AUCRECALL[i,idi] = tpr;AUCMRR[i,idi] = fpr;AUCBMV[i,idi] = auc;
        tpr,fpr,auc = eval_fn(xrefreal,js_distance_max,topkref); idi = 20; AUCRECALL[i,idi] = tpr;AUCMRR[i,idi] = fpr;AUCBMV[i,idi] = auc;
        tpr,fpr,auc = eval_fn(xrefreal,js_distance_mul,topkref); idi = 21; AUCRECALL[i,idi] = tpr;AUCMRR[i,idi] = fpr;AUCBMV[i,idi] = auc;
    end

    end
    return AUCRECALL[keepids,:],AUCMRR[keepids,:],AUCBMV[keepids,:],count_wedges[keepids]
end

function general_pairseed_different_evaluation(
        percentage_of_experiments::Union{Float64,Int64,Array{Int64,2}}, #0.5 to run a random set of half of the experiments, 10 for 10 experiments, or specific edges
        myalpha::Float64, #PageRank's alpha
        topk, #number of top k we are interested in
        Atrain,
        Atest,
        eval_type, #BMV, MRR, recall_topk
        atestval; #"AND" or "OR" experiment
        dropnodes=[]
    )

    A = Atrain+Atest
    n = size(Atrain,1)
    community_sizes = [0,0,n]
    n0 = community_sizes

    if typeof(percentage_of_experiments) == Int64
        ei,ej,ev = findnz(triu(Atrain))
        total_experiments = length(ei)
        experiments_to_run = percentage_of_experiments
        experimentsids = sample(1:total_experiments,experiments_to_run,replace=false)
        newei = ei[experimentsids]
        newej = ej[experimentsids]
    elseif typeof(percentage_of_experiments) == Float64
        ei,ej,ev = findnz(triu(Atrain))
        total_experiments = length(ei)
        experiments_to_run = floor(Int,percentage_of_experiments*total_experiments)
        experimentsids = sample(1:total_experiments,experiments_to_run,replace=false)
        newei = ei[experimentsids]
        newej = ej[experimentsids]
    elseif typeof(percentage_of_experiments) == Array{Int64,2}
        newei = percentage_of_experiments[:,1]
        newej = percentage_of_experiments[:,2]
        experiments_to_run = length(newei)
    end

    ei = newei
    ej = newej

    AUC = zeros(experiments_to_run,21)
    count_wedges = zeros(experiments_to_run)
    keepids = []

    if eval_type == "recall_topk"
        eval_fn = recall_topk
    elseif eval_type == "MRR"
        eval_fn = mean_reciprocal_rank
    elseif eval_type == "BMV"
        eval_fn = binary_mean_value
    else
        error("only input allowed is recall_topk, MRR, BMV")
    end
    
    for i = 1:experiments_to_run
        print(i)
        print(", ")
        v1 = ei[i]
        v2 = ej[i]
        
        xd_pr,xs1_pr,xs2_pr = double_seed(Atrain,v1,v2,myalpha,"pr")
        xalternate = pairseed_trpr(Atrain,v1,v2,myalpha,10)
        #find ground truth
        if atestval==false
            ek_nodes = ((A[:,v1] .* A[:,v2]).>0).nzind # everything that either v1 or v2 connects to
            ek1 = ((Atrain[:,v1] .* Atrain[:,v2]).>0).nzind
            ek_nodes = setdiff(ek_nodes,ek1)

            aset = setdiff(1:n,ek1)
            aset = setdiff(aset,v1,v2)

            if eval_type == "MRR"
                topk = length(aset)
            end
            # aset = 1:n
            # ek_nodes_drop = ((Atrain[:,v1] .* Atrain[:,v2]).>0).nzind # everything that either v1 or v2 connects to
            # ek_nodes = setdiff(ek_nodes,ek_nodes_drop)
        elseif atestval==true
            ek_nodes = ((Atest[:,v1] .* Atest[:,v2]).>0).nzind
            ek1 = ((A[:,v1] .* A[:,v2]).>0).nzind
            ek2 = ((Atrain[:,v1] .* Atrain[:,v2]).>0).nzind
            # ek3 = intersect(Atrain[v1,:].nzind,ek1)
            ek3 = Atrain[v1,:].nzind
            # ek4 = intersect(Atrain[v2,:].nzind,ek1)
            ek4 = Atrain[v2,:].nzind
            cset = union(ek2,ek3,ek4)
            aset = setdiff(1:n,cset)
            aset = setdiff(aset,v1,v2)
            if !isempty(dropnodes)
                aset = setdiff(aset,dropnodes)
            end
            aset = sort(aset) # nodes we are interested in looking at (everything except nodes that are already connected to that edge)
            if eval_type == "MRR"
                topk = length(aset)
            end
        else
            error("atestval can only be true or false")
        end

        if !isempty(ek_nodes)
            push!(keepids,i)
            count_wedges[i] = length(ek_nodes)

            ek_nodes = findin_index(ek_nodes,aset)
            xrefreal = ek_nodes

            @assert findfirst(xrefreal.==0) == nothing # we should be able to find everything in aset

            xk4 = sortperm(xd_pr[aset],rev=true)[1:topk]
            xk5 = sortperm(xs1_pr[aset],rev=true)[1:topk]
            xk6 = sortperm(xs2_pr[aset],rev=true)[1:topk]

            xk7 = sortperm((xs1_pr.*xs2_pr)[aset],rev=true)[1:topk]
            xk8 = sortperm(max.(xs1_pr,xs2_pr)[aset],rev=true)[1:topk]
            xk11 = sortperm(xalternate[aset],rev=true)[1:topk]

            #random
            xk1 = sortperm(rand(size(Atrain,1))[aset],rev=true)[1:topk]
            
        tpr,fpr,auc = eval_fn(xrefreal,xk1); AUC[i,1] = auc; #@show auc; #@show myauc(xrefreal,xk1)
        tpr,fpr,auc = eval_fn(xrefreal,xk4); AUC[i,4] = auc; #@show auc; #@show myauc(xrefreal,xk4)
        tpr,fpr,auc = eval_fn(xrefreal,xk5); AUC[i,5] = auc; #@show auc; #@show myauc(xrefreal,xk5)
        tpr,fpr,auc = eval_fn(xrefreal,xk6); AUC[i,6] = auc; #@show auc; #@show myauc(xrefreal,xk6)
        tpr,fpr,auc = eval_fn(xrefreal,xk7); AUC[i,7] = auc; #@show auc; #@show myauc(xrefreal,xk5)
        tpr,fpr,auc = eval_fn(xrefreal,xk8); AUC[i,8] = auc; #@show auc; #@show myauc(xrefreal,xk6)
        tpr,fpr,auc = eval_fn(xrefreal,xk11); AUC[i,11] = auc

        aa_distance,pf_distance,js_distance,aa_distance_union,pf_distance_union,js_distance_union = similarity_predict_withJS_union(Atrain,v1,v2,1:n)
        aa_distance_max,aa_distance_mul,js_distance_max,js_distance_mul = similarity_predict_singles(Atrain,v1,v2,1:n)
######
        aa_distance = sortperm(aa_distance[aset],rev=true)[1:topk]
        pf_distance = sortperm(pf_distance[aset],rev=true)[1:topk]
        js_distance = sortperm(js_distance[aset],rev=true)[1:topk]
        aa_distance_union = sortperm(aa_distance_union[aset],rev=true)[1:topk]
        pf_distance_union = sortperm(pf_distance_union[aset],rev=true)[1:topk]
        js_distance_union = sortperm(js_distance_union[aset],rev=true)[1:topk]
        aa_distance_max = sortperm(aa_distance_max[aset],rev=true)[1:topk]
        aa_distance_mul = sortperm(aa_distance_mul[aset],rev=true)[1:topk]
        js_distance_max = sortperm(js_distance_max[aset],rev=true)[1:topk]
        js_distance_mul = sortperm(js_distance_mul[aset],rev=true)[1:topk]

        # # similarity predict
        # tpr,fpr,auc = calc_AUC_new(xrefreal,aa_distance); AUC[i,12] = auc
        # tpr,fpr,auc = calc_AUC_new(xrefreal,pf_distance); AUC[i,13] = auc
        # tpr,fpr,auc = calc_AUC_new(xrefreal,js_distance); AUC[i,14] = auc

        sum(aa_distance) == 0 ? auc = 0 : auc = eval_fn(xrefreal,aa_distance)[3]; AUC[i,12] = auc
        sum(pf_distance) == 0 ? auc = 0 : auc = eval_fn(xrefreal,pf_distance)[3]; AUC[i,13] = auc
        sum(js_distance) == 0 ? auc = 0 : auc = eval_fn(xrefreal,js_distance)[3]; AUC[i,14] = auc

        # settoconsider = setdiff(1:length(aa_distance_union),v1,v2) # no need here because they are removed already
        # @show aa_distance_union
        sum(aa_distance_union) == 0 ? auc = 0 : auc = eval_fn(xrefreal,aa_distance_union)[3]; AUC[i,15] = auc
        sum(pf_distance_union) == 0 ? auc = 0 : auc = eval_fn(xrefreal,pf_distance_union)[3]; AUC[i,16] = auc
        sum(js_distance_union) == 0 ? auc = 0 : auc = eval_fn(xrefreal,js_distance_union)[3]; AUC[i,17] = auc

        sum(aa_distance_max) == 0 ? auc = 0 : auc = eval_fn(xrefreal,aa_distance_max)[3]; AUC[i,18] = auc
        sum(aa_distance_mul) == 0 ? auc = 0 : auc = eval_fn(xrefreal,aa_distance_mul)[3]; AUC[i,19] = auc
        sum(js_distance_max) == 0 ? auc = 0 : auc = eval_fn(xrefreal,js_distance_max)[3]; AUC[i,20] = auc
        sum(js_distance_mul) == 0 ? auc = 0 : auc = eval_fn(xrefreal,js_distance_mul)[3]; AUC[i,21] = auc
    end

    end
    return AUC[keepids,:],count_wedges[keepids]
end

#s1 is ref nodes, s2 are topk
function recall_topk(s1,s2)
    t = 0
    for i = 1:length(s1)
        if in(s1[i],s2)
            t+=1
        end
    end
    return t,s1,t/length(s1)
end

function mean_reciprocal_rank(s1,s2)
    ranks = map(i->findfirst(s2.==s1[i]),1:length(s1))
    reciprocals = 1 ./ ranks
    return ranks,reciprocals,mean(reciprocals)
end

function binary_mean_value(s1,s2)
    if isempty(intersect(s1,s2))
        return s1,s2,false
    else
        return s1,s2,true
    end
end

function calcall_measures(s1,s2)
    recallval = recall_topk(s1,s2[1:topk])
    mean_reciprocal_val = mean_reciprocal_rank(s1,s2)
    binary_mean_val = binary_mean_value(s1,s2[1:topk])
    return recallval[3],mean_reciprocal_val[3],binary_mean_val[3]
end

function alltopk(s1,s2,topkref)
    curtopk = topkref[1]
    binary_mean_val1 = binary_mean_value(s1,s2[1:curtopk])

    curtopk = topkref[2]
    binary_mean_val2 = binary_mean_value(s1,s2[1:curtopk])

    curtopk = topkref[3]
    binary_mean_val3 = binary_mean_value(s1,s2[1:curtopk])
    return binary_mean_val1[3],binary_mean_val2[3],binary_mean_val3[3]
end

function general_wedges(A,expnb,topk,andorval,wi,nbkeep,evalmethod)
    @show "deow"
A2 = A*A
A2 = A2-Diagonal(A2)
A2 = A2.*A
if !(wi=="any")
    CI = findall(triu(A2).==wi)
else
    CI = findall(triu(A2).>=1)
end
CI = CI[randperm(length(CI))]
GAUC = zeros(expnb,21)
GCWEDGES = zeros(expnb)
keepids = []
expnb = min(expnb,length(CI))
for i = 1:expnb
    print(i)
    print(", ")
    ei,ej = CI[i][1],CI[i][2]

    ek =(A[ei,:].*A[ej,:]).nzind
    @assert length(ek) == A2[ei,ej]
    if !(nbkeep == "all")
        if nbkeep == "half"
            ek = ek[randperm(length(ek))][1:div(length(ek),2)]
        else
            ek = ek[randperm(length(ek))][1:nbkeep]#[1:div(length(ek),2)]
        end
    end

    Atest = spzeros(Int,size(A)...)
    for i = 1:length(ek)
        Atest[ei,ek[i]] = 1
        Atest[ej,ek[i]] = 1
    end
    Atest = max.(Atest,Atest')
    Atrain = A-Atest
    dropzeros!(Atrain)
    dropzeros!(Atest)
    Edges = [ei ej]

    # if isconnected(Atrain) #doesn't matter
    
        AUC1,CWEDGES = general_pairseed_different_evaluation(
            Edges,#Edges,#500,
            0.85,
            topk,
            Atrain,
            Atest,
            evalmethod, #BMV, MRR, recall_topk
            andorval #"AND" (true) or "OR" (false) experiment
        );

        if size(AUC1,1) != 0
            GAUC[i,:] = AUC1
            GCWEDGES[i] = CWEDGES[1]
            push!(keepids,i)
        end
    # end
end
AUC1 = GAUC[keepids,:]
GCWEDGES = GCWEDGES[keepids]
return AUC1,GCWEDGES
end