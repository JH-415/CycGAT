%%
load('data/DEMO_DATA.mat')
sub_num = size(DEMO_DATA,1);
roi_num = size(DEMO_DATA{1,2},1);
X = zeros(roi_num, roi_num, sub_num);
for i = 1:sub_num
    X(:,:,i) = DEMO_DATA{i,2};
end
sc_mean = mean(X,3);
sc_var = std(X,[],3);

%% group level mask and cycle basis
thresh = maxk(sc_mean(:), int16(268*268*0.25));
sf_mask = sc_mean > thresh(end);
sc_mean = single(sf_mask) .* (1 ./ abs(sc_mean+1e-6));
sc_mean = sc_mean - diag(diag(sc_mean));
G = graph(sc_mean);
G_edges = table2array(G.Edges(:,1));
%     cycle_basis{i,1} = G_edges;
T = minspantree(G);
T_edges = table2array(T.Edges(:,1));

%%
ker = zeros(size(G_edges,1)-size(T_edges,1), size(G_edges,1));
j = 1;
for e_idx = 1:size(G_edges,1)
    temp = G_edges(e_idx,:) == T_edges;
    if nnz(temp(:,1)&temp(:,2)) == 0
        temp_edges = cat(1, T_edges, G_edges(e_idx,:));
        G1 = graph(sc_mean);
        par = index2par(temp_edges, 268);
        L1 = par' * par;
        [EigVector,~] = Hodge_ker(L1);
        EigVector = abs(EigVector)>0.01;
        temp_idx = temp_edges(EigVector,:);

        for t = 1:size(temp_idx,1)
            temp = temp_idx(t,:) == G_edges;
            ker(j, find(temp(:,1)&temp(:,2) == 1,1)) = 1;
        end
        j = j + 1;
    end
end
ker = sparse(ker);
%% save data

save(fullfile('data','DEMO_groupSC'),"sc_mean","sc_var");
save(fullfile('data','DEMO_groupSC_Basis'),"G_edges","ker");

