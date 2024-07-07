function par = index2par(edge_index, node_num)
par = zeros(node_num,size(edge_index,1));
for i = 1:size(edge_index,1)
    par(edge_index(i,1),i) = -1;
    par(edge_index(i,2),i) = 1;
end
end