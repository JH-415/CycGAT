#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 21:07:12 2022

@author: jinghan
"""

import os.path as osp
from torch_geometric.data import Dataset, download_url, Data, InMemoryDataset
from torch_geometric.utils import add_self_loops, degree, to_undirected, dense_to_sparse, coalesce,to_scipy_sparse_matrix
from scipy.io import loadmat
import torch
import torch.utils.data as tud
import numpy as np
from lib.Hodge_Cheb_Conv import *
from torch_geometric.datasets import GNNBenchmarkDataset, ZINC
from torch_geometric.loader import DataLoader
from timm.data.mixup import Mixup
import torchvision.transforms as transforms
from torch_cluster import graclus_cluster
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from sklearn.metrics import average_precision_score

class PairData(Data):
    def __init__(self, edge_index_s=None, x_s=None, edge_index_t=None, x_t=None,
                edge_weight_s=None, edge_weight_t=None, edge_index=None, y=None,
                pos_s=None, pos_t=None):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.x_t = x_t
        self.edge_weight_s = edge_weight_s
        self.edge_weight_t = edge_weight_t
        self.edge_index = edge_index
        self.pos_s = pos_s
        self.pos_t = pos_t
        self.y = y

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index':
            return self.x_t.size(0)
        if key == 'edge_index_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)
        
        
def remove_zero_insparse(sparse_matrix):
    # Ensure the sparse matrix is coalesced
    sparse_matrix = sparse_matrix.coalesce()

    # Remove zero entries
    non_zero_mask = sparse_matrix._values() != 0
    new_values = sparse_matrix._values()[non_zero_mask]
    new_indices = sparse_matrix._indices()[:, non_zero_mask]

    # Create a new sparse matrix with the non-zero entries
    new_sparse_matrix = torch.sparse_coo_tensor(new_indices, new_values, sparse_matrix.size())
    
    return new_sparse_matrix

def cycle_adjacency_sparse(ker, skeleton):
    '''
    Compute cycle adjacency matrix (sparse mm)
    Inputs:
        ker: cycle incidence matrix (sparse matrix with 
            dim = [# of cycles * # of edges]).
        skeleton: edge index (dim = [2 * # of edges])
    '''
    par1 = adj2par1(skeleton, 268, skeleton.shape[-1])
    L1 = torch.sparse.mm(par1.transpose(0, 1), par1).to(torch.bool).to(torch.float)
    tree_index = torch.logical_or(torch.sparse.sum(ker, dim=0).to_dense()>1,
                                  torch.sparse.sum(ker, dim=0).to_dense()==0)
    ker = torch.cat([ker, tree_index.to(torch.float64).view(1,-1).to_sparse()], dim=0)
    ker1 = torch.sparse.mm(ker.transpose(0, 1), ker).to(torch.bool).to(torch.float)
    A = L1 - torch.eye(L1.size()[0]).to_sparse_coo()
    return remove_zero_insparse(A*ker1).coalesce()

class ABCD_ALL_GI_CyclePos(Dataset):
    def __init__(self, root, ABCD_ALL, eig=False, ker=None, 
                 skeleton=None, mode=0, y='mean'):
        # data aug
        # mode: 2. both; 1. only function; 0. only structure;
        self.mode = mode
        self.root = root
        self.ABCD_ALL = ABCD_ALL
        self.size = len(ABCD_ALL)
        self.y = y
        self.eig = eig
        self.skeleton = skeleton
        self.ker = ker
        tree_index = torch.logical_or(torch.sparse.sum(self.ker, dim=0).to_dense()>1,
                                      torch.sparse.sum(self.ker, dim=0).to_dense()==0)
        self.tree_index = tree_index.to(torch.float).view(-1,1)
        self.A_cycle = cycle_adjacency_sparse(self.ker, self.skeleton)
        super().__init__(root)
  
    @property
    def processed_file_names(self):
        return ['ABCD_MLGC_'+str(fileidx)+'.pt' for fileidx in range(self.len())]

    def len(self):
        return self.size

    def get(self,idx):
        ### load SC and label
        y = torch.tensor(self.ABCD_ALL[idx][2]).view(-1)[8]
        fc = torch.tensor(self.ABCD_ALL[idx][1][self.skeleton[0], 
                                                self.skeleton[1]]).to(torch.float)

        ### cycle graph
        temp = torch.eye(268)
        roi_pos = temp[self.skeleton[0]] + temp[self.skeleton[1]]
        pos = torch.cat([self.tree_index,roi_pos,self.eig], dim=-1)
        data = PairData(x_s=fc.view(-1,1), edge_index_s=self.A_cycle.coalesce().indices(), 
                        pos_s=pos, y=y.to(torch.float), x_t=torch.ones(268,1), edge_index_t=self.skeleton)
        data.num_nodes = 268
        data.num_node1 = 268
        data.num_edge1 = fc.shape[0]
        return data
        
    def process(self):        
        return None        

def FC2mask(FC, threshmode=1, k_ratio=0.25):
    '''
    Construct graph skeleton (group-level) by thresholding
    '''
    num_rois = FC.shape[1]
    FC_mean = FC.mean(dim=0)
    mean_FC = FC_mean.abs()
    if threshmode == 1:
        # select top k percent absolute average values
        v,i = mean_FC[mean_FC>0].topk(k=int(k_ratio*num_rois**2))
        mask = mean_FC>v[-1]
        mask = mask.to(torch.long)

    elif threshmode == 2:
        # select bottom k percent Consistency
        std_FC = FC.std(dim=0)
        mean_FC = std_FC / mean_FC
        v,i = mean_FC[mean_FC>0].topk(k=int(k_ratio*num_rois**2),largest=False)
        mask = mean_FC<v[-1]
        mask = mask.to(torch.long)

    else:
        # select top k percent absolute average values per roi
        mask = torch.zeros_like(mean_FC)
        for i in range(mean_FC.shape[0]):
            v,i = mean_FC[i].topk(k=int(num_rois*k_ratio))
            temp = mean_FC[i]>v[-1]
            mask[i] = temp.to(torch.float)
        mask = mask + mask.T
        mask[mask == 2] = 1     
    return mask.triu(1)

def eval_ap(y_true, y_pred):
    '''
        compute Average Precision (AP) averaged across tasks
    '''

    ap_list = []
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            # ignore nan values
            is_labeled = y_true[:, i] == y_true[:, i]
            # try:
            ap = average_precision_score(y_true[is_labeled, i], y_pred[is_labeled, i])
            # except:
            #     print(y_true[is_labeled, i], y_pred[is_labeled, i])
            #     print(torch.count_nonzero(torch.isnan(torch.tensor(y_true[is_labeled, i]))), 
            #           torch.count_nonzero(torch.isnan(torch.tensor(y_pred[is_labeled, i]))))

            ap_list.append(ap)

    if len(ap_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute Average Precision.')

    return sum(ap_list) / len(ap_list)


def eig_pe(L, k=9, i=1):
    # L: numpy Laplacian matrix
    # k: number of eigenvectors
    # i: index of the first eigenvector
    eig_vals, eig_vecs = eigh(L)
    eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
    pe = torch.from_numpy(eig_vecs[:, 1:k])
    return pe


def adj2par1(edge_index, num_node, num_edge):
    col_idx = torch.cat([torch.arange(edge_index.shape[1]),torch.arange(edge_index.shape[1])]
                        ,dim=-1).to(edge_index.device)
    row_idx = torch.cat([edge_index[0],edge_index[1]], dim=-1).to(edge_index.device)
    val = torch.cat([edge_index[0].new_full(edge_index[0].shape,-1),
                     edge_index[0].new_full(edge_index[0].shape,1)],dim=-1).to(torch.float)
    par1_sparse = torch.sparse.FloatTensor(torch.cat([row_idx, col_idx], dim=-1).view(2,-1),
                                           val,torch.Size([num_node, num_edge]))
    return par1_sparse

def post2poss(pos_t, edge_index, edge_index1):
    pos_s = torch.zeros(edge_index.shape[1],1)
    idx = pos_t[edge_index[0]]==pos_t[edge_index[1]]
    pos_s[idx] = float('inf')
    for i in range(edge_index.shape[1]):
        if pos_t[edge_index[0][i]] == pos_t[edge_index[1][i]]:
            pos_s[i] = float('inf')
        else:
#             print(min(edge_index[0][i],edge_index[1][i]),max(edge_index[0][i],edge_index[1][i]))
            temp1 = min(pos_t[edge_index[0][i]],pos_t[edge_index[1][i]]) == edge_index1[0]
            temp2 = max(pos_t[edge_index[0][i]],pos_t[edge_index[1][i]]) == edge_index1[1]
            temp = torch.logical_and(temp1,temp2)
            pos_s[i] = torch.arange(temp.shape[0])[temp]
    return pos_s

