#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 21:58:34 2022

@author: jinghan
"""
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import OptTensor
from torch_geometric.nn.pool import graclus, max_pool
from torch_geometric.data import Data, Batch
from torch_scatter import scatter
from torch_geometric.utils import add_self_loops, dense_to_sparse
from typing import Callable, Optional, Tuple, Union
from torch_geometric.typing import SparseTensor
import torch_sparse
from torch_scatter import scatter_add, scatter_max, scatter_mean

from torch_geometric.utils import unbatch_edge_index, softmax
from torch_geometric.utils.num_nodes import maybe_num_nodes
import torch_geometric.utils as ut
from scipy.sparse.linalg import eigsh

###############################################################################
######################### Preprocessing #######################################
###############################################################################

def unbatch_edge_attr(edge_index: Tensor, edge_attr: Tensor, batch: Tensor):
    deg = ut.degree(batch, dtype=torch.int64)
    ptr = torch.cat([deg.new_zeros(1), deg.cumsum(dim=0)[:-1]], dim=0)

    edge_batch = batch[edge_index[0]]
    edge_index = edge_index - ptr[edge_batch]
    sizes = ut.degree(edge_batch, dtype=torch.int64).cpu().tolist()
    return edge_index.split(sizes, dim=1), edge_attr.split(sizes, dim=0)

def adj2par1(edge_index, num_node, num_edge):
    col_idx = torch.cat([torch.arange(edge_index.shape[1]),torch.arange(edge_index.shape[1])]
                        ,dim=-1).to(edge_index.device)
    row_idx = torch.cat([edge_index[0],edge_index[1]], dim=-1).to(edge_index.device)
    val = torch.cat([edge_index[0].new_full(edge_index[0].shape,-1),
                     edge_index[0].new_full(edge_index[0].shape,1)],dim=-1).to(torch.float)
    par1_sparse = torch.sparse.FloatTensor(torch.cat([row_idx, col_idx], dim=-1).view(2,-1),
                                           val,torch.Size([num_node, num_edge]))
    return par1_sparse

def par2adj(par1):
    a = par1.to_sparse()
    _, perm = a.indices()[1].sort(dim=-1, descending=False)
    return a.indices()[0][perm].view(-1,2).T

 
###############################################################################
############################# Convolution #####################################
###############################################################################

class HodgeLaguerreFastConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, K: int, 
                  bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        assert K > 0
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lins = torch.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False,
                   weight_initializer='glorot') for _ in range(K)])
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)

    
    def forward(self, x, adj_t):
        """"""
        # x: N*T*C
        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = self.lins[0](Tx_0)
        xshape = x.shape
        k = 1

        # propagate_type: (x: Tensor, norm: Tensor)
        if len(self.lins) > 1:
            x = x.view(xshape[0],-1)
            Tx_1 = x - self.message_and_aggregate(adj_t=adj_t, x=x)
            if len(xshape)>=3:
                Tx_1 = Tx_1.view(xshape[0],xshape[1],-1)
            out = out + self.lins[1](Tx_1)
        for lin in self.lins[2:]:
            inshape = Tx_1.shape
            Tx_1 = Tx_1.view(inshape[0],-1)
            Tx_2 = self.message_and_aggregate(adj_t=adj_t, x=x)
            if len(xshape)>=3:
                Tx_2 = Tx_2.view(inshape[0],inshape[1],-1)
                Tx_1 = Tx_1.view(xshape[0],xshape[1],-1)
            # print(Tx_0.shape,Tx_1.shape,Tx_2.shape)
            Tx_2 = (-Tx_2 + (2*k+1)*Tx_1 - k* Tx_0) / (k+1)
            k += 1
            out = out + lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2
        if self.bias is not None:
            out = out + self.bias
        return out
    
    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return torch_sparse.matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={len(self.lins)}')
    
###############################################################################
class HodgeLaguerreConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, K: int, 
                  bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        assert K > 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lins = torch.nn.ModuleList([
            Linear(in_channels, out_channels, bias=False,
                    weight_initializer='glorot') for _ in range(K)
        ])

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        zeros(self.bias)

    
    def forward(self, x: Tensor, edge_index: Tensor,
                edge_weight: OptTensor = None, batch: OptTensor = None):
        """"""
        # x: N*T*C
        norm = edge_weight
        Tx_0 = x
        Tx_1 = x  # Dummy.
        out = self.lins[0](Tx_0)
        xshape = x.shape
        k = 1

        # propagate_type: (x: Tensor, norm: Tensor)
        if len(self.lins) > 1:
            x = x.view(xshape[0],-1)
            Tx_1 = x - self.propagate(edge_index, x=x, norm=norm, size=None)
            if len(xshape)>=3:
                Tx_1 = Tx_1.view(xshape[0],xshape[1],-1)
            out = out + self.lins[1](Tx_1)

        for lin in self.lins[2:]:
            inshape = Tx_1.shape
            Tx_1 = Tx_1.view(inshape[0],-1)
            Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm, size=None)
            if len(xshape)>=3:
                Tx_2 = Tx_2.view(inshape[0],inshape[1],-1)
                Tx_1 = Tx_1.view(xshape[0],xshape[1],-1)
            # print(Tx_0.shape,Tx_1.shape,Tx_2.shape)
            Tx_2 = (-Tx_2 + (2*k+1)*Tx_1 - k* Tx_0) / (k+1)
            k += 1
            out = out + lin.forward(Tx_2)
            Tx_0, Tx_1 = Tx_1, Tx_2

        if self.bias is not None:
            out = out + self.bias

        return out


    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={len(self.lins)}')

