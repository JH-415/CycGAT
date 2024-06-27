#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 17:04:28 2024

@author: jinghan
"""


import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
import time
from lib.Hodge_Cheb_Conv import *
from lib.Hodge_Dataset import *
from scipy.sparse.linalg import eigsh
import mat73
import os
from scipy.io import savemat

parser = argparse.ArgumentParser()
parser.add_argument('--l2', type=float, default=1e-4, help='weight decay')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--mode', type=int, default=1, help='train mode')
parser.add_argument('--test', type=int, default=0, help='if train')
parser.add_argument('--batch_size', type=int, default=5, help='train mode')
parser.add_argument('--epoch', type=int, default=5, help='max epoch')
parser.add_argument('--dropout_ratio', type=float, default=0.25, help='layer num in block1')
parser.add_argument('--k_ratio', type=float, default=0.25, help='layer num in block1')
parser.add_argument('--fold', type=int, default=-1, help='The second pooling loc')
parser.add_argument('--seed', type=int, default=-1, help='num of deconv filters')

parser.add_argument('--layers', type=str, default="1_1_1", help='num of channels')
parser.add_argument('--channels', type=str, default="32_64_128", help='num of filterss')
parser.add_argument('--heads', type=int, default=2, help='polynomial order of the indirect FC')
parser.add_argument('--mlp_channels', type=int, default=0, help='num of fully connected layers')
## mode: 0. only structure; 1. only function; 2. both structure and function
##       3. direct and indirect; 4. denoised structure; 5. 
args = parser.parse_args()
layers = [int(item) for item in args.layers.split('_')]
channels = [int(item) for item in args.channels.split('_')]

    
class CycGAT(torch.nn.Module):
    def __init__(self, in_channel=1, layers=[1,1,1], channels=[16,32,64], mlp_channels=[],
                 dropout_ratio=0.25, num_classes=1, heads=2, num_edges=None):
        super(CycGAT, self).__init__()
        self.layers = layers
        self.channels = channels
        self.mlp_channels = mlp_channels
        gcn_outsize = channels[0]
        gcn_insize = in_channel
        heads = heads
        layers= [(gnn.GATConv(gcn_insize, gcn_outsize, heads=heads, 
                                       edge_dim=None, fill_value='mean'), 
                                       'x_t, adj_t -> x_t'),
                (gnn.BatchNorm(gcn_outsize*heads), 'x_t -> x_t'),
                (nn.LeakyReLU(), 'x_t -> x_t'),
                (nn.Dropout(p=dropout_ratio), 'x_t -> x_t'),]
        self.init_conv = gnn.Sequential('x_t, adj_t', layers)
        gcn_insize = gcn_outsize*heads
        self.init_pos = nn.Sequential(
            nn.Linear(268+32, gcn_insize),
            nn.LeakyReLU(),
            nn.Linear(gcn_insize, gcn_insize),
            nn.LeakyReLU())

        for i, gcn_outsize in enumerate(self.channels):
            if i != 0:
                ### pos convolution
                fc = gnn.GATConv(gcn_insize, gcn_outsize, heads=heads, 
                                 edge_dim=None, fill_value='mean')
                setattr(self, 'PosConv{}'.format(i), fc)
                layers = [(gnn.BatchNorm(gcn_outsize*heads), 'x_t -> x_t'),
                        (nn.LeakyReLU(), 'x_t -> x_t'),
                        (nn.Dropout(p=dropout_ratio), 'x_t -> x_t'),]
                fc = gnn.Sequential('x_t', layers)
                setattr(self, 'PosACT{}'.format(i), fc)
                ### align feature dimension for residual connection
                fc = nn.Linear(gcn_insize, gcn_outsize*heads) 
                setattr(self, 'Proj{}'.format(i), fc)
                gcn_insize = gcn_outsize*heads

            ### cycle convolution
            for j in range(self.layers[i]):
                fc = gnn.GATConv(gcn_insize, gcn_outsize, heads=heads, 
                                 edge_dim=None, fill_value='mean')
                setattr(self, 'CycleConv{}{}'.format(i,j), fc)
                layers = [(gnn.BatchNorm(gcn_outsize*heads), 'x_t -> x_t'),
                        (nn.LeakyReLU(), 'x_t -> x_t'),
                        (nn.Dropout(p=dropout_ratio), 'x_t -> x_t'),]
                fc = gnn.Sequential('x_t', layers)
                setattr(self, 'CycleACT{}{}'.format(i,j), fc)
                gcn_insize = gcn_outsize*heads
                
        ## Readout and fully connected layer
        self.readout = nn.Linear(gcn_insize, 1) 
        mlp_insize = num_edges
            
        for i, mlp_outsize in enumerate(self.mlp_channels):
            fc = nn.Sequential(
                nn.Linear(mlp_insize, mlp_outsize),
                nn.BatchNorm1d(mlp_outsize),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Dropout(dropout_ratio),
                )
            setattr(self, 'mlp%d' % i, fc)
            mlp_insize = mlp_outsize

        self.out = nn.Linear(mlp_insize, num_classes)
        
        
    def forward(self, graph, device='cpu', if_att=False):
        cycle_index = graph.edge_index_s
        tree_index = graph.pos_s[:,:1].to(torch.bool).view(-1)
        pos = graph.pos_s[:,1:].to(torch.float)
        adj_c = SparseTensor(row=cycle_index[0], col=cycle_index[1], 
                             value=torch.ones(cycle_index[0].shape).to(device)).t()
        x_cycle = graph.x_s.to(torch.float)
        
        x_cycle = self.init_conv(x_cycle, adj_c)
        pos = self.init_pos(pos)
        x_cycle = x_cycle + pos
        
        for i in range(len(self.layers)):
            if i != 0:
                fc = getattr(self, 'PosConv{}'.format(i))
                pos = fc(pos, adj_c)
                fc = getattr(self, 'PosACT{}'.format(i))
                pos = fc(pos)
                ### projection layer
                fc = getattr(self, 'Proj{}'.format(i))
                x_cycle = fc(x_cycle) + pos
            
            for j in range(self.layers[i]):
                ### convolution layer
                fc = getattr(self, 'CycleConv{}{}'.format(i,j))
                x_cycle1 = fc(x_cycle, adj_c)
                fc = getattr(self, 'CycleACT{}{}'.format(i,j))
                x_cycle = fc(x_cycle1) + x_cycle

                
        ### readout as node feature
        x_cycle = self.readout(x_cycle).view(graph.num_graphs,-1)

        for i, _ in enumerate(self.mlp_channels):
            fc = getattr(self, 'mlp%d' % i)
            x_cycle = fc(x_cycle)
        return self.out(x_cycle)

def train(loader):
    model.train()
    total_loss = 0
    y_pred, y = [], []
    for data in loader: 
        out = model(data.to(device),device=device)
        loss = criterion(out.view(-1,1), data.y.to(device).view(-1,1))
        total_loss += loss*data.num_graphs
        loss.backward()
        optimizer.step() 
        optimizer.zero_grad()  
        try:
            y_pred.extend(out.squeeze()>0)
            y.extend(data.y.squeeze())
        except:
            continue

    y_pred, y = torch.tensor(y_pred).to(torch.float), torch.tensor(y)
    acc = torch.count_nonzero(y_pred==y) / y_pred.shape[0]
    return acc, total_loss/len(loader.dataset)

def test(loader):
     model.eval()
     y_pred, y = [], []
     total_loss = 0
     
     for data in loader:
         with torch.no_grad():
             out = model(data.to(device),device=device) 
         loss = criterion(out.view(-1,1), data.y.to(device).view(-1,1)) 
         total_loss += loss * data.num_graphs
         try:
             y_pred.extend(out.squeeze()>0)
             y.extend(data.y.squeeze())
         except:
             continue
     y_pred, y = torch.tensor(y_pred).to(torch.float), torch.tensor(y)
     acc = torch.count_nonzero(y_pred==y) / y_pred.shape[0]
     return acc, total_loss/len(loader.dataset)
 
       
###############################################################################
##################### main #########################
###############################################################################
if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print(device)
    start = time.time()
    name = 'data/DEMO_DATA.mat'
    data = loadmat(name)
    ABCD_ALL = data['DEMO_DATA']
    # torch.manual_seed(args.seed)

    name = 'data/DEMO_groupSC.mat'
    matdata = loadmat(name)
    FC = torch.tensor(matdata['sc_mean'])/(torch.tensor(matdata['sc_var'])+1e-4)
    FC = FC.abs()
    
    temp = loadmat(osp.join('data', 'DEMO_groupSC_Basis.mat'))
    skeleton = torch.tensor(temp['G_edges'].astype(float)).to(torch.int64).T-1
    csr = temp['ker']
    ker = torch.sparse_coo_tensor(np.array(csr.nonzero()),csr.data,csr.shape)
    FC = FC[skeleton[0], skeleton[1]].to(torch.float)
    
    ###### compute edge cycle positional encoding
    ker = ker.to_dense()
    A = torch.matmul(ker, ker.T)
    A = A - A.diag().diag()
    D = A.sum(dim=-1).diag()
    L = D-A
    lambda0, eig = torch.linalg.eigh(L)
    ker2 = ker.to(torch.bool).to(torch.float).T
    ker2 = ker2 / ker2.sum(dim=1).view(-1,1)
    edge_eig = torch.matmul(ker2.to(torch.float), eig[:,:32].to(torch.float))
    print(eig.shape, edge_eig.shape, ker.shape)
    ##### build dataset
    dataset = ABCD_ALL_GI_CyclePos(root='data', ABCD_ALL=ABCD_ALL, ker=ker.to_sparse(), 
                                   eig=edge_eig, skeleton=skeleton)
    in_channel = 1

###############################################################################
###############################################################################
    batch_size = args.batch_size
    if args.fold == -1:
        folds = [0,1,2,3,4]
    else:
        folds = [args.fold]
    for fold in folds:
        print('==================================================================================')
        print('Fold {} begin'.format(fold))
        print('==================================================================================')
        num_workers = 2
        trainset = Subset(dataset, range(20))
        validset = Subset(dataset, range(20,25))
        testset = Subset(dataset, range(25,30))
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        valid_loader = DataLoader(validset, batch_size=batch_size, num_workers=num_workers)
        test_loader = DataLoader(testset, batch_size=batch_size, num_workers=num_workers)

        criterion = nn.BCEWithLogitsLoss()
        criterion1 = nn.L1Loss()
        lr = args.lr
        beta1 = 0.5
        mlp_channels = [] if args.mlp_channels==0 else [256]*args.mlp_channels
        model = CycGAT(layers=layers, channels=channels, num_edges=dataset.skeleton.shape[1],
                                        mlp_channels=mlp_channels).to(device)
        
        filename = './weights/CycGAT/'
        if not osp.exists(filename):
            os.makedirs(filename)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.l2, betas=(beta1, 0.999))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, 
                                                               patience=5, factor=0.25, min_lr=1e-6)
        num_epochs = args.epoch
        
        print("Starting Training Loop...")
        start = time.time()
        save_name = 'Cycformer_'+'SCABCD'+'_channels'+str(args.channels)\
            +'_layer'+str(args.layers)+'_heads'+str(args.heads)+'_mlp'\
                +str(args.mlp_channels)+'_FOLD{}'.format(fold)+'_k'+str(int(args.k_ratio*100))
        save_path = filename + save_name + '.pt'
        
        if args.test == 0:
            best_acc, best_loss = test(test_loader)
            print('==================================================================================')
            print(f'Test Loss: {best_loss:.4f}, Test Acc: {best_acc:.4f}')
            print('==================================================================================')
            for epoch in range(1, num_epochs):
                train_acc, train_loss = train(train_loader)
                valid_acc, valid_loss = test(valid_loader)
                scheduler.step(valid_loss)
                if optimizer.param_groups[-1]['lr']<1.1e-5:
                    break    
                
                elapsed = (time.time()-start) / 60
                # with open(txt_path, "a") as f:
                #     f.write(f'Epoch: {epoch:03d}, time: {elapsed:.2f} Train Loss: {total_loss:.4f}, Train Corr: {train_corr:.4f}, Valid Loss: {valid_loss:.4f}, Valid Corr: {valid_corr:.4f}, Valid RMSE: {valid_rmse:.4f}')
                print(f'Epoch: {epoch:03d}, time: {elapsed:.2f} Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}')
                if valid_loss<best_loss:
                    best_loss = valid_loss
                    torch.save(model.state_dict(), save_path)
                    print('Model saved! \n')  
                    best_acc1, best_loss1 = test(test_loader)
                    print('==================================================================================')
                    print(f'Test Loss: {best_loss1:.4f}, Test Corr: {best_acc1:.4f}')
                    print('==================================================================================')


