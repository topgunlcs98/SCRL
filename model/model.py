import torch.nn as nn
import torch.nn.functional as F
from model.layers import GraphConvolution
# from torch.nn.parameter import Parameter
# import torch
# import math


class GCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid1)
        self.gc2 = GraphConvolution(nhid1, nhid2)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class MultiPrototypes(nn.Module):
    '''prototype in [Unsupervised Learning of Visual Features by Contrasting Cluster Assignments](https://arxiv.org/abs/2006.09882).'''
    def __init__(self, output_dim, nmb_prototypes):
        super(MultiPrototypes, self).__init__()
        self.prototype = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, nmb_prototypes, bias=False),
            nn.ReLU(inplace=True)
            )
        # self.prototype=nn.Linear(output_dim, nmb_prototypes, bias=False)

    def forward(self, x):
        out = self.prototype(x) 
        return out


class CGCN(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, nclasses, dropout):
        super(CGCN, self).__init__()
        self.gcn1 = GCN(nfeat, nhid1, nhid2, dropout)  # GCN for topological graph
        self.gcn2 = GCN(nfeat, nhid1, nhid2, dropout)  # GCN for feature graph
        self.prototype = MultiPrototypes(nhid2, nclasses)

    def forward(self, X, nsadj, nfadj):
        x_1 = self.gcn1(X, nsadj)
        x_2 = self.gcn2(X, nfadj)
        protype_x_1 = self.prototype(x_1)
        protype_x_2 = self.prototype(x_2)
        return protype_x_1, protype_x_2, x_1, x_2
