import sys
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0, indep_weights=True):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid, indep_weights=indep_weights)
        self.gc2 = GraphConvolution(nhid, nhid, indep_weights=indep_weights)
        # self.gc3 = GraphConvolution(nhid, nhid, indep_weights=indep_weights)
        self.gc4 = GraphConvolution(nhid, nclass, indep_weights=indep_weights)
        self.dropout = dropout

    def forward(self, x, adj, labels):
        x = F.relu(self.gc1(x, adj, labels))
        x = F.dropout(x, self.dropout)
        x = F.relu(self.gc2(x, adj, labels))
        # x = F.relu(self.gc3(x, adj, labels))
        x = self.gc4(x, adj, labels)
        # return F.log_softmax(x, dim=1)
        return x


class MLP(nn.Module):
    def __init__(self, ninput=200, nhidden=150, nclass=2, dropout=0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(ninput, nhidden)
        self.fc2 = nn.Linear(nhidden, nclass)
        self.dropout = dropout

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = F.dropout(out, self.dropout)
        out = self.fc2(out)
        return out
