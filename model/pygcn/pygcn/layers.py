import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True,indep_weights=True):
        super(GraphConvolution, self).__init__()
        self.indep_weights = indep_weights

        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        # create weights for Global, Leaf, OR, AND, NOT
        # Global
        self.weight_global = Parameter(torch.FloatTensor(in_features, out_features))
        # Leaf
        self.weight_leaf = Parameter(torch.FloatTensor(in_features, out_features))
        # OR
        self.weight_or = Parameter(torch.FloatTensor(in_features, out_features))
        # AND
        self.weight_and = Parameter(torch.FloatTensor(in_features, out_features))
        # NOT
        self.weight_not = Parameter(torch.FloatTensor(in_features, out_features))

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)



        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight_global.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        self.weight_global.data.uniform_(-stdv, stdv)
        self.weight_leaf.data.uniform_(-stdv, stdv)
        self.weight_or.data.uniform_(-stdv, stdv)
        self.weight_and.data.uniform_(-stdv, stdv)
        self.weight_not.data.uniform_(-stdv, stdv)

        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj,labels):
        if self.indep_weights is True:
            support = None
            for i in range(len(labels)):
                if labels[i] == 0: # global node
                    temp = torch.mm(input[i].unsqueeze(0), self.weight_global)
                elif labels[i] == 1: #leaf node
                    temp = torch.mm(input[i].unsqueeze(0), self.weight_leaf)
                elif labels[i] == 2: # OR
                    temp = torch.mm(input[i].unsqueeze(0), self.weight_or)
                elif labels[i] == 3: # and
                    temp = torch.mm(input[i].unsqueeze(0), self.weight_and)
                elif labels[i] == 4: # not
                    temp = torch.mm(input[i].unsqueeze(0), self.weight_not)

                if support is None:
                    support = temp
                else:
                    support = torch.cat((support,temp),0)
        else:
            support = torch.mm(input, self.weight)

        # output = torch.spmm(adj, support)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
