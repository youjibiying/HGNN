from torch import nn
from models import HGNN_conv, HGraphConvolutionBS
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math


class HGNN(nn.Module):
    def __init__(self, in_ch, n_class, n_hid, dropout=0.5):
        super(HGNN, self).__init__()
        self.dropout = dropout
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)

    def forward(self, x, G):
        x = F.relu(self.hgc1(x, G))  # G*x*\theta
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        return x


class MultiLayerHGNN(nn.Module):
    """
        The base block for Multi-layer GCN / ResGCN / Dense GCN
        """

    def __init__(self, in_features, hidden_features, out_features, nbaselayer,
                 withbn=True, withloop=True, activation=F.relu, dropout=0.5,
                 aggrmethod="nores", dense=False,res=True):
        """
        The base block for constructing DeepGCN model.
        :param in_features: the input feature dimension.
        :param out_features: the hidden feature dimension.
        :param nbaselayer: the number of layers in the base block.
        :param withbn: using batch normalization in graph convolution.
        :param withloop: using self feature modeling in graph convolution.
        :param activation: the activation function, default is ReLu.
        :param dropout: the dropout ratio.
        :param aggrmethod: the aggregation function for baseblock, can be "concat" and "add". For "resgcn", the default
                           is "add", for others the default is "concat".
        :param dense: enable dense connection
        """
        super(MultiLayerHGNN, self).__init__()
        self.in_features = in_features
        self.hiddendim = hidden_features
        self.out_features = out_features
        self.nhiddenlayer = nbaselayer
        self.activation = activation
        self.aggrmethod = aggrmethod
        self.dense = dense
        self.dropout = dropout
        self.withbn = withbn
        self.withloop = withloop
        self.hiddenlayers = nn.ModuleList()
        self.baselayer=HGraphConvolutionBS
        # self.baselayer = HGNN_conv
        self.res=res
        self.__makehidden()

    def __makehidden(self):
        # for i in xrange(self.nhiddenlayer):
        for i in range(self.nhiddenlayer):
            if i == 0:
                layer=self.baselayer(self.in_features,self.hiddendim,activation=self.activation)
                # layer = HGraphConvolutionBS(self.in_features, self.hiddendim, self.activation, self.withbn,
                #                             self.withloop)
            elif i == self.nhiddenlayer - 1:
                layer=self.baselayer(self.hiddendim,self.out_features)
                # layer = HGraphConvolutionBS(self.hiddendim, self.out_features, self.activation, self.withbn,
                #                             self.withloop)
            else:
                layer=self.baselayer(self.hiddendim,self.hiddendim,activation=self.activation,res=self.res)
                # layer = HGraphConvolutionBS(self.hiddendim, self.hiddendim, self.activation, self.withbn, self.withloop)
            self.hiddenlayers.append(layer)

    def _doconcat(self, x, subx):
        if x is None:
            return subx
        if self.aggrmethod == "concat":
            return torch.cat((x, subx), 1)
        elif self.aggrmethod == "add":
            return x + subx
        elif self.aggrmethod == "nores":
            return x

    def forward(self, input, G):
        x = input
        denseout = None
        # Here out is the result in all levels.
        for num,gc in enumerate(self.hiddenlayers):
            denseout = self._doconcat(denseout, x)
            x = gc(x, G)
            if num==self.nhiddenlayer - 1:
                continue
            # x = self.activation(x)
            x = F.dropout(x, self.dropout, training=self.training)

        if not self.dense:
            return self._doconcat(x, input)
        return self._doconcat(x, denseout)

    def get_outdim(self):
        return self.out_features

#
# class decoder(nn.Module):
#     def __init__(self,in_size,out_size,ffn_num):
#         self.ffn=nn.ModuleList()
#         self.in_size=