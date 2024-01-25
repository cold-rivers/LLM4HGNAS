import torch
import torch.nn as nn

from ..layers.linear import HeteroAdaptLinear


class RelGraphEmbed(nn.Module):
    r"""Embedding layer for featureless heterograph."""

    def __init__(self,
                 node_dict,
                 number_of_nodes,
                 n_inp,
                 embed_name='embed',
                 activation=None,
                 dropout=0.0,
                 batch_train=False):
        super(RelGraphEmbed, self).__init__()
        self.embed_name = embed_name
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.batch_train = batch_train

        self.embeds = nn.ParameterDict()
        for ntype, n_node in number_of_nodes.items():
            embed = nn.Parameter(torch.Tensor(n_node, n_inp))
            nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))
            self.embeds[ntype] = embed

    def extract_embed(self, input_nodes):
        emb = {}
        for ntype, nid in input_nodes.items():
            nid = input_nodes[ntype]
            emb[ntype] = self.embeds[ntype][nid]
        return emb

    def forward(self, G=None, inputs=None):
        if self.batch_train:
            assert "input_nodes" in inputs
            input_nodes = inputs["input_nodes"]
            return self.extract_embed(input_nodes)
        else:
            return self.embeds


class AdaptiveRelGraphEmbed(nn.Module):
    r"""Embedding layer for featureless heterograph."""

    def __init__(self,
                 ntypes,
                 origin_feature,
                 n_hid,
                 activation=None,
                 dropout=0.0,
                 batch_train=False):
        super(AdaptiveRelGraphEmbed, self).__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.batch_train = batch_train

        self.embeds = nn.ParameterDict()
        n_inp = {}
        for ntype in ntypes:
            embed = nn.Parameter(origin_feature[ntype])
            self.embeds[ntype] = embed
            n_inp[ntype] = origin_feature[ntype].size(-1)
        self.out_linear = HeteroAdaptLinear(n_inp, n_hid, ntypes, act=activation)

    def extract_embed(self, input_nodes):
        emb = {}
        for ntype, nid in input_nodes.items():
            nid = input_nodes[ntype]
            emb[ntype] = self.embeds[ntype][nid]
        return emb

    def forward(self, G=None, inputs=None):
        if self.batch_train:
            assert "input_nodes" in inputs
            input_nodes = inputs["input_nodes"]
            return self.out_linear(self.extract_embed(input_nodes))
        else:
            return self.out_linear(self.embeds)
