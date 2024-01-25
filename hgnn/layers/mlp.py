import torch.nn as nn
import torch.nn.functional as F
import dgl


class HeteroMLP(nn.Module):
    def __init__(self,
                 n_inp: int,
                 n_hid: int,
                 n_out: int,
                 n_layer: int = 2,
                 act=F.relu,
                 dropout:float = 0.2,
                 residual=False,
                 bias: bool = False):
        assert n_layer > 0
        super(HeteroMLP, self).__init__()
        self.layers = nn.ModuleList()
        self.act = act
        self.dropout = nn.Dropout(dropout)
        for i in range(n_layer):
            in_dim = n_inp if i == 0 else n_hid
            out_dim = n_out if i == n_layer - 1 else n_hid
            self.layers.append(nn.Linear(in_dim, out_dim, bias=bias))

        self.residual = residual
        if residual:
            self.residual_layer = nn.Linear(n_inp, n_out, bias=bias)

    def forward(self, G:dgl.DGLHeteroGraph, inputs: dict):
        if G.is_block:
            h = {k: v[:G.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            h = inputs
        for key in G.ntypes:
            for layer in self.layers:
                h[key] = self.dropout(self.act(layer(h[key])))
        if self.residual:
            for key in G.ntypes:
                h[key] = self.act(self.residual_layer(inputs[key]))
        return h


class HeteroAdaptMLP(nn.Module):
    def __init__(self,
                 n_inp: int,
                 n_hid: int,
                 n_out: int,
                 node_type:(list, dict),
                 n_layer: int = 2,
                 act=F.relu,
                 dropout=0.6,
                 residual=False,
                 bias: bool = False):
        assert n_layer > 0
        super(HeteroAdaptMLP, self).__init__()
        self.layers = nn.ModuleList()
        self.act = act
        self.node_type = node_type
        self.residual = residual
        for i in range(n_layer):
            out_dim = n_out if i == n_layer - 1 else n_hid
            layer = nn.ModuleDict()
            for node_type in self.node_type:
                if i == 0:
                    in_dim = n_inp if isinstance(n_inp, int) else n_inp[node_type]
                else:
                    in_dim = n_hid
                layer[str(node_type)] = nn.Linear(in_dim, out_dim, bias=bias)
            self.layers.append(layer)
        if residual:
            self.residual_layer = nn.ModuleDict()
            for node_type in self.node_type:
                self.residual_layer[node_type] = nn.Linear(n_inp, n_out, bias=bias)

    def forward(self, G:dgl.DGLHeteroGraph, inputs: dict):
        if not isinstance(G, dict) and G.is_block:
            h = {k: v[:G.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            h = inputs
        outputs = [h]
        for layer in self.layers:
            h = {}
            for key in self.node_type:
                h[key] = self.act(layer[key](outputs[-1][key]))
            outputs.append(h)
        h = outputs[-1]
        if self.residual:
            for key in self.node_type:
                h[key] = self.act(self.residual_layer[key](inputs[key]))
        return h


class HeteroAdaptMLPNew(nn.Module):
    def __init__(self,
                 n_inp: int,
                 n_hid: int,
                 n_out: int,
                 node_type:(list, dict),
                 n_layer: int = 2,
                 act=F.relu,
                 dropout: float = 0.2,
                 residual=False,
                 bias: bool = False):
        assert n_layer > 0
        super(HeteroAdaptMLPNew, self).__init__()
        self.layers = nn.ModuleList()
        self.act = act
        self.node_type = node_type
        self.residual = residual
        self.dropout = nn.Dropout(dropout)
        for i in range(n_layer):
            in_dim = n_inp if i == 0 else n_hid
            out_dim = n_out if i == n_layer - 1 else n_hid
            layer = nn.ModuleDict()
            for node_type in self.node_type:
                layer[node_type] = nn.Linear(in_dim, out_dim, bias=bias)
            self.layers.add_module(layer)
        if residual:
            self.residual_layer = nn.ModuleDict()
            for node_type in self.node_type:
                self.residual_layer[node_type] = nn.Linear(n_inp, n_out, bias=bias)

    def forward(self, G:dgl.DGLHeteroGraph, inputs: dict):
        if G.is_block:
            h = {k: v[:G.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            h = inputs
        outputs = [h]
        for layer in self.layers:
            h = {}
            for key in self.node_type:
                h[key] = self.dropout(self.act(layer[key](outputs[-1][key])))
            outputs.append(h)
        h = outputs[-1]
        if self.residual:
            for key in G.ntypes:
                h[key] = self.act(self.residual_layer[key](inputs[key]) + h[key])
        return h
