"""RGCN layer implementation"""
import math

import dgl.function as fn
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .operators import gnn_map, HeteroAdaptMLP, HomoZero, get_aggregate_fn


def evaluate_actions(gnn_desc, n_layers, rel_names):
    len_desc = len(gnn_desc)
    assert len_desc == n_layers
    for layer_opts in gnn_desc:
        for rels in rel_names:
            assert rels in gnn_desc[layer_opts], f"relation {rels} not in gnn_desc"


class RelationOnlyNet(nn.Module):

    def __init__(self,
                 gnn_desc,
                 node_dict,  # {str:int} map node type to their index
                 edge_dict,  # {str:int} map edge type to their index
                 n_inp,
                 n_hid,
                 n_out,
                 dropout,
                 n_heads=4,
                 use_norm=True,
                 task_specific_encoder=None,
                 n_layers=2,
                 self_loop=False
                 ):
        self.n_layers = n_layers
        self.self_loop = self_loop
        self.node_dict = node_dict
        self.edge_dict = edge_dict
        self.n_inp = n_inp
        self.n_hid = n_hid
        self.n_out = n_out
        self.dropout = dropout
        super(RelationOnlyNet, self).__init__()
        if task_specific_encoder is None:
            self.task_specific_encoder = HeteroAdaptMLP(n_inp, n_hid, n_hid, node_dict, n_layer=1, act=F.gelu)
        else:
            self.task_specific_encoder = task_specific_encoder
        self.gcs = nn.ModuleList()
        self.n_layers = self.evaluate_actions(gnn_desc, None)
        self.build_model(gnn_desc, node_dict, edge_dict, n_hid, n_out, n_heads, use_norm)

    def build_model(self, gnn_desc, node_dict, edge_dict, n_hid, n_out, n_heads, use_norm):
        for layer_desc in gnn_desc.values():
            self.gcs.append(RelGraphConvLayer(n_hid,
                                              n_hid,
                                              layer_desc,
                                              aggregate=get_aggregate_fn(layer_desc["multi_aggr"], n_hid, n_hid,
                                                                         node_dict),
                                              activation=None,
                                              self_loop=False,
                                              dropout=self.dropout,
                                              n_heads=n_heads))
        self.out = nn.Linear(n_hid, n_out)

    def evaluate_actions(self, actions, tmp):
        evaluate_actions(actions, self.n_layers, self.edge_dict)
        return self.n_layers

    def forward(self, G, inputs, out_key):
        h = self.task_specific_encoder(G, inputs)
        # h = inputs
        if isinstance(G, list):
            for i in range(len(self.gcs)):
                h = self.gcs[i](G[i], h)
        else:
            for i in range(len(self.gcs)):
                h = self.gcs[i](G, h)
        if out_key:
            return self.out(h[out_key])
        else:
            # out_key is None, do not need a task specific mlp
            return h


class RelGraphConvLayer(nn.Module):
    r"""Relational graph convolution layer.
        Designed to deal with edge type.

    Parameters
    ----------
    n_inp : int
        Input feature size.
    n_out : int
        Output feature size.
    mods : dict [str: str]
        edge type and its gnn type
    aggregate : str or function
        to deal with different outputs generated by edge type
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.6
    """

    def __init__(self,
                 n_inp,
                 n_out,
                 mods,
                 aggregate='sum',
                 activation=None,
                 self_loop=False,
                 dropout=0.6,
                 n_heads=4):
        super(RelGraphConvLayer, self).__init__()
        self.n_inp = n_inp
        self.n_out = n_out
        self.activation = activation
        self.self_loop = self_loop

        self.mods = nn.ModuleDict()
        self.build_subgraph_gnn(activation, dropout, mods, n_heads, n_inp, n_out)
        assert not isinstance(aggregate, str)
        self.agg_fn = aggregate

        # opt_weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(th.Tensor(n_inp, n_out))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def build_subgraph_gnn(self, activation, dropout, mods, n_heads, n_inp, n_out):
        for etype in mods:
            if len(etype) <= 3 and not isinstance(etype[0], tuple):  # exclude meta_paths_dict
                self.mods["_".join(etype)] = gnn_map(mods[etype], n_inp, n_out,
                                                     n_heads=n_heads, act=activation, dropout=dropout)
        #print(mods)
        for _, v in self.mods.items():
            set_allow_zero_in_degree_fn = getattr(v, 'set_allow_zero_in_degree', None)
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)

    def forward(self, G, inputs):
        """Forward computation

        Parameters
        ----------
        G : DGLHeteroGraph
            Input graph.
        inputs : dict[str, torch.Tensor]
            Node feature for each node type.

        Returns
        -------
        dict[str, torch.Tensor]
            New node features for each node type.
        """
        outputs = {nty: [] for nty in G.dsttypes}

        if G.is_block:
            src_inputs = inputs
            dst_inputs = {k: v[:G.number_of_dst_nodes(k)] for k, v in inputs.items()}
        else:
            if isinstance(inputs, list):
                src_inputs = inputs
                dst_inputs = inputs[-1]
            else:
                src_inputs = dst_inputs = inputs
        self.passing_channels(G, src_inputs, dst_inputs, outputs)
        rsts = {}
        for ntype, alist in outputs.items():
            if len(alist) != 0:
                rsts[ntype] = self.agg_fn(alist, ntype)
            elif self.self_loop:
                rsts[ntype] = None
            else:
                # TODO if no channel, use skip connection
                rsts[ntype] = dst_inputs[ntype]
                pass

        outputs = {}
        for ntype, h in rsts.items():
            #print(ntype,len(h))
            if self.self_loop:
                if h is None:
                    h = th.matmul(dst_inputs[ntype], self.loop_weight)
                else:
                    h = h + th.matmul(dst_inputs[ntype], self.loop_weight)
            if self.activation:
                h = self.activation(h)
            outputs[ntype] = self.dropout(h)
        return outputs

    def passing_channels(self, G, src_inputs, dst_inputs, outputs):
        for stype, etype, dtype in G.canonical_etypes:
            rel_graph = G[stype, etype, dtype]
            if rel_graph.number_of_edges() == 0:
                continue
            if stype not in src_inputs or dtype not in dst_inputs:
                continue
            homo_gnn = self.mods["_".join((stype, etype, dtype))]
            if isinstance(homo_gnn, HomoZero):
                continue
            dstdata = homo_gnn(
                rel_graph,
                (src_inputs[stype], dst_inputs[dtype]))
            outputs[dtype].append(dstdata)


class JKRelGraphConvLayer(RelGraphConvLayer):
    def __init__(self,
                 layer_index,
                 skip_connect_mods,
                 n_inp,
                 n_out,
                 mods,
                 aggregate='sum',
                 activation=None,
                 self_loop=False,
                 dropout=0.6,
                 n_heads=4):

        self.layer_index = layer_index
        self.skip_connect_mods_dict = skip_connect_mods

        super(JKRelGraphConvLayer, self).__init__(
            n_inp,
            n_out,
            mods,
            aggregate,
            activation,
            self_loop,
            dropout,
            n_heads
        )
        self.skip_connect_mods = nn.ModuleDict()
        self.build_skip_gnn(activation, dropout, skip_connect_mods, n_heads, n_inp, n_out)

    def build_skip_gnn(self, activation, dropout, mods, n_heads, n_inp, n_out):
        for etype in mods:
            from_to, _, _, _ = etype
            self.skip_connect_mods["_".join(etype)] = gnn_map(mods[etype], n_inp, n_out,
                                                              n_heads=n_heads, act=activation, dropout=dropout)
        for _, v in self.mods.items():
            set_allow_zero_in_degree_fn = getattr(v, 'set_allow_zero_in_degree', None)
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)

    def passing_channels(self, G, src_inputs_list, dst_inputs, outputs):
        src_inputs = src_inputs_list[-1]
        # dst_inputs = dst_inputs_list[-1]
        for stype, etype, dtype in G.canonical_etypes:
            rel_graph = G[stype, etype, dtype]
            if rel_graph.number_of_edges() == 0:
                continue
            if stype not in src_inputs or dtype not in dst_inputs:
                continue
            model_name = "_".join((stype, etype, dtype))
            if model_name not in self.mods:
                continue
            homo_gnn = self.mods[model_name]
            if isinstance(homo_gnn, HomoZero):
                continue
            dstdata = homo_gnn(
                rel_graph,
                (src_inputs[stype], dst_inputs[dtype]))
            outputs[dtype].append(dstdata)
        for from_to, stype, etype, dtype in self.skip_connect_mods_dict.keys():
            from_, to_ = [int(each) for each in from_to.split()]
            if to_ == self.layer_index:
                rel_graph = G[stype, etype, dtype]
                src_inputs = src_inputs_list[from_]
                # dst_inputs = dst_inputs[to_]
                if rel_graph.number_of_edges() == 0:
                    continue
                if stype not in src_inputs or dtype not in dst_inputs:
                    continue
                # homo_gnn = self.mods["_".join((stype, etype, dtype))]
                homo_gnn = self.skip_connect_mods["_".join((from_to, stype, etype, dtype))]
                if isinstance(homo_gnn, HomoZero):
                    continue
                dstdata = homo_gnn(
                    rel_graph,
                    (src_inputs[stype], dst_inputs[dtype]))
                outputs[dtype].append(dstdata)


class SkipConnectionNet(RelationOnlyNet):
    def evaluate_actions(self, gnn_desc, state_num):
        return len(gnn_desc) - 1

    def build_model(self, gnn_desc, node_dict, edge_dict, n_hid, n_out, n_heads, use_norm):
        inter_modes = gnn_desc["inter_modes"]
        for i, (k, layer_desc) in enumerate(gnn_desc.items()):
            if "layer" not in k:
                continue
            modes = layer_desc
            self.gcs.append(JKRelGraphConvLayer(
                i,
                inter_modes,
                n_hid,
                n_hid,
                modes,
                aggregate=get_aggregate_fn(layer_desc["multi_aggr"], n_hid, n_hid,
                                           node_dict),
                activation=None,
                self_loop=False,
                dropout=self.dropout,
                n_heads=n_heads))
        self.out = nn.Linear(n_hid, n_out)

    def forward(self, G, inputs, out_key, return_embeding=False):
        h = self.task_specific_encoder(G, inputs)
        # h = inputs
        input_list = [h]
        if isinstance(G, list):
            for i in range(len(self.gcs)):
                h = self.gcs[i](G[i], input_list)
                input_list.append(h)
        else:
            for i in range(len(self.gcs)):
                h = self.gcs[i](G, input_list)
                input_list.append(h)
        if out_key:
            if return_embeding:
                return self.out(h[out_key]), h
            else:
                return self.out(h[out_key])
        else:
            # out_key is None, do not need a task specific mlp
            return h
        pass

