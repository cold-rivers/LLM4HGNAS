####
# Author: 高扬
# Description:
# 异质图GNN中,处理的数据格式一般是G和inputs,
# G:DGLGraph
# inputs: (str:Tensor)
# 其中 G 包含了网络的结构信息,inputs包含的节点特征信息.
# 由于节点的异质性,不同类型节点的特征不一样,或者需要不同的处理方式,所以采用dict的方式存储节点特征.
####
import torch.nn as nn


class HeteroLinear(nn.Linear):
    """
        所有类型的节点使用同一特征变换矩阵
    """

    def forward(self, inputs: dict):
        res = {}
        for each in inputs:
            res[each] = super(HeteroLinear, self).forward(inputs[each])
        return res


class HeteroAdaptLinear(nn.Module):
    """
       同一类型的节点共享同一特征变换矩阵
    """

    def __init__(self,
                 n_inp: (dict, int),  # {str:int} or int
                 n_hid: int,
                 node_dict,
                 act=None):
        super(HeteroAdaptLinear, self).__init__()
        self.node_dict = node_dict
        self.adapt_ws = nn.ModuleDict()
        for t in node_dict:
            if isinstance(n_inp, int):
                self.adapt_ws[t] = nn.Linear(n_inp, n_hid)
            else:
                self.adapt_ws[t] = nn.Linear(n_inp[t], n_hid)
        self.act = act if act else lambda x:x

    def forward(self, inputs: dict):
        h = {}
        for ntype in self.node_dict:
            h[ntype] = self.act(self.adapt_ws[ntype](inputs[ntype]))
        return h


class HeteroAdaptLinearNew(nn.Module):
    """
       同一类型的节点共享同一特征变换矩阵
    """

    def __init__(self, n_inp, n_hid, node_dict):
        super(HeteroAdaptLinearNew, self).__init__()
        self.node_dict = node_dict
        self.adapt_ws = nn.ModuleDict()
        for t in node_dict:
            for i in range(3):
                cur_inp = n_inp if i == 0 else n_hid
                tmp = nn.Linear(cur_inp, n_hid)
                name = self.opt_name(t, i)
                self.adapt_ws[name] = tmp

    def opt_name(self, ntype, layer):
        return f"{ntype}_layer_{layer}"

    def forward(self, G, inputs: dict):
        outputs = [inputs]
        for i in range(3):
            tmp = {}
            for ntype in self.node_dict:
                opt_name = self.opt_name(ntype, i)
                tmp[ntype] = self.adapt_ws[opt_name](outputs[-1][ntype])
            outputs.append(tmp)
        return outputs[-1]
