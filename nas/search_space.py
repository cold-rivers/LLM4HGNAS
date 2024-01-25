from collections import OrderedDict, defaultdict
import numpy as np
import torch

gnn_list_full = [
    "zero_conv",
    "gcn_conv",
    "gat_conv",
    "edge_conv",
    "sage_pool",
]
gnn_list_0_1 = ["zero_conv", "gcn_conv", ]  # "id_conv"
multi_aggr_list = ['sum', 'max', 'mean', "att", 'min']  #
#multi_aggr_list = ['sum']

class GNNArch(object):
    def __init__(self):
        self.cls_name = "basic"
        self.option_id = ""
        self.code = ""
        self.desc = ""

        self.train_score = ""
        self.train_loss = ""
        self.train_time = ""

        self.val_score = ""
        self.val_loss = ""
        self.val_time = ""

        self.test_score = ""
        self.test_loss = ""
        self.test_time = ""


class SearchSpaceBase(object):
    def __init__(
            self,
            option_dict,
            type_list,
            to_model_func=None
    ):
        """
        :param option_dict: {type_name:options}, a dictionary that records all options/operator used in current  search space.
        :param type_list: [type_name_1, type_name_2,...]
        """
        self.option_dict = option_dict
        self.type_list = type_list
        self.to_model_func = None

    def encode(self, arch: GNNArch):
        """
        Translate the given architecture description to its unique code.
        :param desc:
        :return: vec
        """
        return arch.option_id

    def decode(self, arch: GNNArch):
        """
        Translate the arch code to its description.
        :param arch_desc:
        :return: vec
        """
        res = []
        for type_name, op_index in zip(self.type_list, arch.option_id):
            res.append(self.option_dict[type_name][op_index])
        return res

    def sample(self, n_samples=1):
        """
        Randomly sample architecture from the search space.
        :param n_samples:
        :return:
        """
        res = []
        for i in range(n_samples):
            cur_arch = []
            for type_name in self.type_list:
                can_op = self.option_dict[type_name]
                op = np.random.choice(can_op, 1)
                cur_arch.append(op)
            arch = GNNArch()
            arch.option_id = cur_arch
            res.append(arch)
        return res

    def generate_baseline(self, n_samples=1):
        """
        return classical model in the search space.
        :param n_samples:
        :return:
        """
        pass

    def to_model(self, arch: GNNArch, args):
        return self.to_model_func(arch, args)


class SearchSpace(SearchSpaceBase):
    def __init__(
            self,
            edge_types,
            n_layers,
            predict_keys,
            full_gnn_list=False,
            predict_inter_layer=False,
            contain_zero=True,
    ):
        super(SearchSpace, self).__init__(None, None)
        # hyper parameters of the search space
        self.n_layers = n_layers
        self.predict_inter_layer = predict_inter_layer
        self.contain_zero = contain_zero
        self.full_gnn_list = full_gnn_list

        self.predict_keys = predict_keys
        self.edge_types = edge_types

        # build the option space for further architecture search
        self.option_dict = self.build_default_option_dict(self.full_gnn_list, self.contain_zero)
        self.type_list, self.num_opt_list, self.candidate_rels, self.out_keys, self.desc_index_map = \
            self.build_default_type_list(n_layers, edge_types, predict_keys=predict_keys, return_info=True)
        self.index_desc_map = {v: k for k, v in self.desc_index_map.items()}

    def build_default_option_dict(self, full_gnn_list, contain_zero):
        """
        Build the Option Space.
        :return:
        """
        option_dict = {}
        if full_gnn_list:
            option_dict["gnn"] = gnn_list_full  # gnn type
        else:
            option_dict["gnn"] = gnn_list_0_1  # gnn type
        option_dict["multi_aggr"] = multi_aggr_list  # multi-channel aggregator type

        if not contain_zero:  # exclude zero
            option_dict["gnn"] = option_dict["gnn"][1:]
        return option_dict

    def get_option_space(self):
        return self.option_dict

    def encode(self, gnn_desc: dict):
        """

        :param gnn_desc: dict,  {layer_0:..., layer_1:..., inter_modes:...}
        :return: code
        """
        n_ops = len(self.desc_index_map)
        edge_value = torch.zeros(n_ops, dtype=torch.long)

        for layer, layer_info in gnn_desc.items():
            for type, op in layer_info.items():
                op_name = (layer, *type)  # full name restore in the desc_index_map
                op_index = self.option_dict[type].index(op)
                index = self.desc_index_map[op_name]
                edge_value[index] = op_index

        return edge_value

    def decode(self, code, args=None):
        gnn_desc = defaultdict(OrderedDict)
        for type_full_name, index in self.desc_index_map.items():

            # build keys for gnn_desc
            layer_name = type_full_name[0]
            if len(type_full_name) > 2:
                type_name = tuple(type_full_name[1:])
            else:  # aggr
                type_name = type_full_name[1]
            op_index = code[index]
            op_name = self.option_dict[type_name][op_index]

            if "layer" not in layer_name:
                layer_name = "inter_modes"
                type_name = type_full_name
            gnn_desc[layer_name][type_name] = op_name
        return gnn_desc

    @staticmethod
    def get_output_keys(predict_keys, n_layers, edge_type, skip_conn=False):
        """
        Get the output keys of each layer.
        In heterogeneous graph, now all the nodes are required for downstream tasks.
        Take node classification in DBLP dataset, only representations of author node are required, similar to other layers.
        This function calculate the necessary output keys for each layer.
        :param predict_keys: output keys, list object
        :param n_layers: total layers of HGNN
        :param edge_type:  relation scheme of given dataset.
        :return
            candidate_relations: candidate relations on each layer that used to generate outputs
            out_keys: output keys of each layer.
        """
        out_keys = [set(predict_keys)] if isinstance(predict_keys, list) else [set([predict_keys])]
        #print(out_keys)
        candidate_relations = []
        total_ntype = out_keys[0]
        for i in range(n_layers):
            tmp_rels = []
            tmp_ntype = set()
            for each in edge_type:
                if (skip_conn and each[-1] in total_ntype) or each[-1] in out_keys[0]:
                    tmp_rels.append(each)
                    tmp_ntype.add(each[0])
            total_ntype = total_ntype.union(tmp_ntype)
            candidate_relations.insert(0, tmp_rels)
            out_keys.insert(0, tmp_ntype)
        return candidate_relations, out_keys

    def build_default_type_list(
            self,
            n_layers=2,
            edge_types=None,
            return_info=False,
            predict_keys=None
    ):
        edge_types = edge_types if edge_types is not None else self.edge_types
        predict_keys = predict_keys if predict_keys is not None else self.predict_keys
        assert edge_types is not None
        for etype in edge_types:
            assert isinstance(etype, tuple)

        candidate_rels, out_keys = self.get_output_keys(predict_keys, n_layers, edge_types)

        type_list = []
        num_opt_list = []
        desc_index_map = {}
        for i, relations in enumerate(candidate_rels):
            for e_name in relations:
                self.option_dict[e_name] = self.option_dict["gnn"]
                type_list.append(e_name)
                num_opt_list.append(len(self.option_dict["gnn"]))
                desc_name = (f"layer_{i}", *e_name)
                desc_index_map[desc_name] = len(desc_index_map)

            type_list.append("multi_aggr")
            num_opt_list.append(len(self.option_dict["multi_aggr"]))
            desc_name = (f"layer_{i}", "multi_aggr")
            desc_index_map[desc_name] = len(desc_index_map)

        if self.predict_inter_layer:
            for to_ in range(1, n_layers):
                for from_ in range(to_):
                    for etype in candidate_rels[to_]:
                        key = f"{from_} {to_}"
                        e_name = (key, *etype)
                        self.option_dict[e_name] = self.option_dict["gnn"]
                        type_list.append(e_name)
                        num_opt_list.append(len(self.option_dict["gnn"]))
                        desc_index_map[e_name] = len(desc_index_map)

        if return_info:
            return type_list, num_opt_list, candidate_rels, out_keys, desc_index_map
        else:
            return type_list

    def oplist2desc(self, action):
        """
        Translate a description list to a dict that is the input of gnn_manager.
        :param action:
        :param args:
        :return:
        """
        gnn_desc = {}
        index = 0
        for i in range(self.n_layers):
            layer_name = f"layer_{i}"
            gnn_desc[layer_name] = {}
            for etype in self.candidate_rels[i]:
                gnn_desc[layer_name][etype] = action[index]
                index += 1


            gnn_desc[layer_name][self.type_list[index]] = action[index]  # aggr
            index += 1
        start_index = index
        inter_mods = {}
        if self.predict_inter_layer:
            for name, action in zip(self.type_list[start_index:], action[start_index:]):
                inter_mods[name] = action
        gnn_desc["inter_modes"] = inter_mods
        return gnn_desc

    def generate_baselines(self, args):
        pass
        # gnn_desc = []
        # for action_name in self.type_list:
        #     gnn_desc.append(self.option_dict[action_name][0])
        # return [self.form_gnn_info(gnn_desc, args)]

    def sample(self, n_sample=1, return_index=True, no_repeat=False, sparse_rate=0.0):
        """
        randomly sample HGNNs from the search space
        :param n_sample:
        :param no_zero:
        :param return_index: if True, return the index of the option instead of its name
        :return:
        """
        origin_n_sample = n_sample
        cache = set()
        if no_repeat:
            n_sample = n_sample * 10
        no_zero = not self.contain_zero
        res = []
        for i in range(n_sample):
            gnn_desc = []
            code = []
            for action_name, n_opt in zip(self.type_list, self.num_opt_list):
                if no_zero and "aggr" not in action_name:
                    index = np.random.choice(range(1, n_opt))
                else:
                    if sparse_rate > 0:
                        if np.random.uniform() > sparse_rate:
                            index = np.random.choice(range(1, n_opt))
                        else:
                            index = 0
                    else:
                        index = np.random.choice(n_opt)  # select from zero and gcn_conv

                code.append(index)
                gnn_desc.append(self.option_dict[action_name][index])
            if no_repeat:
                signature = "_".join([str(each) for each in code])
                if signature not in cache:
                    cache.add(signature)
                    res.append(code if return_index else gnn_desc)
                else:
                    continue
            else:
                res.append(code if return_index else gnn_desc)
            if len(cache) >= origin_n_sample:
                break
        return res


if __name__ == "__main__":
    #n_layer = 3
    import argparse
    from hgnn.configs import register_hgnn_args
    from hgnn.meta_manager import AggrManager
    from link_predict.meta_manager import MetaOptLinkPredictorManager
    parser = argparse.ArgumentParser('HGNAS')
    register_hgnn_args(parser)

    args = parser.parse_args()
    #args.dataset = 'imdb'
    args.dataset = "amazon"
    args.task = "lp"
    args.use_feat = False
    args.metrics = "auc"
    args.debug = True
    from hgnn.meta_manager import HGNNLinkPredictorManager
    #gnn_manager_obj = AggrManager(args)
    gnn_manager_obj =HGNNLinkPredictorManager(args)
    edge_dict = gnn_manager_obj.edge_dict
    predict_keys = [gnn_manager_obj.pre_dst_type,gnn_manager_obj.pre_src_type,gnn_manager_obj.pre_link]
    #predict_keys = gnn_manager_obj.predict_keys
    obj = SearchSpace(edge_types=edge_dict, n_layers=2, predict_keys=predict_keys,predict_inter_layer=True,full_gnn_list=True)
    type_list = obj.build_default_type_list(n_layers=2)
    print(type_list)
    print(obj.get_option_space())
    print(obj.oplist2desc(type_list))
    print(obj.sample(1))
    print(obj.sample(1))
    code = obj.sample(1, return_index=False)[0]
    print(code,obj.oplist2desc(code))
    print(len(obj.sample(1)[0]))
    print(obj.oplist2desc(obj.sample(1, return_index=False)[0]))

    print(obj.decode(obj.sample(1, return_index=True, no_repeat=True)[0]))
