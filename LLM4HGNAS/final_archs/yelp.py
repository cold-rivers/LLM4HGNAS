import argparse
from general_manager import GeneralManagerLP
from data.data_util import set_seed
import torch
import numpy as np
from link_predict.meta_manager import MetaOptLinkPredictorManager
from hgnn.configs import register_hgnn_args
from hyper_opt import search_hyper
from nas.search_space import SearchSpace
best_config = {}
# Best configs for Top 1 HGNN
best_config["lr"] = 0.01
best_config["dropout"] = 0.0
best_config["weight_decay"] = 5e-05
best_config["n_hid"] = 256
best_config["epochs"] = 200
best_config["n_layer"] = 2

parser = argparse.ArgumentParser('HGNAS')
register_hgnn_args(parser)
args = parser.parse_args()


def build_model(args, manager):
    gnn_desc = args.model_desc
    hgnn_manager = MetaOptLinkPredictorManager(args)
    model = hgnn_manager.build_gnn(gnn_desc)

    return model


def assign_config(best_config, args, manager):
    args.lr = best_config["lr"]
    args.dropout = best_config["dropout"]
    args.weight_decay = best_config["weight_decay"]
    args.n_hid = best_config["n_hid"]
    args.epochs = best_config["epochs"]
    args.n_layer = best_config["n_layer"]

    print(args)
    print(manager.args)


def main(args):
    global best_config

    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
    manager = GeneralManagerLP(args, device)

    gnn_manager_obj = MetaOptLinkPredictorManager(args)
    edge_dict = gnn_manager_obj.edge_dict
    predict_keys = [gnn_manager_obj.pre_dst_type, gnn_manager_obj.pre_src_type, gnn_manager_obj.pre_link]
    search_space = SearchSpace(
        edge_types=edge_dict, n_layers=2, predict_keys=predict_keys,
        full_gnn_list=True, contain_zero=True, predict_inter_layer=True,
    )

    import json
    import heapq

    top_code = [[3, 1, 1, 1, 1, 1, 1, 1, 3, 2, 1, 2, 1, 1, 0, 3, 4, 1, 4, 4, 4, 1, 4, 0, 4, 2, 1, 1, 1, 2, 2, 3]]
    data = top_code

    model_list = [{'layer_0': {('business', 'bb', 'business'): 'edge_conv', ('business', 'buca', 'category'): 'gcn_conv',
                                ('business', 'buci', 'city'): 'gcn_conv', ('business', 'buus', 'user'): 'gcn_conv', 
                                ('category', 'cabu', 'business'): 'gcn_conv', ('category', 'cc', 'category'): 'gcn_conv', 
                                ('city', 'cc', 'city'): 'gcn_conv', ('city', 'cibu', 'business'): 'gcn_conv', 
                                ('compliment', 'cc', 'compliment'): 'edge_conv',('compliment', 'cous', 'user'): 'gat_conv', 
                                ('user', 'usbu', 'business'): 'gcn_conv', ('user', 'usco', 'compliment'): 'gat_conv', 
                                ('user', 'usus', 'user'): 'gcn_conv', ('user', 'uu', 'user'): 'gcn_conv', 'multi_aggr': 'sum'},
                    'layer_1': {('business', 'bb', 'business'): 'edge_conv', ('business', 'buus', 'user'): 'sage_pool', 
                                ('category', 'cabu', 'business'): 'gcn_conv', ('city', 'cibu', 'business'): 'sage_pool', 
                                ('compliment', 'cous', 'user'): 'sage_pool', ('user', 'usbu', 'business'): 'sage_pool', 
                                ('user', 'usus', 'user'): 'gcn_conv', ('user', 'uu', 'user'): 'sage_pool', 'multi_aggr': 'sum'}, 
                   'inter_modes': {('0 1', 'business', 'bb', 'business'): 'sage_pool', ('0 1', 'business', 'buus', 'user'): 'gat_conv', 
                                   ('0 1', 'category', 'cabu', 'business'): 'gcn_conv', ('0 1', 'city', 'cibu', 'business'): 'gcn_conv', 
                                   ('0 1', 'compliment', 'cous', 'user'): 'gcn_conv', ('0 1', 'user', 'usbu', 'business'): 'gat_conv',
                                   ('0 1', 'user', 'usus', 'user'): 'gat_conv', ('0 1', 'user', 'uu', 'user'): 'edge_conv'}}]

    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
    manager = GeneralManagerLP(args, device)
    # for i in range(len(data)):
    #     data[i] = dict(search_space.decode(data[i]))

    #     for key, ordered_dict in data[i].items():
    #         data[i][key] = {k: v for k, v in ordered_dict.items()}
    #     args.model_desc = data
    #     model_list.append(data[i])
    result = []
    result_list = []
    res_list = []
    res_val_list = []
    for model,code in zip(model_list,top_code):
        args.model_desc = model
        set_seed(1)
        print("model:",code)
        res,val_res,result = eval_model(args, device, manager)
        result_list.append(result)
        res_list.append(np.mean(res))
        res_val_list.append(np.mean(val_res))


def eval_model(args, device, manager):
    global best_config
    if len(best_config) == 0:
        best_config = search_hyper(args, manager, build_model, n_eval=50)
    assign_config(best_config, args, manager)
    print(args)
    val_res = []
    res = []
    for i in range(10):
        model = build_model(args, manager)
        val_score, test_score = manager.evaluate(model, device)
        val_res.append(val_score)
        res.append(test_score)
    print(f"Dataset:{args.dataset} {args.metrics} "
          f"val_score:{np.mean(val_res) * 100:.4f} +/- {np.std(val_res) * 100:.4f}, "
          f"test_score:{np.mean(res) * 100:.4f} +/- {np.std(res) * 100:.4f}")
    return res, val_res,val_score, test_score


if __name__ == '__main__':
    args.dataset = "Yelp"
    args.task = "lp"
    args.use_feat = False
    args.metrics = "auc"
    args.debug = True
    args.seed =1
    set_seed(args.seed)
    print(args)
    main(args)
