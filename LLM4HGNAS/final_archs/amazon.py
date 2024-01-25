import argparse
from data.data_util import set_seed
import numpy as np
from link_predict.meta_manager import MetaOptLinkPredictorManager
from hgnn.configs import register_hgnn_args
from hyper_opt import search_hyper
import torch
from general_manager import GeneralManagerLP
from nas.search_space import SearchSpace

best_config = {}
# #s
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

    model_list = []

    gnn_manager_obj = MetaOptLinkPredictorManager(args)

    edge_dict = gnn_manager_obj.edge_dict
    predict_keys = [gnn_manager_obj.pre_dst_type, gnn_manager_obj.pre_src_type, gnn_manager_obj.pre_link]

    search_space = SearchSpace(
        edge_types=edge_dict, n_layers=2, predict_keys=predict_keys,
        full_gnn_list=True, contain_zero=not False, predict_inter_layer=True
    )

    top_code = [[1, 1, 1, 1, 3, 0, 1, 2, 3, 4, 0, 1, 3, 0, 4, 0, 1, 4, 4, 2, 3, 0, 4, 4, 2, 4, 4, 2, 4]]
    # for i in range(len(top_code)):
    #     temp = dict(search_space.decode(top_code[i]))
    #     print(i)
    #     for key, ordered_dict in temp.items():
    #         temp[key] = {k: v for k, v in ordered_dict.items()}
    #     #args.modelsc = d_deata[i]
    #     model_list.append(temp)
    model_list = [{'layer_0': {('brand', 'bb', 'brand'): 'gcn_conv', ('brand', 'brit', 'item'): 'gcn_conv', ('category', 'cait', 'item'): 'gcn_conv', 
                               ('category', 'cc', 'category'): 'gcn_conv', ('item', 'ii', 'item'): 'sage_pool', ('item', 'itbr', 'brand'): 'zero_conv',
                                 ('item', 'itca', 'category'): 'gcn_conv', ('item', 'itus', 'user'): 'gat_conv', ('item', 'itvi', 'view'): 'edge_conv', 
                                 ('user', 'usit', 'item'): 'sage_pool', ('user', 'uu', 'user'): 'zero_conv', ('view', 'viit', 'item'): 'gcn_conv', 
                                 ('view', 'vv', 'view'): 'edge_conv', 'multi_aggr': 'sum'},
                'layer_1': {('brand', 'brit', 'item'): 'sage_pool', ('category', 'cait', 'item'): 'zero_conv', ('item', 'ii', 'item'): 'gcn_conv', 
                            ('item', 'itus', 'user'): 'sage_pool', ('user', 'usit', 'item'): 'sage_pool', ('user', 'uu', 'user'): 'gat_conv', 
                            ('view', 'viit', 'item'): 'edge_conv', 'multi_aggr': 'sum'}, 
                   'inter_modes': {('0 1', 'brand', 'brit', 'item'): 'sage_pool', ('0 1', 'category', 'cait', 'item'): 'sage_pool', 
                                   ('0 1', 'item', 'ii', 'item'): 'gat_conv', ('0 1', 'item', 'itus', 'user'): 'sage_pool', 
                                   ('0 1', 'user', 'usit', 'item'): 'sage_pool', ('0 1', 'user', 'uu', 'user'): 'gat_conv', 
                                   ('0 1', 'view', 'viit', 'item'): 'gat_conv'}}]
    result_list = []
    res_list = []
    res_val_list = []
    for model,code in zip(model_list,top_code):
        args.model_desc = model
        set_seed(0)
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
    return res, val_res ,{"val_score":{np.mean(val_res),np.std(val_res)},"test_score":{np.mean(res),np.std(res)}}



if __name__ == '__main__':
    args.repeats = 10
    args.dataset = "Amazon"
    args.task = "lp"
    args.use_feat = False
    args.metrics = "auc"
    args.cycle_lr = True
    args.n_layer =2
    set_seed(0)
    print(args)
    main(args)
