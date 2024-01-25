import argparse
from data.data_util import set_seed
import numpy as np
from link_predict.meta_manager import MetaOptLinkPredictorManager
from hgnn.configs import register_hgnn_args
from final_archs.hyper_opt import search_hyper
import torch
from final_archs.general_manager import GeneralManagerLP
from nas.search_space import SearchSpace

best_config = {}
# #s
best_config["lr"] = 0.01
best_config["dropout"] = 0.0
best_config["weight_decay"] = 5e-05
best_config["n_hid"] = 512
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

    top_code = [[0, 0, 2, 0, 0, 0, 2, 0, 0, 4, 0, 1, 1, 1, 0, 2, 4, 3, 0, 2, 4, 1, 4, 4, 2, 0, 3, 0, 0, 2, 3, 0, 4, 4, 0]]
    # for i in range(len(top_code)):
    #     temp = dict(search_space.decode(top_code[i]))
    #     print(i)
    #     for key, ordered_dict in temp.items():
    #         temp[key] = {k: v for k, v in ordered_dict.items()}
    #     #args.modelsc = d_deata[i]
    #     model_list.append(temp)
    model_list = [{'layer_0': {('age', 'aa', 'age'): 'zero_conv', ('age', 'agus', 'user'): 'zero_conv', ('genre', 'gemo', 'movie'): 'gat_conv', 
                               ('genre', 'gg', 'genre'): 'zero_conv', ('movie', 'mm', 'movie'): 'zero_conv', ('movie', 'moge', 'genre'): 'zero_conv', 
                               ('movie', 'momo', 'movie'): 'gat_conv', ('movie', 'mous', 'user'): 'zero_conv', ('occupation', 'ocus', 'user'): 'zero_conv', 
                               ('occupation', 'oo', 'occupation'): 'sage_pool', ('user', 'usag', 'age'): 'zero_conv', ('user', 'usmo', 'movie'): 'gcn_conv',
                                 ('user', 'usoc', 'occupation'): 'gcn_conv', ('user', 'usus', 'user'): 'gcn_conv', ('user', 'uu', 'user'): 'zero_conv', 
                                 'multi_aggr': 'mean'}, 
                    'layer_1': {('age', 'agus', 'user'): 'sage_pool', ('genre', 'gemo', 'movie'): 'edge_conv', ('movie', 'mm', 'movie'): 'zero_conv', 
                                ('movie', 'momo', 'movie'): 'gat_conv', ('movie', 'mous', 'user'): 'sage_pool', ('occupation', 'ocus', 'user'): 'gcn_conv', 
                                ('user', 'usmo', 'movie'): 'sage_pool', ('user', 'usus', 'user'): 'sage_pool', ('user', 'uu', 'user'): 'gat_conv',
                                  'multi_aggr': 'sum'}, 
                    'inter_modes': {('0 1', 'age', 'agus', 'user'): 'edge_conv', ('0 1', 'genre', 'gemo', 'movie'): 'zero_conv',
                                     ('0 1', 'movie', 'mm', 'movie'): 'zero_conv', ('0 1', 'movie', 'momo', 'movie'): 'gat_conv',
                                       ('0 1', 'movie', 'mous', 'user'): 'edge_conv', ('0 1', 'occupation', 'ocus', 'user'): 'zero_conv', 
                                       ('0 1', 'user', 'usmo', 'movie'): 'sage_pool', ('0 1', 'user', 'usus', 'user'): 'sage_pool', 
                                       ('0 1', 'user', 'uu', 'user'): 'zero_conv'}}]
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
    args.dataset = "Movielens"
    args.task = "lp"
    args.use_feat = False
    args.metrics = "auc"
    args.cycle_lr = True
    args.n_layer =2
    set_seed(1)
    print(args)
    main(args)
