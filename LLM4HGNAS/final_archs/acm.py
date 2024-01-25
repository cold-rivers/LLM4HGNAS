import argparse
import time

import os
import psutil

from general_manager import GeneralManagerNC
from data.data_util import set_seed
import torch
import numpy as np
from hgnn.meta_manager import AggrManagerSK
from hgnn.configs import register_hgnn_args
from hyper_opt import search_hyper
from collections import OrderedDict
from nas.search_space import SearchSpace


best_config = {}


best_config["lr"] = 0.0001
best_config["dropout"] = 0.6
best_config["weight_decay"] = 0.005
best_config["n_hid"] = 512
best_config["epochs"] = 200
best_config["n_layer"] = 2

class HGNASManagerNC(GeneralManagerNC):

    def forward_model(self, G, inputs, model, out_key, require_embed=False):
        logits, embedding = model(G, inputs, out_key, return_embeding=True)
        if not require_embed:
            return logits
        else:
            return logits, embedding


parser = argparse.ArgumentParser('HGNAS')
register_hgnn_args(parser)
args = parser.parse_args()


def build_model(args, manager):
    gnn_desc = args.model_desc
    hgnn_manager = AggrManagerSK(args)
    model = hgnn_manager.build_gnn(gnn_desc)

    return model


def assign_config(best_config, args, manager):
    args.lr = best_config["lr"]
    args.dropout = best_config["dropout"]
    args.weight_decay = best_config["weight_decay"]
    args.n_hid = best_config["n_hid"]
    args.epochs = best_config["epochs"]
    args.n_layer = best_config["n_layer"]

    # print(args)
    # print(manager.args)


def main(args):
    device = torch.device("cuda:0") if args.cuda else torch.device("cuda:0")
    if args.task == "nc":
        manager = HGNASManagerNC(args, device)
    edge_dict = manager.edge_dict
    predict_keys = manager.predict_keys
    search_space = SearchSpace(
        edge_types=edge_dict, n_layers=2, predict_keys=predict_keys,
        full_gnn_list=True, contain_zero=True, predict_inter_layer=True,
    )

    model_list = []
    top_code = [[2, 1, 3, 0, 4, 2, 0, 4, 1, 3, 3, 2, 1, 0, 4]]
    # for i in range(len(top_code)):
    #     top_code[i] = dict(search_space.decode(top_code[i]))
    #     for key, ordered_dict in top_code[i].items():
    #         top_code[i][key] = {k: v for k, v in ordered_dict.items()}
    #     #top_code[i]['inter_modes'] = {}
    #     model_list.append(top_code[i])
    model_list = [{'layer_0': {('author', 'aa', 'author'): 'gat_conv', ('author', 'ap', 'paper'): 'gcn_conv', ('field', 'ff', 'field'): 'edge_conv',
                                ('field', 'fp', 'paper'): 'zero_conv', ('paper', 'pa', 'author'): 'sage_pool', ('paper', 'pf', 'field'): 'gat_conv', 
                                ('paper', 'pp', 'paper'): 'zero_conv', 'multi_aggr': 'min'}, 
                   'layer_1': {('author', 'ap', 'paper'): 'gcn_conv', ('field', 'fp', 'paper'): 'edge_conv', 
                               ('paper', 'pp', 'paper'): 'edge_conv', 'multi_aggr': 'mean'}, 
                   'inter_modes': {('0 1', 'author', 'ap', 'paper'): 'gcn_conv', ('0 1', 'field', 'fp', 'paper'): 'zero_conv', 
                                   ('0 1', 'paper', 'pp', 'paper'): 'sage_pool'}}]
    best_score = 0
    best_score_info = None
    best_model_info = None
    best_test_score =None
    i = 0
    best = 0
    for model in model_list:
        args.model_desc = OrderedDict(model)
        macro_score, micro_score = eval_model(args, device, manager)
        i = i + 1
        print("___________________")
        if best_score < macro_score.mean():
            best_score = macro_score.mean()
            best_score_info = macro_score
            best_test_score = micro_score
            best_model_info = model
            best = i
        print("model",i)
    print('Best Model:', best_model_info, best)
    print('Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
        macro_f1[:, 0].mean(), macro_f1[:, 1].mean(), train_size) for macro_f1, train_size in
        zip(best_score_info, [0.6, 0.4, 0.2])]))
    print('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
        micro_f1[:, 0].mean(), micro_f1[:, 1].mean(), train_size) for micro_f1, train_size in
        zip(best_test_score, [0.6, 0.4, 0.2])]))

def eval_model(args, device, manager):
    global best_config
    if len(best_config) == 0:
        best_config = search_hyper(args, manager, build_model, n_eval=50)
    assign_config(best_config, args, manager)
    print(args)
    val_res = []
    res = []
    svm_macro_f1_lists = []
    svm_micro_f1_lists = []
    set_seed(1)
    for i in range(10):
        model = build_model(args, manager)
        if args.task == "nc":
            # val_score, test_score = manager.evaluate(model, device, svm_test=False)
            torch.cuda.synchronize()
            start = time.time()
            val_score, test_score, svm_macro_f1_list, svm_micro_f1_list = manager.evaluate(model, device, svm_test=True)
            torch.cuda.synchronize()
            end = time.time()
            print("time:",  end - start)
            svm_macro_f1_lists.append(svm_macro_f1_list)
            svm_micro_f1_lists.append(svm_micro_f1_list)
        else:
            val_score, test_score = manager.evaluate(model, device)
        val_res.append(val_score)
        res.append(test_score)
    #print(f"Dataset:{args.dataset} {args.metrics}:{np.mean(res) * 100:.4f} +/- {np.std(res) * 100:.4f}")
    print( f"Dataset:{args.dataset} {args.metrics} "
                       f"val_score:{np.mean(val_res) * 100:.4f} +/- {np.std(val_res) * 100:.4f}, "
                       f"test_score:{np.mean(res) * 100:.4f} +/- {np.std(res) * 100:.4f}")
    if args.task == "nc":
        svm_macro_f1_lists = np.transpose(np.array(svm_macro_f1_lists), (1, 0, 2)) * 100
        svm_micro_f1_lists = np.transpose(np.array(svm_micro_f1_lists), (1, 0, 2)) * 100
        print('----------------------------------------------------------------')
        print('SVM tests summary')
        print('Macro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
            macro_f1[:, 0].mean(), macro_f1[:, 1].mean(), train_size) for macro_f1, train_size in
            zip(svm_macro_f1_lists, [0.6, 0.4, 0.2])]))
        print('Micro-F1: ' + ', '.join(['{:.6f}~{:.6f} ({:.1f})'.format(
            micro_f1[:, 0].mean(), micro_f1[:, 1].mean(), train_size) for micro_f1, train_size in
            zip(svm_micro_f1_lists, [0.6, 0.4, 0.2])]))
    return svm_macro_f1_lists, svm_micro_f1_lists


if __name__ == '__main__':
    args.dataset = "acm"
    args.task = "nc"
    args.use_feat = True
    args.metrics = "macro_f1"
    args.debug = True
    set_seed(1)
    print(args)
    main(args)
