import os

import numpy as np
import torch
from hyperopt import fmin, tpe, hp
from data.data_util import set_seed


lr_list = [1e-2, 1e-3, 1e-4, 5e-3, 5e-4]
dropout_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
decay_list = [0, 1e-3, 1e-4, 1e-5, 5e-5, 5e-4]
dim_list = [8, 16, 32, 64, 128, 256, 512]
epochs_list = [100, 200, 300, 400]
layer_list = [2]

param_dict = {
    'lr': lr_list,
    'dropout': dropout_list,
    'weight_decay': decay_list,
    'n_hid': dim_list,
    'epochs': epochs_list,
    'n_layer': layer_list,
}

space = {
    'lr': hp.choice("lr", lr_list),
    'dropout': hp.choice("dropout", dropout_list),
    'weight_decay': hp.choice("weight_decay", decay_list),
    'n_hid': hp.choice("n_hid", dim_list),
    'epochs': hp.choice("epochs", epochs_list),
    'n_layer': hp.choice("n_layer", layer_list),
}


def search_hyper(args, manager, build_model, n_eval=50):
    os.environ["HYPEROPT_FMIN_SEED"] = str(args.random_seed)
    set_seed(args.random_seed)

    def f(params):
        #接收超参数并且构建模型
        try:
            print(params)
            args.dropout = params["dropout"]
            args.lr = params["lr"]
            args.n_hid = params["n_hid"]
            args.weight_decay = params["weight_decay"]
            args.epochs = params["epochs"]
            args.n_layer = params["n_layer"]

            all_val_score = 0
            gnn = build_model(args, manager)
            val_score, _ = manager.evaluate(gnn)#评估
            all_val_score += val_score

            return 0 - all_val_score
        except Exception as e:
            print(e)
            if hasattr(args, "debug") and args.debug:
                raise e
            return 0
    #寻找最小，所以用0-all_val_score
    best = fmin(
        fn=f,
        space=space,
        algo=tpe.suggest,#选择下一个超参数组合进行评估
        max_evals=n_eval,
        show_progressbar=True,
        verbose=True
    )
    
    best_config = {}
    for k,v in best.items():
        best_config[k] = param_dict[k][v]
    return best_config


