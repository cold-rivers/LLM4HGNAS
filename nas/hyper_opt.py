import os

import numpy as np
import torch
from hyperopt import fmin, tpe, hp
from hgnn.utils import set_seed


lr_list = [1e-2, 1e-3, 1e-4, 5e-3, 5e-4]
dropout_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
decay_list = [0, 1e-3, 1e-4, 1e-5, 5e-5, 5e-4]
dim_list = [8, 16, 32, 64, 128, 256, 512]
epochs_list = [100, 200, 300, 400]
layer_list = [1, 2, 3, 4, 5, 6, 7]

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


def search_hyper(args, manager, build_model,  n_eval=50):
    os.environ["HYPEROPT_FMIN_SEED"] = str(args.random_seed)
    set_seed(args.random_seed)

    def f(params):
        print(params)
        manager.args.dropout = params["dropout"]
        manager.args.lr = params["lr"]
        manager.args.n_hid = params["n_hid"]
        manager.args.weight_decay = params["weight_decay"]
        manager.args.epochs = params["epochs"]
        if hasattr(args, "predict_n_layer") and args.predict_n_layer:
            manager.args.layers_of_child_model = params["n_layer"]

        all_val_score = 0
        model_list = build_model(args, manager)
        for gnn in model_list:
            val_score, _ = manager.evaluate(gnn)
            all_val_score += val_score
        return 0 - all_val_score

    best = fmin(
        fn=f,
        space=space,
        algo=tpe.suggest,
        max_evals=n_eval,
        show_progressbar=False)
    print(best)

    best_config = {}
    for k,v in best.items():
        best_config[k] = param_dict[k][v]

    manager.args.dropout = best_config["dropout"]
    manager.args.lr = best_config["lr"]
    manager.args.n_hid = best_config["n_hid"]
    manager.args.weight_decay = best_config["weight_decay"]
    manager.args.epochs = best_config["epochs"]
    if hasattr(args, "predict_n_layer") and args.predict_n_layer:
        manager.args.layers_of_child_model = best_config["n_layer"]
    print(args)
    return manager


if __name__ == "__main__":
    from archi2vec.main_with_mlp_aggr import args, channels, OPT_CONFIG
    from hgnn.meta_manager import MetaOptManager
    from archi2vec.generate_archs import ClassificationBench

    bench = ClassificationBench(n_rel=len(channels["reverse_edge_dict"]),
                                n_opt=len(OPT_CONFIG["reverse_opt_map"]),
                                n_layer=2,
                                reverse_edge_dict=channels["reverse_edge_dict"],
                                opt_dict=OPT_CONFIG["reverse_opt_map"],
                                edge_dict=channels["edge_dict"],
                                out_key=channels["out_key"])
    adj, opt = bench.generate_baseline()
    baseline = bench.to_gnn_desc(adj[0], opt[0])
    gnn_manager = MetaOptManager(args)
    search_hyper(args, gnn_manager, [baseline], seed=123, n_eval=50)

    best_test_score = []
    for _ in range(10):
        best_test_score.append(gnn_manager.evaluate(baseline)[1])
    print(f"best_gnn:{baseline}, test_score:{np.mean(best_test_score)} +/- {np.std(best_test_score)}")
