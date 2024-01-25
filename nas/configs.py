import argparse


def register_default_args(parser: argparse.ArgumentParser,
                          epochs=200,
                          lr=0.005,
                          weight_decay=5e-4,
                          dropout=0.6,
                          layers_of_child_model=2,
                          with_details=False,
                          controller_lr=3.5e-3,
                          dataset="Citeseer"):
    # common arguments
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'derive'],
                        help='train: Training GraphNAS, derive: Deriving Architectures')
    parser.add_argument("--cuda", type=bool, default=True, required=False,
                        help="run in cuda mode")
    parser.add_argument('--search_mode', type=str, default='macro')
    parser.add_argument('--format', type=str, default='origin')

    trainer_parser = parser.add_argument_group(title="Trainer options")
    controller_parser = parser.add_argument_group(title="Controller options")
    gnn_parser = parser.add_argument_group(title="GNN options")
    # args of controller
    trainer_parser.add_argument('--max_epoch', type=int, default=3)
    trainer_parser.add_argument('--save_epoch', type=int, default=2)
    trainer_parser.add_argument('--max_save_num', type=int, default=5)
    trainer_parser.add_argument('--load_path', type=str, default='')
    trainer_parser.add_argument('--batch_size', type=int, default=10)
    trainer_parser.add_argument('--controller_max_step', type=int, default=100,
                                help='step for controller parameters')
    trainer_parser.add_argument('--controller_optim', type=str, default='adam')
    trainer_parser.add_argument('--controller_lr', type=float, default=controller_lr)
    trainer_parser.add_argument('--derive_num_sample', type=int, default=100)
    trainer_parser.add_argument('--derive_finally', type=bool, default=True)
    trainer_parser.add_argument('--derive_from_history', type=bool, default=True)

    # for build reward
    controller_parser.add_argument('--layers_of_child_model', type=int, default=layers_of_child_model)
    controller_parser.add_argument('--entropy_mode', type=str, default='reward', choices=['reward', 'regularizer'])
    controller_parser.add_argument('--entropy_coeff', type=float, default=1e-4)
    controller_parser.add_argument('--ema_baseline_decay', type=float, default=0.95)
    controller_parser.add_argument('--discount', type=float, default=1.0)
    controller_parser.add_argument('--controller_grad_clip', type=float, default=0)
    controller_parser.add_argument('--tanh_c', type=float, default=2.5)
    controller_parser.add_argument('--softmax_temperature', type=float, default=5.0)
    controller_parser.add_argument('--controller_hid', type=int, default=100)

    # child model
    gnn_parser.add_argument("--dataset", type=str, default=dataset, required=False,
                            help="The input dataset.")
    gnn_parser.add_argument("--epochs", type=int, default=epochs,
                            help="number of training epochs")
    gnn_parser.add_argument("--multi_label", type=bool, default=False,
                            help="multi_label or single_label task")
    gnn_parser.add_argument("--dropout", type=float, default=dropout,
                            help="input feature dropout")
    gnn_parser.add_argument("--lr", type=float, default=lr,
                            help="learning rate")
    gnn_parser.add_argument('--weight_decay', type=float, default=weight_decay)
    gnn_parser.add_argument('--max_param', type=float, default=5E6)
    gnn_parser.add_argument('--inductive', default=False, action='store_true')
    gnn_parser.add_argument('--gnn_log_file', type=str, default="gnn_log.log")

    if with_details:
        return parser, trainer_parser, controller_parser, gnn_parser
    else:
        return parser


def build_args():
    parser = argparse.ArgumentParser(description='GraphNAS')
    register_default_args(parser, dataset="acm")
    parser.add_argument('--predict_hyper',  action="store_true", default=False)
    parser.add_argument('--predict_inter_layer', default=True)  # TODO
    parser.add_argument('--n_hid', type=int, default=256)
    parser.add_argument('--n_inp', type=int, default=256)
    parser.add_argument('--clip', type=int, default=5.0)
    parser.add_argument('--use_feat', action="store_true", default=False)
    parser.add_argument('--metrics', type=str, default="macro_f1")
    parser.add_argument('--self_loop', action="store_true", default=False)
    parser.add_argument('--show_info', action="store_true", default=False)
    parser.add_argument('--full_gnn_list', action="store_true", default=False)
    parser.add_argument('--predict_n_layer', action="store_true", default=False)
    parser.add_argument('--sparse_rate', type=float, default=0.5)
    args = parser.parse_args()
    args.format = "meta_opt_1"
    # args.dataset = "dblp"
    # args.self_loop = True
    return args
