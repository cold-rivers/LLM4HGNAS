def register_hgnn_args(parser, dataset="acm"):
    parser.add_argument('-s', '--random_seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--dataset', type=str, default=dataset)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--n_hid', type=int, default=256)
    parser.add_argument('--n_inp', type=int, default=256)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--self_loop', type=bool, default=True)
    parser.add_argument('--metrics', type=str, default="macro_f1")
    parser.add_argument('--use_feat', type=bool, default=True)
    parser.add_argument('--task', type=str, choices=["nc", "lp"], default="nc")
    parser.add_argument('--debug', action="store_true", default=False)
