import os.path as osp
import pickle
from pprint import pprint
import os
import dgl
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from dgl.data.utils import download, get_download_dir, _get_dgl_url
from scipy import io as sio
from scipy import sparse
from scipy.sparse import load_npz
from torch.nn.functional import one_hot
import logging
from collections import OrderedDict
from dgl.data.utils import save_graphs, load_graphs, save_info, load_info
from scipy.sparse import csr_matrix


def load_data(dataset="acm", self_loop=False):
    dataset = dataset.lower()
    base_path = osp.dirname(__file__)
    if dataset == "dblp":
        return load_DBLP(prefix=base_path + "/../input/DBLP_processed/", train_size=400, self_loop=self_loop)
    elif dataset.lower() == "acm":
        return load_acm_raw(self_loop=self_loop, train_size=400, val_size=400)


def load_DBLP(prefix="input/DBLP_processed/",
              rel_names=[('paper', 'pa', 'author'), ("paper", "pt", "term"), ("paper", "pc", "conference")],
              rel_filenames=["p_vs_a.npy", "p_vs_t.npy", "p_vs_c.npy"],
              labels_file="labels.npy",
              feature_map={
                  "author": "feature_a.npz",
                  "paper": "feature_p.npz",
                  "term": "feature_t.npz",
                  "conference": "feature_c.npz",
              },
              shuffle_data=False,
              train_size=400,
              val_size=400,
              predict_key="author",
              self_loop=False):
    dgl_graph_file = f"{prefix}/data_{self_loop}.bin"
    info_path = f"{prefix}/info_{self_loop}.pk"
    if os.path.exists(dgl_graph_file):
        G, edge_dict, feat_sizes, inputs, labels, node_dict, test_idx, train_idx, val_idx = load_dgl_dataset(
            dgl_graph_file, info_path)
    else:
        rel_edges_dict = OrderedDict()
        edge_dict = OrderedDict()
        for rel_name, rel_file in zip(rel_names, rel_filenames):
            relations = np.load(prefix + rel_file)
            rel_edges_dict[rel_name] = torch.LongTensor(relations).nonzero(as_tuple=True)
            revserse_rel = (rel_name[-1], rel_name[1][::-1], rel_name[0])
            rel_edges_dict[revserse_rel] = torch.LongTensor(relations).t().nonzero(as_tuple=True)
            # TODO 修复BUG，对现有模型没有影响，现有模型只需要有边的信息，不需要边的ID
            # reverse_edge_dict[rel_name] = len(reverse_edge_dict)

        G = dgl.heterograph(rel_edges_dict)
        if self_loop:
            for each in G.ntypes:
                self_loop_name = (each, each[0] + each[0], each)
                rel_edges_dict[self_loop_name] = (
                    torch.arange(G.number_of_nodes(each)),
                    torch.arange(G.number_of_nodes(each))
                )
            G = dgl.heterograph(rel_edges_dict)
        labels = np.load(prefix + labels_file)
        # OH-labels
        # labels_oh = np.zeros((labels.size, labels.max() + 1))
        # labels_oh[np.arange(labels.size), labels] = 1
        # load features
        inputs = OrderedDict()
        node_dict = OrderedDict()
        feat_sizes = OrderedDict()
        for feat_name in feature_map:
            G.nodes[feat_name].data["h"] = torch.Tensor(
                scipy.sparse.load_npz(prefix + feature_map[feat_name]).toarray())
            inputs[feat_name] = G.nodes[feat_name].data["h"]
            node_dict[feat_name] = len(node_dict)
            feat_sizes[feat_name] = inputs[feat_name].size(-1)
        # 为每天边设置比边类型ID
        for etype in G.canonical_etypes:
            edge_dict[etype] = len(edge_dict)
            # G.edges[etype].data['id'] = torch.ones(G.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]

        if shuffle_data:
            shuffle = np.random.permutation(np.arange(G.num_nodes(predict_key)))
        else:
            shuffle = np.arange(G.num_nodes(predict_key))
        num_train = train_size
        num_val = val_size
        train_idx = torch.tensor(shuffle[0:num_train]).long()
        val_idx = torch.tensor(shuffle[num_train:num_train + num_val]).long()
        test_idx = torch.tensor(shuffle[num_train + num_val:]).long()
        save_dgl_dataset(G, dgl_graph_file, edge_dict, feat_sizes, info_path, labels, node_dict, test_idx,
                         train_idx, val_idx)
    return G, inputs, torch.LongTensor(
        labels), node_dict, edge_dict, train_idx, val_idx, test_idx, predict_key, feat_sizes


def load_acm_raw(train_size=400, val_size=400, self_loop=False):
    dgl_graph_file = f"{get_download_dir()}/ACM/data_{self_loop}.bin"
    info_path = f"{get_download_dir()}/ACM/info_{self_loop}.pk"
    if os.path.exists(dgl_graph_file):
        hg, edge_dict, feat_sizes, inputs, labels, node_dict, test_idx, train_idx, val_idx = load_dgl_dataset(
            dgl_graph_file, info_path)
    else:
        url = 'dataset/ACM.mat'
        data_path = get_download_dir() + '/ACM.mat'
        if not osp.exists(data_path):
            download(_get_dgl_url(url), path=data_path)
        data = sio.loadmat(data_path)
        p_vs_l = data['PvsL']  # paper-field?
        p_vs_a = data['PvsA']  # paper-author
        p_vs_t = data['PvsT']  # paper-term, bag of words
        p_vs_c = data['PvsC']  # paper-conference, labels come from that

        # We assign
        # (1) KDD papers as class 0 (data mining),
        # (2) SIGMOD and VLDB papers as class 1 (database),
        # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
        conf_ids = [0, 1, 9, 10, 13]
        label_ids = [0, 1, 2, 2, 1]

        p_vs_c_filter = p_vs_c[:, conf_ids]
        p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
        p_vs_l = p_vs_l[p_selected]
        p_vs_a = p_vs_a[p_selected]
        p_vs_t = p_vs_t[p_selected]
        p_vs_c = p_vs_c[p_selected]

        rel_dict = {
            ('paper', 'pa', 'author'): p_vs_a.nonzero(),
            ('author', 'ap', 'paper'): p_vs_a.transpose().nonzero(),
            ('paper', 'pf', 'field'): p_vs_l.nonzero(),
            ('field', 'fp', 'paper'): p_vs_l.transpose().nonzero()
        }
        hg = dgl.heterograph(rel_dict)

        # view self loop as one kind of channels
        if self_loop:
            for each in hg.ntypes:
                self_loop_name = (each, each[0] + each[0], each)
                rel_dict[self_loop_name] = (
                    torch.arange(hg.number_of_nodes(each), dtype=torch.int64),
                    torch.arange(hg.number_of_nodes(each), dtype=torch.int64)
                )
            hg = dgl.heterograph(rel_dict)

        features = torch.FloatTensor(p_vs_t.toarray())

        pc_p, pc_c = p_vs_c.nonzero()
        labels = np.zeros(len(p_selected), dtype=np.int64)
        for conf_id, label_id in zip(conf_ids, label_ids):
            labels[pc_p[pc_c == conf_id]] = label_id
        labels = torch.LongTensor(labels)

        shuffle = np.random.permutation(pc_p)
        train_idx = torch.LongTensor(shuffle[:train_size])
        val_idx = torch.LongTensor(shuffle[train_size:train_size + val_size])
        test_idx = torch.LongTensor(shuffle[train_size + val_size:])

        num_nodes = hg.number_of_nodes('paper')
        train_mask = get_binary_mask(num_nodes, train_idx)
        val_mask = get_binary_mask(num_nodes, val_idx)
        test_mask = get_binary_mask(num_nodes, test_idx)

        inputs = OrderedDict()
        num_author = hg.number_of_nodes('author')
        num_field = hg.number_of_nodes('field')
        inputs["paper"] = features
        inputs["author"] = one_hot(torch.arange(num_author)).float()
        inputs["field"] = one_hot(torch.arange(num_field)).float()
        feat_sizes = OrderedDict()
        for feat_name in inputs:
            feat_sizes[feat_name] = inputs[feat_name].size(-1)

        for each in inputs:
            hg.nodes[each].data['h'] = inputs[each]

        node_dict = OrderedDict()
        for ntype in hg.ntypes:
            node_dict[ntype] = len(node_dict)
        edge_dict = OrderedDict()
        for etype in hg.canonical_etypes:
            edge_dict[etype] = len(edge_dict)
            hg.edges[etype].data['id'] = torch.ones(hg.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]
        save_dgl_dataset(hg, dgl_graph_file, edge_dict, feat_sizes, info_path, labels, node_dict, test_idx,
                         train_idx, val_idx)
    return hg, inputs, labels, node_dict, edge_dict, train_idx, val_idx, test_idx, "paper", feat_sizes


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()


def set_seed(seed):
    print("random seed:", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    dgl.seed(seed)


def get_logger(name=__file__, level=logging.INFO):
    logger = logging.getLogger(name)

    if getattr(logger, '_init_done__', None):
        logger.setLevel(level)
        return logger

    logger._init_done__ = True
    logger.propagate = False
    logger.setLevel(level)

    formatter = logging.Formatter("%(asctime)s:%(levelname)s::%(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(0)

    del logger.handlers[:]
    logger.addHandler(handler)

    return logger


def save_dgl_dataset(G, dgl_graph_file, edge_dict, feat_sizes, info_path, labels_tensor, node_dict, test_node,
                     train_node, valid_node):
    save_graphs(dgl_graph_file, G)
    infos = {}
    infos["labels"], infos["node_dict"], infos["edge_dict"], infos["train_node"], infos["valid_node"], \
    infos["test_node"], infos["feat_sizes"] = \
        labels_tensor, node_dict, edge_dict, train_node, valid_node, test_node, feat_sizes
    save_info(info_path, infos)


def load_dgl_dataset(dgl_graph_file, info_path):
    G, _ = load_graphs(dgl_graph_file)
    G = G[0]
    inputs = G.ndata["h"]
    infos = load_info(info_path)
    labels_tensor, node_dict, edge_dict, train_node, valid_node, test_node, feat_sizes = \
        infos["labels"], infos["node_dict"], infos["edge_dict"], infos["train_node"], infos["valid_node"], \
        infos["test_node"], infos["feat_sizes"]
    return G, edge_dict, feat_sizes, inputs, labels_tensor, node_dict, test_node, train_node, valid_node


if __name__ == "__main__":
    # hg, features, labels, node_dict, edge_dict, train_idx, val_idx, test_idx, category, feat_sizes = load_data("acm")
    dataset_list = ["dblp", "acm"]
    for dataset in dataset_list:
        load_data(dataset)
