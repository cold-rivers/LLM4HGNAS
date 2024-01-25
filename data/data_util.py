import os

import pandas as pd
import scipy.sparse as sp

base_path = os.path.dirname(__file__) + "/"

import os.path as osp
import pickle
import os
import dgl
import numpy as np
import scipy.io
import torch
from dgl.data.utils import download, get_download_dir, _get_dgl_url
from scipy import io as sio
from torch.nn.functional import one_hot
import logging
from collections import OrderedDict
from dgl.data.utils import save_graphs, load_graphs, save_info, load_info
from scipy.sparse import csr_matrix
from sklearn.utils import check_random_state
from openhgnn.dataset import GTNDataset
from dgl.data.utils import idx2mask

predict_links = {
    "Yelp": ("user", "business", ("user", "usbu", "business")),
    "Amazon": ("user", "item", ("user", "usit", "item")),
    "Douban": ("user", "movie", ('user', 'usmo', 'movie')),
    "Movielens": ("user", "movie", ('user', 'usmo', 'movie')),
}
rating_threshold_dict = {
    "Yelp": 3,
    "Amazon": 3,
    "Douban": 3,
    "Movielens":3,
}


def set_seed(seed):
    print("random seed:", seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
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


def load_data(dataset="acm", self_loop=True):
    dataset = dataset.lower()
    base_path = osp.dirname(__file__)
    if dataset.lower() == "acm":
        return load_acm_raw(self_loop=self_loop, train_size=400, val_size=400)
    elif dataset.lower() == "dblp":
        return load_DBLP(prefix=base_path + "/dblp/DBLP_processed/", self_loop=self_loop, train_size=400, val_size=400)
    elif dataset.lower() == "imdb":
        return load_imdb(self_loop=self_loop)
    else:
        raise RuntimeError("Not support such dataset:", dataset)


def load_acm_raw(train_size=400, val_size=400, self_loop=False):
    # offered by DGL
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

        num_classes = 3
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
    #print(hg, inputs, labels, len(labels),node_dict, edge_dict, train_idx, val_idx, test_idx, "paper", feat_sizes)
    return hg, inputs, labels, node_dict, edge_dict, train_idx, val_idx, test_idx, "paper", feat_sizes


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


def load_imdb(self_loop=False):
    dataset = GTNDataset(name='imdb4GTN')
    target_ntype = 'movie'
    g = dataset[0]
    canonical_etypes = [('movie', 'md', 'director'), ('director', 'dm', 'movie'),
                                ('movie', 'ma', 'actor'), ('actor', 'am', 'movie')]
    dgl_graph_file = f"/IMDB/data_{self_loop}.bin"
    info_path = f"/IMDB/ACM/info_{self_loop}.pk"
    if os.path.exists(dgl_graph_file):
        hg, edge_dict, feat_sizes, inputs, labels, node_dict, test_idx, train_idx, val_idx = load_dgl_dataset(
            dgl_graph_file, info_path)
    else:
        with open('data/IMDB/node_features.pkl', 'rb') as f:
                node_features = pickle.load(f)
        with open('data/IMDB/edges.pkl', 'rb') as f:
                edges = pickle.load(f)
        with open('data/IMDB/labels.pkl', 'rb') as f:
                labels = pickle.load(f)
        num_nodes = edges[0].shape[0]
        assert len(canonical_etypes) == len(edges)
        ntype_mask = dict()
        ntype_idmap = dict()
        ntypes = set()
        data_dict = {}

        # create dgl graph
        for etype in canonical_etypes:
            ntypes.add(etype[0])
            ntypes.add(etype[2])
        for ntype in ntypes:
            ntype_mask[ntype] = np.zeros(num_nodes, dtype=bool)
            ntype_idmap[ntype] = np.full(num_nodes, -1, dtype=int)
        for i, etype in enumerate(canonical_etypes):
            src_nodes = edges[i].nonzero()[0]
            dst_nodes = edges[i].nonzero()[1]
            src_ntype = etype[0]
            dst_ntype = etype[2]
            ntype_mask[src_ntype][src_nodes] = True
            ntype_mask[dst_ntype][dst_nodes] = True
        for ntype in ntypes:
            ntype_idx = ntype_mask[ntype].nonzero()[0]
            ntype_idmap[ntype][ntype_idx] = np.arange(ntype_idx.size)
        for i, etype in enumerate(canonical_etypes):
            src_nodes = edges[i].nonzero()[0]
            dst_nodes = edges[i].nonzero()[1]
            src_ntype = etype[0]
            dst_ntype = etype[2]
            data_dict[etype] = \
                (torch.from_numpy(ntype_idmap[src_ntype][src_nodes]).type(torch.int64),
                torch.from_numpy(ntype_idmap[dst_ntype][dst_nodes]).type(torch.int64))
        hg = dgl.heterograph(data_dict)
        if self_loop:
            for each in hg.ntypes:
                self_loop_name = (each, each[0] + each[0], each)
                data_dict[self_loop_name] = (
                    torch.arange(hg.number_of_nodes(each), dtype=torch.int64),
                    torch.arange(hg.number_of_nodes(each), dtype=torch.int64)
                )
            hg = dgl.heterograph(data_dict)
        all_label = np.full(hg.num_nodes(target_ntype), -1, dtype=int)
        for i, split in enumerate(['train', 'val', 'test']):
            node = np.array(labels[i])[:, 0]
            label = np.array(labels[i])[:, 1]
            all_label[node] = label
            hg.nodes[target_ntype].data['{}_mask'.format(split)] = \
                torch.from_numpy(idx2mask(node, hg.num_nodes(target_ntype))).type(torch.bool)
        hg.nodes[target_ntype].data['label'] = torch.from_numpy(all_label).type(torch.long)

        # node feature
        node_features = torch.from_numpy(node_features).type(torch.FloatTensor)
        for ntype in ntypes:
            idx = ntype_mask[ntype].nonzero()[0]
            hg.nodes[ntype].data['h'] = node_features[idx]
    inputs = hg.ndata['h']             
    node_dict = OrderedDict()
    for ntype in hg.ntypes:
        node_dict[ntype] = len(node_dict)
    edge_dict = OrderedDict()
    for etype in hg.canonical_etypes:
        edge_dict[etype] = len(edge_dict)
        hg.edges[etype].data['id'] = torch.ones(hg.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]
    labels = hg.nodes['movie'].data['label']
    #print(labels, len(labels))
    train_mask = hg.nodes[target_ntype].data['train_mask']
    val_mask = hg.nodes[target_ntype].data['val_mask']
    test_mask = hg.nodes[target_ntype].data['test_mask']
    #print(len(train_mask))
    train_idx = torch.nonzero(train_mask, as_tuple=True)[0]
    val_idx = torch.nonzero(val_mask, as_tuple=True)[0]
    test_idx = torch.nonzero(test_mask, as_tuple=True)[0]
    #print(train_ids,val_ids,test_ids)
    feat_sizes = OrderedDict()
    for feat_name in inputs:
        feat_sizes[feat_name] = inputs[feat_name].size(-1)
    # save_dgl_dataset(hg, dgl_graph_file, edge_dict, feat_sizes, info_path, labels, node_dict, test_idx,
    #                      train_idx, val_idx)
    print(hg)
    return hg, inputs, labels, node_dict, edge_dict, train_idx, val_idx, test_idx, "movie", feat_sizes
    


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()


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
    #print(G, edge_dict, feat_sizes, inputs, labels_tensor, node_dict, test_node, train_node, valid_node)
    return G, edge_dict, feat_sizes, inputs, labels_tensor, node_dict, test_node, train_node, valid_node


def get_node_count(dataset="ACM", self_loop=False):
    G, inputs, labels_tensor, node_dict, edge_dict, train_node, valid_node, test_node, \
    predict_category, feat_sizes = load_data(dataset, self_loop)
    for i, predict_key in enumerate([predict_category]):
        predict_key_index = G.ntypes.index(predict_key)
        assert predict_key_index >= 0, f"Wrong predict_key:{predict_key}"
        if predict_key_index > i:
            tmp = G.ntypes[i]
            G.ntypes[i] = predict_key
            G.ntypes[predict_key_index] = tmp
    g, node_count, edge_count = dgl.to_homogeneous(G, return_count=True)
    return node_count


def get_rel_name(filename: str):
    start_index = filename.rindex("/")
    filename = filename[start_index + 1:-4]
    src_rel, dst_rel = filename.split("_")
    return (src_rel, src_rel[:2] + dst_rel[:2], dst_rel)


def get_reverse_rel(rel_name):
    src_rel, dst_rel = rel_name[0], rel_name[2]
    return (dst_rel, dst_rel[:2] + src_rel[:2], src_rel)


def build_edge_index(filename, rating_score=0):
    data = pd.read_csv(filename, sep="[,\t]", header=None)
    if data.shape[-1] > 2:
        pos_data = data[data[2] > rating_score]
        neg_data = data[data[2] <= rating_score]
        return (pos_data[0].values, pos_data[1].values), (neg_data[0].values, neg_data[1].values)
    else:
        return (data[0].values, data[1].values), ([], [])


def load_link_prediction_data(dataset="Yelp", self_loop=True, path=base_path, force=False):
    dataset = dataset.lower()
    assert dataset in ["yelp", "douban", "amazon","movielens","doubanb"]
    dataset = dataset.capitalize()
    predict_edge = predict_links[dataset][-1]

    dgl_graph_file = f"{path}/{dataset}/data.bin"
    info_path = f"{path}/{dataset}/info.pk"
    link_path = f"{path}/{dataset}/link.pk"  # links removed from origin graph and negative links
    modified_graph_file = f"{path}/{dataset}/modified_graph.bin"
    if not force and os.path.exists(dgl_graph_file):
        G, edge_dict, feat_sizes, inputs, labels_tensor, node_dict, test_node, train_node, valid_node = load_dgl_dataset(
            dgl_graph_file, info_path)
    else:
        filelist = []
        # get filenames
        for file in os.listdir(f"{path}/{dataset}"):
            if file.endswith(".dat"):
                filelist.append(f"{path}/{dataset}/{file}")

        rating_threshold = rating_threshold_dict[dataset]
        rel_edges_dict = {}
        neg_edge_list = []
        for i, filename in enumerate(filelist):
            rel_name = get_rel_name(filename)
            threshold = 0
            if rel_name == predict_edge:
                threshold = rating_threshold
            edge_index, neg_edge = build_edge_index(filename, rating_score=threshold)

            neg_edge_list.append(neg_edge)
            rel_edges_dict[rel_name] = edge_index
            reverse_relation = get_reverse_rel(rel_name)
            rel_edges_dict[reverse_relation] = (edge_index[1], edge_index[0])

        G = dgl.heterograph(rel_edges_dict)
        if self_loop:
            for each in G.ntypes:
                self_loop_name = (each, each[0] + each[0], each)
                if self_loop_name in rel_edges_dict:
                    merge_self_loop(G, each, rel_edges_dict, self_loop_name)
                else:
                    rel_edges_dict[self_loop_name] = (
                        torch.arange(G.number_of_nodes(each)),
                        torch.arange(G.number_of_nodes(each))
                    )
            G = dgl.heterograph(rel_edges_dict)
        # assign features
        for ntype in G.ntypes:
            feature = torch.eye(G.number_of_nodes(ntype))
            G.nodes[ntype].data["h"] = feature

        inputs = G.ndata["h"]
        node_dict = {}
        for ntype in G.ntypes:
            node_dict[ntype] = len(node_dict)
        edge_dict = {}
        for etype in G.canonical_etypes:
            edge_dict[etype] = len(edge_dict)
            # hg.edges[etype].data['id'] = torch.ones(hg.number_of_edges(etype), dtype=torch.long) * edge_dict[etype]

        feat_sizes = {}
        for k, v in inputs.items():
            feat_sizes[k] = v.size(-1)
        save_dgl_dataset(G, dgl_graph_file, edge_dict, feat_sizes, info_path, None, node_dict, None,
                         None, None)
    if os.path.exists(link_path) and os.path.exists(modified_graph_file) and force:
        G, _ = load_graphs(modified_graph_file)
        info = load_info(link_path)
        train_pos, train_neg, val_pos, val_neg, test_pos, test_neg = \
            info["train_pos"], info["train_neg"], info["val_pos"], info["val_neg"], info["test_pos"], info["test_neg"]
        print("exist")
    else:
        G, train_pos, train_neg, val_pos, val_neg, test_pos, test_neg = negative_sample(G, predict_edge, link_path,
                                                                                        modified_graph_file)
    # print("train_pos:",train_pos,"size:",train_pos[0].size(0),train_pos[1].size(0),"max:",train_pos[0].max(),train_pos[1].max())
    # print("train_neg:",train_neg,"size:",len(train_pos[0]),len(train_pos[1]),"max:",train_neg[0].max(),train_neg[1].max())
    # print("val_pos:",val_pos)
    # print("val_neg:",val_neg)
    # print("test_pos:",test_pos)
    # print("test_neg:",test_neg)
    return G, inputs, None, node_dict, edge_dict, None, None, None, \
           predict_links[dataset], feat_sizes, \
           torch.stack(train_pos), \
           torch.stack(val_pos), \
           torch.stack(test_pos), \
           torch.LongTensor(train_neg), \
           torch.LongTensor(val_neg), \
           torch.LongTensor(test_neg)


def merge_self_loop(G, each, rel_edges_dict, self_loop_name):
    self_loop_index = torch.arange(G.number_of_nodes(each))
    new_src = torch.cat([rel_edges_dict[self_loop_name][0], self_loop_index], dim=-1)
    new_dst = torch.cat([rel_edges_dict[self_loop_name][1], self_loop_index], dim=-1)
    rel_edges_dict[self_loop_name] = (new_src, new_dst)


def negative_sample(G, predict_edge, link_path, modified_graph_file):
    # print(G)
    np.random.seed(1)
    g = G[predict_edge]
    u, v = g.edges()
    eids = np.arange(g.number_of_edges())
    eids = np.random.permutation(eids)[:int(0.5 * len(eids))]  # leave 50% edges to train the model
    #eids = eids[:int(0.5 * len(eids))]
    train_size = int(len(eids) * 0.6)
    val_size = int(len(eids) * 0.2)
    test_size = int(len(eids) * 0.2)
    train_pos_u, train_pos_v = u[eids[:train_size]], v[eids[:train_size]]
    val_pos_u, val_pos_v = u[eids[train_size:train_size + val_size]], v[eids[train_size:train_size + val_size]]
    test_pos_u, test_pos_v = u[eids[-test_size:]], v[eids[-test_size:]]
    # Find all negative edges and split them for training and testing
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))
    adj_neg = 1 - adj.todense()
    neg_u, neg_v = np.where(adj_neg != 0)
    neg_eids = np.random.choice(len(neg_u), len(eids))
    train_neg_u, train_neg_v = neg_u[neg_eids[:train_size]], neg_v[neg_eids[:train_size]]
    val_neg_u, val_neg_v = neg_u[neg_eids[train_size:train_size + val_size]], neg_v[
        neg_eids[train_size:train_size + val_size]]
    test_neg_u, test_neg_v = neg_u[neg_eids[-test_size:]], neg_v[neg_eids[-test_size:]]
    G = dgl.remove_edges(G, eids, etype=predict_edge)
    reverse_relation = get_reverse_rel(predict_edge)
    G = dgl.remove_edges(G, eids, etype=reverse_relation)
    save_graphs(modified_graph_file, G)
    info = {}
    info["train_pos"] = (train_pos_u, train_pos_v)
    info["train_neg"] = (train_neg_u, train_neg_v)
    info["val_pos"] = (val_pos_u, val_pos_v)
    info["val_neg"] = (val_neg_u, val_neg_v)
    info["test_pos"] = (test_pos_u, test_pos_v)
    info["test_neg"] = (test_neg_u, test_neg_v)
    save_info(link_path, info)
    return G, info["train_pos"], info["train_neg"], info["val_pos"], info["val_neg"], info["test_pos"], info["test_neg"]


def load_predict_links(dataset):
    dataset = dataset.capitalize()
    return predict_links[dataset]


if __name__ == '__main__':
    #a = load_link_prediction_data("yelp", force=False)
    a = load_data("imdb")
    # print("_______________________")
    # a = load_data("acm")