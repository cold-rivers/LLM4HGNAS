import os
import pickle

import dgl
import numpy as np
import torch
from scipy.sparse import csr_matrix

from data.data_util import load_predict_links, load_link_prediction_data, load_data


def pad_features(G, feat_sizes, inputs):
    max_size = max([v for k, v in feat_sizes.items()])
    for k, v in feat_sizes.items():
        padded_feat = torch.zeros(inputs[k].size(0), max_size)
        padded_feat[:, :v] = inputs[k][:, :v]
        inputs[k] = padded_feat
        G.nodes[k].data["h"] = padded_feat


def to_hne_format(dataset, G, train_pos, val_pos, test_pos, train_neg, val_neg, test_neg,
                  out_dir="test/"):  # link prediction tasks
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    predict_info = load_predict_links(dataset)
    ntype_list = G.ntypes
    etype_list = G.canonical_etypes
    g, node_count, edge_count = dgl.to_homogeneous(G, return_count=True)
    write_node(g, out_dir)
    write_edge(g, out_dir)
    write_label(g, out_dir)

    src_offset = sum(node_count[:ntype_list.index(predict_info[0])])
    dst_offset = sum(node_count[:ntype_list.index(predict_info[1])])
    etype_index = etype_list.index((predict_info[-1]))

    all_pos = (
        torch.cat([train_pos[0], val_pos[0], test_pos[0]], dim=-1),
        torch.cat([train_pos[1], val_pos[1], test_pos[1]], dim=-1),
    )
    pos_labels = torch.ones_like(all_pos[0], dtype=torch.int)
    all_neg = (
        torch.cat([train_neg[0], val_neg[0], test_neg[0]], dim=-1),
        torch.cat([train_neg[1], val_neg[1], test_neg[1]], dim=-1),
    )
    neg_labels = torch.zeros_like(all_neg[0], dtype=torch.int)
    all = (
        torch.cat([all_pos[0], all_neg[0]], dim=-1),
        torch.cat([all_pos[1], all_neg[1]], dim=-1),
    )
    all = (
        all[0] + src_offset,
        all[1] + dst_offset,
    )
    edge_label = torch.cat([pos_labels, neg_labels], dim=-1)
    write_link_test(all, edge_label, out_dir)
    write_info(G, out_dir)
    write_path(dataset, G, out_dir)


def write_node(g, out_dir="test/"):
    types = g.ndata["_TYPE"].numpy()
    ids = np.arange(len(types))
    with open(f"{out_dir}/node.dat", "w") as f:
        for id, type in zip(ids, types):
            f.write(f"{id}\tnode_name\t{type}\n")


def write_edge(g, out_dir="test/"):
    etypes = g.edata["_TYPE"].numpy()
    src, dst = g.edges()
    with open(f"{out_dir}/link.dat", "w") as f:
        for info in zip(src, dst, etypes):
            msg = "\t".join([str(int(each)) for each in info])
            f.write(f"{msg}\n")


def write_label(g, out_dir="test/"):
    with open(f"{out_dir}/label.dat", "w") as f:
        f.write(f" ")


def write_link_test(all_test_links, edge_labels, out_dir="test/"):
    with open(f"{out_dir}/link.dat.test", "w") as f:
        for info in zip(all_test_links[0], all_test_links[1], edge_labels):
            msg = "\t".join([str(int(each)) for each in info])
            f.write(f"{msg}\n")


def write_info(G, out_dir):
    ntype_list = G.ntypes
    with open(f"{out_dir}/info.dat", "w") as f:
        f.write("LINK	START	END	MEANING\n")
        for i, etype in enumerate(G.canonical_etypes):
            msg = f"{i}\t{ntype_list.index(etype[0])}\t{ntype_list.index(etype[2])} \t{etype} \n"
            f.write(msg)
        f.write("\n")


def write_path(dataset, G, out_dir):
    from baselines.meta_paths import meta_paths_dict
    edge_list = G.canonical_etypes
    ntypes = G.ntypes
    meta_path_list = meta_paths_dict[dataset]
    with open(f"{out_dir}/path.dat", "w") as f:
        for metapath in meta_path_list:
            path = [ntypes.index(metapath[0][0])]
            for etype in metapath:
                path.append(ntypes.index(etype[-1]))
            f.write("\t".join([str(each) for each in path]) + "\n")


def to_homo_full(G, feat_sizes, inputs, labels_tensor, train_idx, val_idx, test_idx,
                 predict_key=("paper")):  # for link-prediction
    ntype_list = G.ntypes
    pad_features(G, feat_sizes, inputs)
    g, node_count, edge_count = dgl.to_homogeneous(G, ndata="h", return_count=True)
    inputs = g.ndata["h"]
    offset = sum(node_count[:ntype_list.index(predict_key)])
    return g, inputs, labels_tensor, train_idx + offset, val_idx + offset, test_idx + offset, \
           labels_tensor[train_idx], labels_tensor[val_idx], labels_tensor[test_idx]


def to_gtn_format(dataset, out_dir="test/"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    G, inputs, labels, node_dict, edge_dict, \
    train_idx, val_idx, test_idx, category, feat_sizes, \
    train_pos, val_pos, test_pos, \
    train_neg, val_neg, test_neg = load_link_prediction_data(dataset, self_loop=False)
    ntype_list = G.ntypes

    g, node_count, edge_count = dgl.to_homogeneous(G, return_count=True)
    edge_list = write_gtn_edges(G, g)

    # write edges
    with open(f"{out_dir}/edges.pkl", "wb") as f:
        pickle.dump(edge_list, f)

    src_offset = sum(node_count[:ntype_list.index(category[0])])
    dst_offset = sum(node_count[:ntype_list.index(category[1])])

    num_nodes = G.num_nodes()
    validate_edges = []
    for edge_index in [train_pos, val_pos, test_pos, train_neg, val_neg, test_neg]:
        sparse_matrix = csr_matrix(
            (
                torch.ones_like(edge_index[0], dtype=torch.int).numpy(),
                (edge_index[0] + src_offset, edge_index[1] + dst_offset)
            ),
            shape=(num_nodes, num_nodes)
        )
        validate_edges.append(sparse_matrix)
    # write edges
    with open(f"{out_dir}/predict_edge.pkl", "wb") as f:
        pickle.dump(validate_edges, f)


def write_gtn_edges(G, g):
    num_nodes = G.num_nodes()
    edge_list = []
    num_etypes = g.edata["_TYPE"].max().item() + 1
    for i in range(num_etypes):
        index = g.edata["_TYPE"] == i
        edge_index = g.edges()
        sparse_matrix = csr_matrix(
            (
                torch.ones_like(edge_index[0][index], dtype=torch.int).numpy(),
                (edge_index[0][index], edge_index[1][index])
            ),
            shape=(num_nodes, num_nodes)
        )
        edge_list.append(sparse_matrix)
    return edge_list


def load_homo_lp(dataset="Amazon", self_loop=False):
    dataset = dataset.capitalize()
    G, inputs, labels, node_dict, edge_dict, \
    train_idx, val_idx, test_idx, category, feat_sizes, \
    train_pos, val_pos, test_pos, \
    train_neg, val_neg, test_neg = load_link_prediction_data(dataset, self_loop=False)
    ntype_list = G.ntypes

    g, node_count, edge_count = dgl.to_homogeneous(G, return_count=True)
    g = dgl.add_self_loop(g)

    src_index, dst_index = ntype_list.index(category[0]), ntype_list.index(category[1])
    src_offset, dst_offset = sum(node_count[:src_index]), sum(node_count[:dst_index])
    src_node_count, dst_node_count = node_count[src_index], node_count[dst_index]

    return g, inputs, labels, node_dict, edge_dict, \
           train_idx, val_idx, test_idx, category, feat_sizes, \
           train_pos, val_pos, test_pos, \
           train_neg, val_neg, test_neg, \
           (src_offset, dst_offset), (src_node_count, dst_node_count)  # additional outputs


def to_gtn_format_nc(dataset, out_dir="test/"):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    G, inputs, labels, node_dict, edge_dict, \
    train_idx, val_idx, test_idx, category, feat_sizes = load_data(dataset, self_loop=False)
    g, inputs, labels_tensor, train_idx, val_idx, test_idx, \
    train_label, val_label, test_label = to_homo_full(G, feat_sizes, inputs, labels, train_idx, val_idx, test_idx,
                                                      category)

    # write features
    with open(f"{out_dir}/node_features.pkl", "wb") as f:
        pickle.dump(inputs.numpy(), f)

    num_nodes = G.num_nodes()
    edge_list = []
    num_etypes = g.edata["_TYPE"].max().item() + 1
    for i in range(num_etypes):
        index = g.edata["_TYPE"] == i
        edge_index = g.edges()
        sparse_matrix = csr_matrix(
            (
                torch.ones_like(edge_index[0][index], dtype=torch.int).numpy(),
                (edge_index[0][index], edge_index[1][index])
            ),
            shape=(num_nodes, num_nodes)
        )
        edge_list.append(sparse_matrix)

    # write edges
    with open(f"{out_dir}/edges.pkl", "wb") as f:
        pickle.dump(edge_list, f)

    labels = []

    for idx, label in zip([train_idx, val_idx, test_idx], [train_label, val_label, test_label]):
        labels.append(torch.stack([idx, label], dim=0).t().numpy())

    # write edges
    with open(f"{out_dir}/labels.pkl", "wb") as f:
        pickle.dump(labels, f)


if __name__ == '__main__':
    for dataset in ["Amazon", "Yelp"]:
        a = load_link_prediction_data(dataset=dataset)
        out_path = f"{dataset}_gtn/"
        to_gtn_format(dataset.lower(), out_dir=out_path)

    dataset_list = ["dblp", "acm"]
    for dataset in dataset_list:
        to_gtn_format_nc(dataset, f"{dataset}_gtn/")
