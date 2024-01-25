#!/usr/bin/env python
# coding: utf-8


from sklearn.metrics import f1_score, accuracy_score

from data.data_util import *
from data.data_util import get_logger
from hgnn.layers.mlp import *
from hgnn.layers.relation_embed import RelGraphEmbed, AdaptiveRelGraphEmbed
from hgnn.meta_gnn import RelationOnlyNet
from hgnn.meta_gnn import SkipConnectionNet



class HGNNManager(object):
    def __init__(self, args, device=None):

        self.args = args
        self.device = torch.device('cuda' if args.cuda else 'cpu') if device is None else device
        # load data
        self.load_dataset(args)

        # load hyper_parameters
        self.dropout = args.dropout
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs

        self.loss_fn = torch.nn.functional.nll_loss
        self.evaluator = self.build_evaluator()
        self.logger = self.build_logger()

    def build_logger(self):
        if hasattr(self.args, "logger"):
            return self.args.logger
        else:
            return get_logger()

    def update_args(self, args):
        self.args = args

    # train from scratch
    def evaluate(self, actions=None, format="origin", device=None):
        _, val_score, test_score = self.train(actions, format, device, return_test_score=True)
        return val_score, test_score

    def record_action_info(self, origin_action, reward, val_acc, filename=None):
        if filename is None:
            filename = self.log_file_name
        with open(filename, "a") as file:
            file.write(str(origin_action))

            file.write(";")
            file.write(str(reward))

            file.write(";")
            file.write(str(val_acc))
            file.write("\n")

    @property
    def log_file_name(self):
        if hasattr(self.args, "gnn_log_file"):
            return self.args.dataset + "_" + self.args.search_mode + "_" + self.args.gnn_log_file
        else:
            return self.args.dataset + "_" + self.args.search_mode + "_history.log"

    def reset(self):
        self.data_to_devices(torch.device("cpu"))
        self.load_dataset(self.args)

    def build_evaluator(self):
        if hasattr(self.args, "metrics"):
            if self.args.metrics == "macro_f1":
                def evaluate(pred, label):
                    return f1_score(label, pred, average="macro")

                return evaluate
            elif self.args.metrics == "micro_f1":
                def evaluate(pred, label):
                    return f1_score(label, pred, average="micro")

                return evaluate
            elif self.args.metrics == "acc":
                return accuracy_score
            else:
                raise RuntimeError(f"No such evaluator:{self.args.metrics}")

        else:
            return accuracy_score

    def load_dataset(self, args):
        supervised = args.supervised if hasattr(args, "supervised") else False
        train_size = args.train_size if hasattr(args, "train_size") else 400
        shuffle_data = args.shuffle_data if hasattr(args, "shuffle_data") else False
        self.data = list(load_data(args.dataset, supervised, shuffle_data, train_size))
        # self.data = list(load_data_new(dataset=args.dataset, self_loop=args.self_loop))
        self.node_dict = self.data[3]
        self.edge_dict = self.data[4]
        self.num_of_nodes = {ntype: self.data[0].num_nodes(ntype) for ntype in self.data[0].ntypes}
        self.args.in_feats = self.in_feats = self.args.n_inp
        self.args.num_class = self.n_classes = self.data[2].max().item() + 1

    def build_embed_layer(self):
        if hasattr(self.args, "use_feat") and self.args.use_feat:
            if hasattr(self.args, "use_embed") and self.args.use_embed:
                self.embed_layer = AdaptiveRelGraphEmbed(self.node_dict, self.data[1], self.args.n_hid,
                                                         dropout=self.args.dropout, activation=F.gelu)
            else:
                self.embed_layer = HeteroAdaptMLP(self.data[9], self.args.n_hid, self.args.n_hid,
                                                  self.node_dict, n_layer=1, act=F.gelu)
        else:
            batch_train = hasattr(self.args, "batch_train") and self.args.batch_train
            self.embed_layer = RelGraphEmbed(self.node_dict, self.num_of_nodes, self.args.n_hid,
                                             batch_train=batch_train)

    def build_gnn(self, actions):
        raise NotImplementedError

    def data_to_devices(self, device):
        for i, each in enumerate(self.data):
            if isinstance(each, torch.Tensor) or isinstance(each, dgl.DGLGraph):
                self.data[i] = each.to(device)
            if isinstance(each, dict):
                for ntype in each:
                    if isinstance(each[ntype], torch.Tensor):
                        each[ntype] = each[ntype].to(device)

    def train(self, gnn_desc, format="cell_based", device=None, return_test_score=False):
        device = self.device if device is None else device
        model_actions = self.process_action(gnn_desc, format)
        args = self.args
        count = 0
        flag = False
        while not flag and count < 10:
            try:
                count += 1
                torch.cuda.empty_cache()
                self.data_to_devices(device)
                model = self.build_gnn(model_actions).to(device)
                flag = True
            except Exception as e:
                logging.log(logging.WARNING, gnn_desc)
                logging.log(logging.WARNING, str(e))
                if hasattr(args, "debug") and args.debug:
                    raise e

        if not flag:  # fail to build model
            if return_test_score:
                return 0, 0, 0
            else:
                return 0, 0
        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.weight_decay)
        if hasattr(args, "cycle_lr") and args.cycle_lr:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.epochs, max_lr=args.lr)
        else:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
        loss_fn = F.cross_entropy
        try:
            show_info = hasattr(args, "show_info") and args.show_info
            res = self.run_model(model, optimizer, scheduler, loss_fn,
                                 self.args.epochs,
                                 self.data,
                                 return_test=return_test_score,
                                 gnn_desc=gnn_desc,
                                 show_info=show_info)
        # except KeyError as e:
        #     res = (0, 0, 0) if return_test_score else (0, 0)
        #     logging.log(logging.DEBUG, gnn_desc)
        #     logging.log(logging.DEBUG, "wrong structure!")
        except Exception as e:
            #  TODO  a right way to process OOM Error
            res = (0, 0, 0) if return_test_score else (0, 0)
            logging.log(logging.WARNING, gnn_desc)
            logging.log(logging.WARNING, str(e))
            if hasattr(args, "debug") and args.debug:
                raise e
            torch.cuda.empty_cache()
            if "CUDA" in str(e) or "cuda" in str(e) or "cuDNN" in str(e):
                res = (0, 0, 0) if return_test_score else (0, 0)
            elif self.data[-2] in str(e):
                logging.log(logging.WARNING, "wrong structure!")
            else:
                logging.log(logging.WARNING, "other errors!!!")
                if hasattr(args, "debug") and args.debug:
                    raise e
        finally:
            del model
            self.reset()
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        if return_test_score:
            model, val_score, test_score = res
            return val_score, val_score, test_score
        else:
            model, val_score = res
            return val_score, val_score

    def process_action(self, actions, format="cell_based"):
        pass
        # self.current_action = actions
        # model_actions = actions["action"]
        # if self.args.predict_hyper:
        #     manager.args.dropout = actions["hyper_param"][1]
        #     manager.args.lr = actions["hyper_param"][0]
        #     manager.args.n_hid = actions["hyper_param"][3]
        #     manager.args.weight_decay = actions["hyper_param"][2]
        # return model_actions

    def run_model(self,
                  model,
                  optimizer,
                  scheduler,
                  loss_fn,
                  epochs,
                  data,
                  grad_clip=1,
                  return_test=False,
                  show_info=False,
                  gnn_desc=None,
                  evaluate=None):
        best_val_score = 0
        best_test_score = 0
        train_score = 0
        train_step = 0
        G, inputs, labels, node_dict, edge_dict, train_idx, val_idx, test_idx, out_key, _ = data
        # inputs = {}
        # for ntype in G.ntypes:
        #     inputs[ntype] = G.nodes[ntype].data['inp']
        for epoch in np.arange(epochs) + 1:
            model.train()
            logits = self.forward_model(G, inputs, model, out_key)
            # The loss is computed only for labeled nodes.
            loss = loss_fn(logits[train_idx], labels[train_idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_step += 1
            best_test_score, best_val_score, train_score = self.eval_process(G, best_test_score, best_val_score,
                                                                             train_score, epoch, inputs, labels, loss,
                                                                             model, optimizer, out_key, show_info,
                                                                             test_idx, train_idx, val_idx)
        self.show_global_info(best_test_score, best_val_score, gnn_desc, train_idx, train_score)
        if return_test:
            return model, best_val_score, best_test_score
        else:
            return model, best_val_score

    def forward_model(self, G, inputs, model, out_key):
        logits = model(G, inputs, out_key)
        return logits

    def eval_process(self, G, best_test_score, best_val_score, train_score, epoch, inputs, labels, loss, model,
                     optimizer, out_key, show_info, test_idx, train_idx, val_idx):
        if epoch % 5 == 0:
            model.eval()
            logits = self.forward_model(G, inputs, model, out_key)
            pred = logits.argmax(1)
            pred = pred.cpu()
            tmp_labels = labels
            train_score = self.evaluator(pred[train_idx], tmp_labels[train_idx])
            val_score = self.evaluator(pred[val_idx], tmp_labels[val_idx])
            test_score = self.evaluator(pred[test_idx], tmp_labels[test_idx])
            if best_val_score < val_score:
                best_val_score = val_score
                best_test_score = test_score
            if show_info:
                print(
                    'Epoch: %d LR: %.5f Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), '
                    'Test Acc %.4f (Best %.4f)' % (
                        epoch,
                        optimizer.param_groups[0]['lr'],
                        loss,
                        train_score,
                        val_score,
                        best_val_score,
                        test_score,
                        best_test_score,
                    ))
        return best_test_score, best_val_score, train_score

    def show_global_info(self, best_test_score, best_val_score, gnn_desc, train_idx, train_score):
        info = f"train action:{gnn_desc} | Number of train datas:{train_idx.size(0)}|" \
               f"train_score:{train_score:.4f},val_score:{best_val_score:.4f},test_score:{best_test_score:.4f}"
        if hasattr(self.args, "logger"):
            self.args.logger.info(info)
        else:
            print(info)


class MetaOptManager(HGNNManager):

    def build_gnn(self, actions):
        self.build_embed_layer()
        self_loop = self.args.self_loop if hasattr(self.args, "self_loop") else False
        model = RelationOnlyNet(actions,
                                node_dict=self.node_dict,
                                edge_dict=self.edge_dict,  # {str:int} map edge type to their index
                                n_inp=self.in_feats,
                                n_hid=self.args.n_hid,
                                n_out=self.n_classes,
                                dropout=self.dropout,
                                n_layers=2,
                                task_specific_encoder=self.embed_layer,
                                n_heads=4,
                                self_loop=self_loop)
        return model

    def process_action(self, actions, format="meta_opt_3"):
        self.current_action = actions
        model_actions = actions
        return model_actions

    def train(self, gnn_desc=None, format="meta_opt_3", device=None, return_test_score=False):
        return super(MetaOptManager, self).train(gnn_desc,
                                                 format=format,
                                                 device=device,
                                                 return_test_score=return_test_score)

    def evaluate(self, actions=None, format="meta_opt_3", device=None):
        return super(MetaOptManager, self).evaluate(actions, format, device)


class AggrManager(MetaOptManager):
    def reset(self):
        self.data_to_devices(torch.device("cpu"))
        self.load_dataset(self.args)

    def build_evaluator(self):
        if hasattr(self.args, "metrics"):
            if self.args.metrics == "macro_f1":
                def evaluate(pred, label):
                    return f1_score(label, pred, average="macro")

                return evaluate
            elif self.args.metrics == "micro_f1":
                def evaluate(pred, label):
                    return f1_score(label, pred, average="micro")

                return evaluate
            elif self.args.metrics == "acc":
                return accuracy_score
            else:
                raise RuntimeError(f"No such evaluator:{self.args.metrics}")

        else:
            return accuracy_score

    def load_dataset(self, args):
        # supervised = args.supervised if hasattr(args, "supervised") else False
        # train_size = args.train_size if hasattr(args, "train_size") else 400
        # shuffle_data = args.shuffle_data if hasattr(args, "shuffle_data") else False
        self.data = list(load_data(args.dataset, self_loop=args.self_loop))
        # self.data = list(load_data_new(dataset=args.dataset, self_loop=args.self_loop))
        self.num_of_nodes = {ntype: self.data[0].num_nodes(ntype) for ntype in self.data[0].ntypes}
        self.args.in_feats = self.in_feats = self.data[9]
        self.args.num_class = self.n_classes = self.data[2].max().item() + 1

        self.node_dict = self.data[3]
        self.edge_dict = self.data[4]
        self.predict_keys = self.data[8] if isinstance(self.data[8], list) else [self.data[8]]

    def build_embed_layer(self):
        if hasattr(self.args, "use_feat") and self.args.use_feat:
            if hasattr(self.args, "use_embed") and self.args.use_embed:
                self.embed_layer = AdaptiveRelGraphEmbed(self.node_dict, self.data[1], self.args.n_hid,
                                                         dropout=self.args.dropout, activation=F.gelu)
            else:
                self.embed_layer = HeteroAdaptMLP(self.data[9], self.args.n_hid, self.args.n_hid,
                                                  self.node_dict, n_layer=1, act=F.gelu)
        else:
            batch_train = hasattr(self.args, "batch_train") and self.args.batch_train
            self.embed_layer = RelGraphEmbed(self.node_dict, self.num_of_nodes, self.args.n_hid,
                                             batch_train=batch_train)

    def record_action_info(self, origin_action, reward, val_acc):
        return super(HGNNManager, self).record_action_info(self.current_action, reward, val_acc)

    def data_to_devices(self, device):
        for i, each in enumerate(self.data):
            if isinstance(each, torch.Tensor) or isinstance(each, dgl.DGLGraph):
                self.data[i] = each.to(device)
            if isinstance(each, dict):
                for ntype in each:
                    if isinstance(each[ntype], torch.Tensor):
                        each[ntype] = each[ntype].to(device)

    def train(self, gnn_desc, format="cell_based", device=None, return_test_score=False):
        device = self.device if device is None else device
        model_actions = self.process_action(gnn_desc, format)
        args = self.args
        count = 0
        flag = False
        while not flag and count < 10:
            try:
                count += 1
                torch.cuda.empty_cache()
                self.data_to_devices(device)
                model = self.build_gnn(model_actions).to(device)
                flag = True
            except Exception as e:
                logging.log(logging.WARNING, gnn_desc)
                logging.log(logging.WARNING, str(e))
                if hasattr(args, "debug") and args.debug:
                    raise e

        if not flag:  # fail to build model
            if return_test_score:
                return 0, 0, 0
            else:
                return 0, 0
        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.weight_decay)
        if hasattr(args, "cycle_lr") and args.cycle_lr:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.epochs, max_lr=args.lr)
        else:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
        loss_fn = F.cross_entropy
        try:
            show_info = hasattr(args, "show_info") and args.show_info
            res = self.run_model(model, optimizer, scheduler, loss_fn,
                                 self.args.epochs,
                                 self.data,
                                 return_test=return_test_score,
                                 gnn_desc=gnn_desc,
                                 show_info=show_info)
        # except KeyError as e:
        #     res = (0, 0, 0) if return_test_score else (0, 0)
        #     logging.log(logging.DEBUG, gnn_desc)
        #     logging.log(logging.DEBUG, "wrong structure!")
        except Exception as e:
            #  TODO  a right way to process OOM Error
            res = (0, 0, 0) if return_test_score else (0, 0)
            logging.log(logging.WARNING, gnn_desc)
            logging.log(logging.WARNING, str(e))
            if hasattr(args, "debug") and args.debug:
                raise e
            torch.cuda.empty_cache()
            if "CUDA" in str(e) or "cuda" in str(e) or "cuDNN" in str(e):
                res = (0, 0, 0) if return_test_score else (0, 0)
            elif self.data[-2] in str(e):
                logging.log(logging.WARNING, "wrong structure!")
            else:
                logging.log(logging.WARNING, "other errors!!!")
                if hasattr(args, "debug") and args.debug:
                    raise e
        finally:
            del model
            self.reset()
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        if return_test_score:
            model, val_score, test_score = res
            return val_score, val_score, test_score
        else:
            model, val_score = res
            return val_score, val_score

    def run_model(self,
                  model,
                  optimizer,
                  scheduler,
                  loss_fn,
                  epochs,
                  data,
                  grad_clip=1,
                  return_test=False,
                  show_info=False,
                  gnn_desc=None,
                  evaluate=None,
                  patience=20):
        best_val_score = 0
        best_test_score = 0
        train_score = 0
        train_step = 0
        G, inputs, labels, node_dict, edge_dict, train_idx, val_idx, test_idx, out_key, _ = data
        # inputs = {}
        # for ntype in G.ntypes:
        #     inputs[ntype] = G.nodes[ntype].data['inp']
        current_best = 0
        current_patience = 0
        for epoch in np.arange(epochs) + 1:
            model.train()
            #print("forward")
            logits = self.forward_model(G, inputs, model, out_key)
            # The loss is computed only for labeled nodes.
            loss = loss_fn(logits[train_idx], labels[train_idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            train_step += 1
            #print("eval_process")
            best_test_score, best_val_score, train_score = self.eval_process(G, best_test_score, best_val_score,
                                                                             train_score, epoch, inputs, labels, loss,
                                                                             model, optimizer, out_key, show_info,
                                                                             test_idx, train_idx, val_idx)
            if current_best < best_val_score:
                current_best = best_val_score
                current_patience = 0
            else:
                current_patience += 1
                if current_patience >= patience:
                    break
        self.show_global_info(best_test_score, best_val_score, gnn_desc, train_idx, train_score)
        if return_test:
            return model, best_val_score, best_test_score
        else:
            return model, best_val_score

    def forward_model(self, G, inputs, model, out_key):
        logits = model(G, inputs, out_key)
        return logits

    def eval_process(self, G, best_test_score, best_val_score, train_score, epoch, inputs, labels, loss, model,
                     optimizer, out_key, show_info, test_idx, train_idx, val_idx):
        if epoch % 5 == 0:
            model.eval()
            logits = self.forward_model(G, inputs, model, out_key)
            pred = logits.argmax(1)
            pred = pred
            tmp_labels = labels
            
            train_score = self.evaluator(pred[train_idx].cpu(), tmp_labels[train_idx].cpu())
            val_score = self.evaluator(pred[val_idx].cpu(), tmp_labels[val_idx].cpu())
            test_score = self.evaluator(pred[test_idx].cpu(), tmp_labels[test_idx].cpu())
            if best_val_score < val_score:
                best_val_score = val_score
                best_test_score = test_score
            if show_info:
                print(
                    'Epoch: %d LR: %.5f Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), '
                    'Test Acc %.4f (Best %.4f)' % (
                        epoch,
                        optimizer.param_groups[0]['lr'],
                        loss,
                        train_score,
                        val_score,
                        best_val_score,
                        test_score,
                        best_test_score,
                    ))
        return best_test_score, best_val_score, train_score

    def show_global_info(self, best_test_score, best_val_score, gnn_desc, train_idx, train_score):
        info = f"train action:{gnn_desc} | Number of train datas:{train_idx.size(0)}|" \
               f"train_score:{train_score:.4f},val_score:{best_val_score:.4f},test_score:{best_test_score:.4f}"
        if hasattr(self.args, "logger"):
            self.args.logger.info(info)
        else:
            print(info)

    def build_gnn(self, actions):
        self.build_embed_layer()
        model = RelationOnlyNet(actions,
                                node_dict=self.node_dict,
                                edge_dict=self.edge_dict,  # {str:int} map edge type to their index
                                n_inp=self.in_feats,
                                n_hid=self.args.n_hid,
                                n_out=self.n_classes,
                                dropout=self.dropout,
                                n_layers=self.args.n_hgnn_layers,
                                task_specific_encoder=self.embed_layer,
                                n_heads=4)
        return model


import logging

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from data.data_util import load_link_prediction_data, load_predict_links


class AggrManagerSK(AggrManager):

    def build_gnn(self, actions):
        self.build_embed_layer()
        self_loop = self.args.self_loop if hasattr(self.args, "self_loop") else False
        model = SkipConnectionNet(actions,
                                  node_dict=self.node_dict,
                                  edge_dict=self.edge_dict,  # {str:int} map edge type to their index
                                  n_inp=self.in_feats,
                                  n_hid=self.args.n_hid,
                                  n_out=self.n_classes,
                                  dropout=self.dropout,
                                  n_layers=2,
                                  task_specific_encoder=self.embed_layer,
                                  n_heads=4,
                                  self_loop=self_loop)
        return model


class HGNNLinkPredictorManager(AggrManagerSK):

    def build_evaluator(self):
        if hasattr(self.args, "metrics") and self.args.metrics == "auc":
            return roc_auc_score
        else:
            return super(HGNNLinkPredictorManager, self).build_evaluator()

    def load_dataset(self, args):
        self.data = list(load_link_prediction_data(dataset=args.dataset))
        # self.data = list(load_link_prediction_data_new(dataset=args.dataset))
        self.node_dict = self.data[3]
        self.edge_dict = self.data[4]
        self.in_feats = self.data[9]
        self.num_of_nodes = {ntype: self.data[0].num_nodes(ntype) for ntype in self.data[0].ntypes}
        self.pre_src_type, self.pre_dst_type, self.pre_link = load_predict_links(args.dataset)
        self.n_classes = args.n_hid

    # def data_to_devices(self, device):
    #     super(HGNNLinkPredictorManager, self).data_to_devices(device)
    #     self.pre_src_type = self.pre_src_type.to(device)
    #     self.train_idx = self.train_idx.to(device)
    #     self.val_idx = self.val_idx.to(device)
    #     self.test_idx = self.test_idx.to(device)

    def train(self, gnn_desc, format="cell_based", device=None, return_test_score=False):
        device = self.device if device is None else device
        model_actions = self.process_action(gnn_desc, format)
        args = self.args

        self.data_to_devices(device)
        model = self.build_gnn(model_actions).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.weight_decay)
        if hasattr(args, "cycle_lr") and args.cycle_lr:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.epochs, max_lr=args.lr)
        else:
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
        loss_fn = F.cross_entropy
        try:
            show_info = hasattr(args, "show_info") and args.show_info
            res = self.run_model(model, optimizer, scheduler, loss_fn,
                                 self.args.epochs,
                                 self.data,
                                 return_test=return_test_score,
                                 gnn_desc=gnn_desc,
                                 show_info=show_info)
        except Exception as e:
            #  TODO  a right way to process OOM Error
            res = (0, 0, 0) if return_test_score else (0, 0)
            if self.args.debug:
                raise e
            else:
                print(e)
        torch.cuda.empty_cache()
        if return_test_score:
            model, val_score, test_score = res
            return val_score, val_score, test_score
        else:
            model, val_score = res
            return val_score, val_score

    def calc_score(self, s, o):
        # simple distance
        score = torch.sum(s * o, dim=1)
        return score

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2))

    def get_loss(self, src_embed, dst_embed, edge_index, labels, mask):
        score = self.calc_score(src_embed[edge_index[0][mask]], dst_embed[edge_index[1][mask]])
        predict_loss = F.binary_cross_entropy_with_logits(score, labels[mask])

        return predict_loss + \
               self.args.reg_param * self.regularization_loss(src_embed[edge_index[0][mask]]) + \
               self.args.reg_param * self.regularization_loss(dst_embed[edge_index[1][mask]])

    def forward_model(self, G, inputs, model, out_key):
        logits = model(G, inputs, out_key)
        return logits

    def get_labels(self, src_embed, dst_embed, edge_index):
        score = self.calc_score(src_embed[edge_index[0]], dst_embed[edge_index[1]])
        labels = torch.sigmoid(score)
        return labels

    def eval_process(self, G, best_test_score, best_val_score, train_score, epoch, inputs, labels, loss, model,
                     optimizer, out_key, show_info, test_idx, train_idx, val_idx):
        if epoch % 10 == 0:
            model.eval()
            logits = self.forward_model(G, inputs, model, None)

            pred_labels = self.get_labels(logits[self.pre_src_type].cpu(), logits[self.pre_dst_type].cpu(),
                                          (self.test_links[0].cpu(), self.test_links[1].cpu()))
            pred_labels = pred_labels.cpu().detach()
            tmp_labels = labels.cpu()
            train_score = self.evaluator(tmp_labels[train_idx], pred_labels[train_idx], )
            val_score = self.evaluator(tmp_labels[val_idx], pred_labels[val_idx])
            test_score = self.evaluator(tmp_labels[test_idx], pred_labels[test_idx])
            if best_val_score < val_score:
                best_val_score = val_score
                best_test_score = test_score
            if show_info:
                print(
                    'Epoch: %d LR: %.5f Loss %.4f, Train Score %.4f, Val Score %.4f (Best %.4f), '
                    'Test Score %.4f (Best %.4f)' % (
                        epoch,
                        optimizer.param_groups[0]['lr'],
                        loss,
                        train_score,
                        val_score,
                        best_val_score,
                        test_score,
                        best_test_score,
                    ))
        return best_test_score, best_val_score, train_score

    def show_global_info(self, best_test_score, best_val_score, gnn_desc, train_idx, train_score):
        info = f"train action:{gnn_desc} | Number of train datas:{train_idx.size(0)}|" \
               f"train_score:{train_score:.4f},val_score:{best_val_score:.4f},test_score:{best_test_score:.4f}"
        if hasattr(self.args, "logger"):
            self.args.logger.info(info)
        else:
            print(info)


class MetaOptLinkPredictorManager(HGNNLinkPredictorManager):
    def process_action(self, actions, format="cell_based"):
        return actions

    def reset(self):
        self.data_to_devices(torch.device("cpu"))
        # self.data[0].clear()
        self.load_dataset(self.args)

    def load_dataset(self, args):
        supervised = args.supervised if hasattr(args, "supervised") else False
        assert not supervised, "Link prediction is unsupervised task"
        self.data = list(load_link_prediction_data(dataset=args.dataset, self_loop=args.self_loop))
        self.num_of_nodes = {ntype: self.data[0].num_nodes(ntype) for ntype in self.data[0].ntypes}
        self.node_dict = self.data[3]
        self.edge_dict = self.data[4]
        self.pre_src_type, self.pre_dst_type, self.pre_link = load_predict_links(args.dataset)
        self.n_classes = args.n_hid
        self.args.in_feats = self.in_feats = self.args.n_inp

    def data_to_devices(self, device):
        super(HGNNLinkPredictorManager, self).data_to_devices(device)

    def run_model(self,
                  model,
                  optimizer,
                  scheduler,
                  loss_fn,
                  epochs,
                  data,
                  grad_clip=1,
                  return_test=False,
                  show_info=False,
                  gnn_desc=None,
                  evaluate=None,
                  patience=20,
                  ):
        best_val_score = 0
        best_test_score = 0
        train_score = 0
        train_step = 0
        G, inputs, _, node_dict, edge_dict, _, _, _, _, _, \
        train_pos, val_pos, test_pos, train_neg, val_neg, test_neg = data
        try:
            current_best = 0
            current_patience = 0
            for epoch in np.arange(epochs) + 1:
                model.train()

                logits = self.forward_model(G, inputs, model, None)
                pos_score = self.get_score(logits[self.pre_src_type], logits[self.pre_dst_type], train_pos)
                neg_score = self.get_score(logits[self.pre_src_type], logits[self.pre_dst_type], train_neg)
                # TODO this loss using all embbeding
                loss = self.get_loss(pos_score, neg_score, logits)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                train_step += 1
                best_test_score, best_val_score, train_score = self.eval_process(G, best_test_score, best_val_score,
                                                                                 train_score, epoch, inputs,
                                                                                 None, loss,
                                                                                 model, optimizer, None, show_info,
                                                                                 data)
                if current_best < best_val_score:
                    current_best = best_val_score
                    current_patience = 0
                else:
                    current_patience += 1
                    if current_patience >= patience:
                        break
        except Exception as e:
            #  TODO  a right way to process OOM Error
            res = (0, 0, 0) if return_test else (0, 0)
            logging.log(logging.WARNING, gnn_desc)
            logging.log(logging.WARNING, str(e))
            if hasattr(self, "args") and hasattr(self.args, "debug") and self.args.debug:
                raise e
            torch.cuda.empty_cache()
            if "CUDA" in str(e) or "cuda" in str(e) or "cuDNN" in str(e):
                res = (0, 0, 0) if return_test else (0, 0)
            elif self.data[-2] in str(e):
                logging.log(logging.WARNING, "wrong structure!")
            else:
                logging.log(logging.WARNING, "other errors!!!")
                if hasattr(self.args, "debug") and self.args.debug:
                    raise e

        self.show_global_info(best_test_score, best_val_score, gnn_desc, train_pos, train_score)
        model = model.to(torch.device("cpu"))
        self.reset()
        torch.cuda.empty_cache()
        if return_test:
            return model, best_val_score, best_test_score
        else:
            return model, best_val_score

    def calc_score(self, s, o):
        # simple distance
        score = torch.sum(s * o, dim=1)
        return score

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2))

    def get_loss(self, pos_score, neg_score, logits):
        predict_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score)) \
                       + F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
        reg_loss = self.regularization_loss(logits[self.pre_src_type]) + self.regularization_loss(
            logits[self.pre_dst_type])
        return predict_loss + reg_loss

    def forward_model(self, G, inputs, model, out_key):
        logits = model(G, inputs, out_key)
        return logits

    def get_score(self, src_embed, dst_embed, edge_index):
        score = self.calc_score(src_embed[edge_index[0]], dst_embed[edge_index[1]])
        labels = torch.sigmoid(score)
        return labels

    def eval_process(
            self,
            G,
            best_test_score,
            best_val_score,
            train_score,
            epoch,
            inputs,
            labels,
            loss,
            model,
            optimizer,
            out_key,
            show_info,
            data):
        if epoch % 10 == 0:
            train_pos_user_artist, val_pos_user_artist, test_pos_user_artist, \
            train_neg_user_artist, val_neg_user_artist, test_neg_user_artist = data[-6:]
            model.eval()
            logits = self.forward_model(G, inputs, model, None)

            train_pos_score = self.get_score(logits[self.pre_src_type], logits[self.pre_dst_type],
                                             train_pos_user_artist).cpu().detach().numpy()
            neg_index = torch.randint(train_neg_user_artist.size(1), size=(train_pos_user_artist.size(1),))
            tmp_neg_edge = (train_neg_user_artist[0][neg_index], train_neg_user_artist[1][neg_index])
            train_neg_score = self.get_score(logits[self.pre_src_type], logits[self.pre_dst_type],
                                             tmp_neg_edge).cpu().detach().numpy()

            val_pos_score = self.get_score(logits[self.pre_src_type], logits[self.pre_dst_type],
                                           val_pos_user_artist).cpu().detach().numpy()
            val_neg_score = self.get_score(logits[self.pre_src_type], logits[self.pre_dst_type],
                                           val_neg_user_artist).cpu().detach().numpy()

            test_pos_score = self.get_score(logits[self.pre_src_type], logits[self.pre_dst_type],
                                            test_pos_user_artist).cpu().detach().numpy()
            test_neg_score = self.get_score(logits[self.pre_src_type], logits[self.pre_dst_type],
                                            test_neg_user_artist).cpu().detach().numpy()

            train_score = self.evaluator(
                np.concatenate([np.ones_like(train_pos_score), np.zeros_like(train_neg_score)], axis=0),
                np.concatenate([train_pos_score, train_neg_score], axis=0))
            val_score = self.evaluator(
                np.concatenate([np.ones_like(val_pos_score), np.zeros_like(val_neg_score)], axis=0),
                np.concatenate([val_pos_score, val_neg_score], axis=0))
            test_score = self.evaluator(
                np.concatenate([np.ones_like(test_pos_score), np.zeros_like(test_neg_score)], axis=0),
                np.concatenate([test_pos_score, test_neg_score], axis=0))
            if best_val_score < val_score:
                best_val_score = val_score
                best_test_score = test_score
            if show_info:
                print(
                    'Epoch: %d LR: %.5f Loss %.4f, Train Acc %.4f, Val Acc %.4f (Best %.4f), '
                    'Test Acc %.4f (Best %.4f)' % (
                        epoch,
                        optimizer.param_groups[0]['lr'],
                        loss,
                        train_score,
                        val_score,
                        best_val_score,
                        test_score,
                        best_test_score,
                    ))
        return best_test_score, best_val_score, train_score

    def show_global_info(self, best_test_score, best_val_score, gnn_desc, train_idx, train_score):
        info = f"train action:{gnn_desc} | Number of train datas:{2 * train_idx.size(1)}|" \
               f"train_score:{train_score:.4f},val_score:{best_val_score:.4f},test_score:{best_test_score:.4f}"
        if hasattr(self.args, "logger"):
            self.args.logger.info(info)
        else:
            print(info)


class AggrManagerSK(AggrManager):

    def build_gnn(self, actions, show_param=True):
        self.build_embed_layer()
        self_loop = self.args.self_loop if hasattr(self.args, "self_loop") else False
        model = SkipConnectionNet(actions,
                                  node_dict=self.node_dict,
                                  edge_dict=self.edge_dict,  # {str:int} map edge type to their index
                                  n_inp=self.in_feats,
                                  n_hid=self.args.n_hid,
                                  n_out=self.n_classes,
                                  dropout=self.dropout,
                                  n_layers=2,
                                  task_specific_encoder=self.embed_layer,
                                  n_heads=4,
                                  self_loop=self_loop)
        if show_param:
            total_params = sum(p.numel() for p in model.parameters())
            total_trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad)
            self.logger.info(f'{total_params:,} total parameters, {total_trainable_params:,} training parameters.')
        return model
