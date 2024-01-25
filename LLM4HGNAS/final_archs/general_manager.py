import torch.nn.functional as F

from data.data_util import *
from final_archs.tools import evaluate_results_nc
from hgnn.meta_manager import AggrManager
from link_predict.meta_manager import MetaOptLinkPredictorManager


class GeneralManagerNC(AggrManager):
    """
    给定模型,验证模型性能
    """

    def evaluate(self, model, device=None, svm_test=False):
        res = self.train(model, device, return_test_score=True, svm_test=svm_test)
        return res

    def eval_process(self, G, best_test_score, best_val_score, train_score, epoch, inputs, labels, loss, model,
                     optimizer, out_key, show_info, test_idx, train_idx, val_idx):
        if epoch % 5 == 0:
            model.eval()
            logits, embedding = self.forward_model(G, inputs, model, out_key, require_embed=True)
            pred = logits.argmax(1)
            pred = pred
            tmp_labels = labels

            train_score = self.evaluator(pred[train_idx].cpu(), tmp_labels[train_idx].cpu())
            val_score = self.evaluator(pred[val_idx].cpu(), tmp_labels[val_idx].cpu())
            test_score = self.evaluator(pred[test_idx].cpu(), tmp_labels[test_idx].cpu())
            if best_val_score < val_score:
                best_val_score = val_score
                best_test_score = test_score
                self.embedding_list[-1] = embedding[out_key].cpu().detach()
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

    def train(self, model, device=None, return_test_score=False, svm_test=False):
        args = self.args
        device = self.device if device is None else device
        count = 0
        flag = False
        flag, model = self.move_model_to_device(count, device, flag, model)
        G, features, labels, node_dict, edge_dict, train_idx, val_idx, test_idx, category, feat_sizes = self.data
        if not flag:  # fail to build model
            if return_test_score:
                return 0, 0, 0
            else:
                return 0, 0
        optimizer = torch.optim.Adam(model.parameters(), weight_decay=args.weight_decay)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.epochs, max_lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
        # scheduler = None
        loss_fn = F.cross_entropy
        if hasattr(self, "embedding_list"):
            self.embedding_list.append(0)
        else:
            self.embedding_list = [0]
        try:
            show_info = hasattr(args, "show_info") and args.show_info
            res = self.run_model(model, optimizer, scheduler, loss_fn,
                                 self.args.epochs,
                                 self.data,
                                 return_test=return_test_score,
                                 gnn_desc=str(type(model)),
                                 show_info=show_info)
        except Exception as e:
            #  TODO  a right way to process OOM Error
            res = (0, 0, 0) if return_test_score else (0, 0)
            logging.log(logging.WARNING, str(model))
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
        if svm_test:
            embedding = self.embedding_list[-1]
            test_idx = test_idx.cpu()
            svm_macro_f1_list, svm_micro_f1_list = evaluate_results_nc(
                embedding[test_idx].numpy(), labels[test_idx].cpu().detach().numpy(), None)

        if return_test_score:
            model, val_score, test_score = res
            if svm_test:
                return val_score, test_score, svm_macro_f1_list, svm_micro_f1_list
            else:
                return val_score, test_score
        else:
            model, val_score = res
            return val_score

    def forward_model(self, G, inputs, model, out_key, require_embed=False):
        logits, embedding = model(G, inputs, out_key)
        if not require_embed:
            return logits
        else:
            return logits, embedding

    def move_model_to_device(self, count, device, flag, model):
        while not flag and count < 10:
            try:
                count += 1
                torch.cuda.empty_cache()
                self.data_to_devices(device)
                model = model.to(device)
                flag = True
            except Exception as e:
                logging.log(logging.WARNING, model)
                logging.log(logging.WARNING, str(e))
                raise e
        return flag, model


class GeneralManagerLP(MetaOptLinkPredictorManager):
    """
    给定模型,验证模型性能
    """

    def load_dataset(self, args):
        supervised = args.supervised if hasattr(args, "supervised") else False
        assert not supervised, "Link prediction is unsupervised task"
        self.data = list(load_link_prediction_data(dataset=args.dataset))
        self.num_of_nodes = {ntype: self.data[0].num_nodes(ntype) for ntype in self.data[0].ntypes}
        self.node_dict = self.data[3]
        self.edge_dict = self.data[4]
        self.pre_src_type, self.pre_dst_type, self.pre_link = load_predict_links(args.dataset)
        self.n_classes = args.n_hid
        self.args.in_feats = self.in_feats = self.args.n_inp

    def evaluate(self, model, device=None):
        _, val_score, test_score = self.train(model, device, return_test_score=True)
        return val_score, test_score

    def train(self, model, device=None, return_test_score=False):
        args = self.args
        device = self.device if device is None else device
        count = 0
        flag = False
        flag, model = self.move_model_to_device(count, device, flag, model)

        if not flag:  # fail to build model
            if return_test_score:
                return 0, 0, 0
            else:
                return 0, 0
        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=args.weight_decay)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=args.epochs, max_lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
        loss_fn = F.cross_entropy
        try:
            show_info = hasattr(args, "show_info") and args.show_info
            res = self.run_model(model, optimizer, scheduler, loss_fn,
                                 self.args.epochs,
                                 self.data,
                                 return_test=return_test_score,
                                 gnn_desc=str(type(model)),
                                 show_info=show_info)
        except Exception as e:
            #  TODO  a right way to process OOM Error
            res = (0, 0, 0) if return_test_score else (0, 0)
            logging.log(logging.WARNING, str(model))
            logging.log(logging.WARNING, str(e))
            if hasattr(args, "debug") and args.debug:
                raise e
            torch.cuda.empty_cache()
            if "CUDA" in str(e) or "cuda" in str(e) or "cuDNN" in str(e):
                res = (0, 0, 0) if return_test_score else (0, 0)
            # elif self.data[-2] in str(e):
            #     logging.log(logging.WARNING, "wrong structure!")
            else:
                logging.log(logging.WARNING, "other errors!!!")
        finally:
            del model
            self.reset()
            torch.cuda.empty_cache()
        if return_test_score:
            model, val_score, test_score = res
            return val_score, val_score, test_score
        else:
            model, val_score = res
            return val_score, val_score

    def move_model_to_device(self, count, device, flag, model):
        while not flag and count < 10:
            try:
                count += 1
                torch.cuda.empty_cache()
                self.data_to_devices(device)
                model = model.to(device)
                flag = True
            except Exception as e:
                logging.log(logging.WARNING, model)
                logging.log(logging.WARNING, str(e))
                if hasattr(self.args, "debug") and self.args.debug:
                    raise e
        return flag, model
