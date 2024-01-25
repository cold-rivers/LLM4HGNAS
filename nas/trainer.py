import glob
import os
import time
import torch
import numpy as np

import nas.utils as utils
from nas.controller import SimpleNASController


class Trainer(object):
    """Manage the training process"""

    def __init__(self, args,
                 search_space_cls=None,
                 nas_controller=None,
                 search_space=None,
                 action_list=None,
                 gnn_model_manager_cls=None,
                 form_gnn_function=None):
        """
        Constructor for training algorithm.
        Build sub-model manager and controller.
        Build optimizer and cross entropy loss for controller.

        Args:
            args: From command line, picked up by `argparse`.
        """
        self.args = args
        self.controller_step = 0  # counter for controller
        self.epoch = 0
        self.start_epoch = 0
        self.baseline = None

        # search space related
        self.search_space_cls = search_space_cls
        self.search_space = search_space
        self.action_list = action_list
        self.form_gnn_function = form_gnn_function

        # nas controller related
        self.controller = nas_controller
        self.controller_optim = None

        # child training manager related
        self.gnn_manager_cls = gnn_model_manager_cls
        self.gnn_manager = None

        # record gnn infos
        self.gnn_history = []  # history gnn architecture description
        self.val_history = []  # history validation score

        # build necessary part of Trainer
        self.build_search_space()
        self.build_gnn_manager()
        self.build_controller()
        self.build_controller_opt()
        self.build_logger()
        if self.args.mode == "derive":
            self.load_model()

    def build_logger(self):
        self.args.logger = utils.get_logger()

    def build_search_space(self):
        if self.search_space_cls is None:
            # default search space
            raise RuntimeError("search_space_cls should not be None!")
        assert self.search_space_cls is not None
        assert self.search_space is not None
        assert self.action_list is not None

    def build_gnn_manager(self):
        if self.gnn_manager_cls is None:
            raise RuntimeError("gnn_manager_cls should not be None!")

        if self.gnn_manager is None:
            self.gnn_manager = self.gnn_manager_cls(self.args)

    def build_controller(self):
        if self.controller is None:
            assert self.gnn_manager is not None, "variants are required for building controller"
            assert self.gnn_manager_cls is not None, "gnn_manager_cls are required for building controller"
            self.controller = SimpleNASController(self.args, action_list=self.action_list,
                                                  search_space=self.search_space,
                                                  cuda=self.args.cuda,
                                                  controller_hid=self.args.controller_hid, )
        if self.args.cuda:
            self.controller.cuda()

    def build_controller_opt(self):
        assert self.controller is not None, "Controller are required for building optimizer!"
        name = self.args.controller_optim
        if name.lower() == 'sgd':
            optim = torch.optim.SGD
        elif name.lower() == 'adam':
            optim = torch.optim.Adam

        self.controller_optim = optim(self.controller.parameters(), lr=self.args.controller_lr)

    def form_gnn_info(self, gnn):
        """
            Parsing the architecture info sampled by the controller
        """
        if self.form_gnn_function is None:
            return gnn
        else:
            return self.form_gnn_function(gnn, self.args)

    def train(self):
        """
        Each epoch consists of two phase:
        - In the first phase, the controller parameters are trained.
        - In the second phase, the GNN architectures are sampled and verified with fixed controller  parameters are trained.
        """

        for self.epoch in range(self.start_epoch, self.args.max_epoch):
            # 1. Training the controller parameters theta
            self.train_controller()
            # 2. Derive architectures:
            self.derive(sample_num=self.args.derive_num_sample)

            if self.epoch % self.args.save_epoch == 0:
                self.save_model()

        self.save_model()

        if self.args.derive_finally:
            # Choose the best GNN architecture from the sampled models
            best_actions, test_score, test_std = self.derive_from_history()
            self.args.logger.info("best structure:" + str(best_actions))
            return best_actions, test_score, test_std

    def train_controller(self):
        """
            Train controller to find better structure.
        """
        # print("*" * 35, "training controller", "*" * 35)
        self.args.logger.info("training controller start")
        model = self.controller
        model.train()

        for step in range(self.args.controller_max_step):
            # sample gnn
            gnn_list, log_probs, entropies = self.controller.sample(with_details=True)
            # evaluate candidate gnn
            metric_list = self.evaluate_gnn(gnn_list)
            # calculate reward
            moving_reward = self.get_reward(metric_list, entropies, log_probs.device)
            if moving_reward is not None:  # No Error, reward is generated.
                pass
            else:
                continue  # CUDA Error happens, drop structure and step into next iteration

            # policy loss
            loss = -log_probs * moving_reward
            if self.args.entropy_mode == 'regularizer':
                loss -= self.args.entropy_coeff * entropies
            loss = loss.sum()  # or loss.mean()

            # train the controller parameter \theta
            self.controller_optim.zero_grad()
            loss.backward()

            if self.args.controller_grad_clip > 0:
                torch.nn.utils.clip_grad_norm(model.parameters(),
                                              self.args.controller_grad_clip)
            self.controller_optim.step()
            self.controller_step += 1

        self.args.logger.info("training controller over")

    def moving_average(self, rewards, device):
        # moving average baseline
        if self.baseline is None:
            self.baseline = rewards.mean()
        else:
            decay = self.args.ema_baseline_decay
            self.baseline = decay * self.baseline + (1 - decay) * rewards.mean()
        self.baseline = self.baseline.detach()

        adv = rewards - self.baseline
        return adv.detach()

    def get_reward(self, metric_list, entropies, device=None):
        """
        Computes the reward of a single sampled model on validation data.
        """
        metric_list = torch.Tensor(metric_list).to(device).view(-1,1)
        if self.args.entropy_mode == 'reward':
            rewards = metric_list + self.args.entropy_coeff * entropies
        elif self.args.entropy_mode == 'regularizer':
            rewards = metric_list * torch.ones_like(entropies)
        else:
            raise NotImplementedError(f'Unkown entropy mode: {self.args.entropy_mode}')
        # rewards = torch.Tensor(rewards).to(device)
        return self.moving_average(rewards, device)

    def evaluate_gnn(self, gnn_list):
        # The description of GNN is recorded in list or dict

        if isinstance(gnn_list, dict): # single structure
            gnn_list = [gnn_list]

        if isinstance(gnn_list[0], list) or isinstance(gnn_list[0], dict):  # a list of gnn
            pass
        else:
            gnn_list = [gnn_list]  # single structure

        reward_list = []
        for gnn in gnn_list:
            val_score, test_score = self.evaluate(gnn)
            reward_list.append(val_score)
            # record informations
        return reward_list

    def evaluate(self, gnn):
        """
        Evaluate a structure on the validation set.
        """
        if gnn is None:
            return 0, 0
        self.controller.eval()
        gnn_desc = self.form_gnn_info(gnn)
        results = self.gnn_manager.evaluate(gnn_desc, format=self.args.format)
        if results:
            val_score, test_score = results
        else:
            val_score, test_score = 0, 0

        self.gnn_history.append(gnn)
        self.val_history.append(val_score)
        # self.args.logger.info(f'eval | {gnn_desc} | val_score: {val_score:8.2f} | test_scores: {test_score:8.2f}')
        return val_score, test_score

    def derive_from_history(self, filename=None, top_n=5, n_repeats=10):
        """
        find the best structure from history (recorded by GNNManager)
        """
        max_index = np.argsort(self.val_history)
        # retrain top N gnn and select the best one
        best_structure = ""
        best_score = 0
        for index in max_index[-top_n:]:
            actions = self.gnn_history[index]
            val_scores_list = []
            for i in range(5):
                val_acc, test_acc = self.evaluate(actions)
                val_scores_list.append(val_acc)

            tmp_score = np.mean(val_scores_list)
            self.args.logger.info(f"Top N gnn:{actions}, validation score:{tmp_score}")
            if tmp_score > best_score:
                best_score = tmp_score
                best_structure = actions

        # retrain the best gnn and get its performance
        self.args.logger.info("best gnn:" + str(best_structure))
        test_scores_list = []
        for i in range(n_repeats):
            val_acc, test_acc = self.evaluate(best_structure)
            test_scores_list.append(test_acc)
        print(f"best results: {best_structure}: {np.mean(test_scores_list):.8f} +/- {np.std(test_scores_list)}")
        return best_structure, np.mean(test_scores_list), np.std(test_scores_list)

    def derive(self, sample_num=None):
        """
        sample a serial of structures, and return the best structure.
        """
        if sample_num is None and self.args.derive_from_history:
            return self.derive_from_history()
        else:
            if sample_num is None:
                sample_num = self.args.derive_num_sample
            if sample_num <=0:
                return "", 0, None

            max_val = 0
            best_actions = None
            for _ in range(sample_num):
                gnn_list, _, _ = self.controller.sample(with_details=True)
                val_score, test_score = self.evaluate(gnn_list[0])
                if val_score > max_val:
                    max_val = val_score
                    best_actions = gnn_list[0]

            self.args.logger.info(f'derive |action:{best_actions} |max_R: {max_val:8.6f}')
            # _, test_score = self.evaluate(best_actions)
            return best_actions, test_score, None

    @property
    def controller_path(self):
        return f'{self.args.dataset}/controller_epoch{self.epoch}_step{self.controller_step}.pth'

    @property
    def controller_optimizer_path(self):
        return f'{self.args.dataset}/controller_epoch{self.epoch}_step{self.controller_step}_optimizer.pth'

    def get_saved_models_info(self):
        paths = glob.glob(os.path.join(self.args.dataset, '*.pth'))
        paths.sort()

        def get_numbers(items, delimiter, idx, replace_word, must_contain=''):
            return list(set([int(
                name.split(delimiter)[idx].replace(replace_word, ''))
                for name in items if must_contain in name]))

        basenames = [os.path.basename(path.rsplit('.', 1)[0]) for path in paths]
        epochs = get_numbers(basenames, '_', 1, 'epoch')
        shared_steps = get_numbers(basenames, '_', 2, 'step', 'shared')
        controller_steps = get_numbers(basenames, '_', 2, 'step', 'controller')

        epochs.sort()
        shared_steps.sort()
        controller_steps.sort()

        return epochs, shared_steps, controller_steps

    def save_model(self):

        torch.save(self.controller.state_dict(), self.controller_path)
        torch.save(self.controller_optim.state_dict(), self.controller_optimizer_path)

        self.args.logger.info(f'[*] SAVED: {self.controller_path}')

        epochs, shared_steps, controller_steps = self.get_saved_models_info()

        for epoch in epochs[:-self.args.max_save_num]:
            paths = glob.glob(
                os.path.join(self.args.dataset, f'*_epoch{epoch}_*.pth'))

            for path in paths:
                utils.remove_file(path)

    def load_model(self):
        epochs, shared_steps, controller_steps = self.get_saved_models_info()

        if len(epochs) == 0:
            self.args.logger.info(f'[!] No checkpoint found in {self.args.dataset}...')
            return

        self.epoch = self.start_epoch = max(epochs)
        self.controller_step = max(controller_steps)

        self.controller.load_state_dict(
            torch.load(self.controller_path))
        self.controller_optim.load_state_dict(
            torch.load(self.controller_optimizer_path))
        self.args.logger.info(f'[*] LOADED: {self.controller_path}')
