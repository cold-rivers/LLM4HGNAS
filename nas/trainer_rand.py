import glob
import os
import time
import torch
import numpy as np

import nas.utils as utils
from nas.trainer import Trainer


class RandomTrainer(Trainer):

    def train_controller(self):
        """
            Train controller to find better structure.
        """
        # print("*" * 35, "training controller", "*" * 35)
        self.args.logger.info("training controller start")

        for gnn in self.search_space_cls.random_sample(self.args.controller_max_step):
            # evaluate candidate gnn
            _ = self.evaluate_gnn(gnn)

        self.args.logger.info("training controller over")

    def derive(self, sample_num=None):
        """
        sample a serial of structures, and return the best structure.
        """
        if sample_num is None and self.args.derive_from_history:
            return self.derive_from_history()
        else:
            if sample_num is None:
                sample_num = self.args.derive_num_sample
            if sample_num <= 0:
                return "", 0, None

            max_val = 0
            best_actions = None
            for gnn in self.search_space_cls.random_sample(sample_num):
                # evaluate candidate gnn
                val_score, test_score = self.evaluate(gnn)
                if val_score > max_val:
                    max_val = val_score

            self.args.logger.info(f'derive |action:{best_actions} |max_R: {max_val:8.6f}')
            # _, test_score = self.evaluate(best_actions)
            return best_actions, test_score, None
