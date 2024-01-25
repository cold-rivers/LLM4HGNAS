import torch
import torch.nn.functional as F

import nas.utils as utils


class SimpleNASController(torch.nn.Module):
    def construct_action(self, actions):
        structure_list = []
        for single_action in actions:
            structure = []
            for action, action_name in zip(single_action, self.action_list):
                predicted_actions = self.search_space[action_name][action]
                structure.append(predicted_actions)
            structure_list.append(structure)
        return structure_list

    def __init__(self, args, search_space, action_list, controller_hid=100, cuda=True, mode="train",
                 softmax_temperature=5.0, tanh_c=2.5, integrate=False):
        """
        search_space:{operators:values}, search_space records operators names and its value names
        action_list: A list of operator names, assigning meaning to each step of the RNN output
        """
        if not self.check_action_list(action_list, search_space):
            raise RuntimeError("There are gnn_desc not contained in search_space")
        super(SimpleNASController, self).__init__()
        self.mode = mode
        # search space or operators set containing operators used to build GNN
        self.search_space = search_space
        # operator categories for each controller RNN output
        self.action_list = action_list
        self.controller_hid = controller_hid
        self.is_cuda = cuda
        self.integrate = integrate

        # set hyperparameters
        if args and args.softmax_temperature:
            self.softmax_temperature = args.softmax_temperature
        else:
            self.softmax_temperature = softmax_temperature
        if args and args.tanh_c:
            self.tanh_c = args.tanh_c
        else:
            self.tanh_c = tanh_c

        # build encoder
        self.num_tokens = []
        if self.integrate:
            for key in self.search_space:
                self.num_tokens.append(len(self.search_space[key]))
        else:
            for key in self.action_list:
                self.num_tokens.append(len(self.search_space[key]))

        num_total_tokens = sum(self.num_tokens)  # count operators numbers
        # Each operator in search space corresponds to one and only one embedding
        self.encoder = torch.nn.Embedding(num_total_tokens, controller_hid)

        # the core of controller
        self.lstm = torch.nn.LSTMCell(controller_hid, controller_hid)

        # build decoder
        if self.integrate:
            self._decoders = torch.nn.ModuleDict()
            for key in self.search_space:
                size = len(self.search_space[key])
                decoder = torch.nn.Linear(controller_hid, size)
                self._decoders[key] = decoder
        else:
            self._decoders = torch.nn.ModuleList()
            for key in self.action_list:
                size = len(self.search_space[key])
                decoder = torch.nn.Linear(controller_hid, size)
                self._decoders.append(decoder)

        self.reset_parameters()

    def update_action_list(self, action_list):
        """
        repalce current action_list
        """
        if not self.check_action_list(action_list, self.search_space):
            raise RuntimeError("There are gnn_desc not contained in search_space")

        self.action_list = action_list

    @staticmethod
    def check_action_list(action_list, search_space):
        if isinstance(search_space, dict):
            keys = search_space.keys()
        else:
            return False
        for each in action_list:
            if each in keys:
                pass
            else:
                return False
        return True

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        for decoder in self._decoders:
            if self.integrate:
                self._decoders[decoder].bias.data.fill_(0)
            else:
                decoder.bias.data.fill_(0)

    def forward(self,
                inputs,
                hidden,
                action_name,
                is_embed):
        # get the hidden embedding of operator values that processed by RNN and MLP
        embed = inputs

        hx, cx = self.lstm(embed, hidden)
        logits = self._decoders[action_name](hx)

        logits /= self.softmax_temperature

        # exploration
        if self.training:
            logits = (self.tanh_c * torch.tanh(logits))

        return logits, (hx, cx)

    def action_index(self, action_name):
        key_names = self.search_space.keys()
        for i, key in enumerate(key_names):
            if action_name == key:
                return i

    def sample(self, batch_size=1, with_details=False):

        if batch_size < 1:
            raise Exception(f'Wrong batch_size: {batch_size} < 1')
        # build inputs of RNN
        device = self.encoder.weight.device
        inputs = torch.zeros([batch_size, self.controller_hid])
        hidden = (torch.zeros([batch_size, self.controller_hid]), torch.zeros([batch_size, self.controller_hid]))
        inputs = inputs.to(device)
        hidden = (hidden[0].to(device), hidden[1].to(device))

        entropies = []
        log_probs = []
        actions = []
        for block_idx, action_name in enumerate(self.action_list):
            decoder_index = self.action_index(action_name) if self.integrate else block_idx
            # get hidden embedding
            logits, hidden = self.forward(inputs,
                                          hidden,
                                          action_name if self.integrate else block_idx,
                                          is_embed=(block_idx == 0))

            probs = F.softmax(logits, dim=-1)
            log_prob = F.log_softmax(logits, dim=-1)

            entropy = -(log_prob * probs).sum(1, keepdim=False)
            # sampling
            action = probs.multinomial(num_samples=1).data
            # recording entropies and log_probs
            selected_log_prob = log_prob.gather(
                1, utils.get_variable(action, requires_grad=False))

            entropies.append(entropy)
            log_probs.append(selected_log_prob[:, 0])

            # current action is fed as inputs of the new iteration
            inputs = utils.get_variable(
                action[:, 0] + sum(self.num_tokens[:decoder_index]),
                self.is_cuda,
                requires_grad=False)
            inputs = inputs.to(device)
            inputs = self.encoder(inputs)

            actions.append(action[:, 0])

        actions = torch.stack(actions).transpose(0, 1)
        dags = self.construct_action(actions)

        if with_details:
            # dags: [N_GNN, N_actions]
            # log_probs: [N_GNN, N_actions]
            # entropies: [N_GNN, N_actions]
            return dags, \
                   torch.cat(log_probs).view(batch_size, -1), \
                   torch.cat(entropies).view(batch_size, -1)

        return dags

    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.controller_hid)
        return (utils.get_variable(zeros, self.is_cuda, requires_grad=False),
                utils.get_variable(zeros.clone(), self.is_cuda, requires_grad=False))


class HierarchicalNASController(SimpleNASController):

    def __init__(self, args, search_space, action_list, controller_hid=100, cuda=True, mode="train",
                 softmax_temperature=5.0, tanh_c=2.5, integrate=False):
        """
        search_space:{operators:values}, search_space records operators names and its value names
        action_list: A list of operator names, assigning meaning to each step of the RNN output
        """
        if not self.check_action_list(action_list, search_space):
            raise RuntimeError("There are gnn_desc not contained in search_space")
        self.args = args
        super(SimpleNASController, self).__init__()
        self.mode = mode
        # search space or operators set containing operators used to build GNN
        self.search_space = search_space
        # operator categories for each controller RNN output
        self.action_list = action_list
        self.controller_hid = controller_hid
        self.is_cuda = cuda
        self.integrate = integrate

        # set hyperparameters
        if args and args.softmax_temperature:
            self.softmax_temperature = args.softmax_temperature
        else:
            self.softmax_temperature = softmax_temperature
        if args and args.tanh_c:
            self.tanh_c = args.tanh_c
        else:
            self.tanh_c = tanh_c

        # build encoder
        self.num_tokens = []
        if self.integrate:
            for key in self.search_space:
                self.num_tokens.append(len(self.search_space[key]))
        else:
            for key in self.action_list:
                self.num_tokens.append(len(self.search_space[key]))

        num_total_tokens = sum(self.num_tokens)  # count operators numbers
        # Each operator in search space corresponds to one and only one embedding
        self.encoder = torch.nn.Embedding(num_total_tokens, controller_hid)

        # the core of controller
        self.lstm = torch.nn.LSTMCell(controller_hid, controller_hid)

        # build decoder
        if self.integrate:
            self._decoders = torch.nn.ModuleDict()
            self._choices = torch.nn.ModuleDict()
            for key in self.search_space:
                size = len(self.search_space[key]) - 1
                decoder = torch.nn.Linear(controller_hid, size)
                self._decoders[key] = decoder
                choice = torch.nn.Linear(controller_hid, 2)
                self._choices[key] = choice
        else:
            self._decoders = torch.nn.ModuleList()
            self._choices = torch.nn.ModuleList()
            for key in self.action_list:
                if "aggr" not in key:  # gnn operators
                    size = len(self.search_space[key]) - 1
                    decoder = torch.nn.Linear(controller_hid, size)
                    choice = torch.nn.Linear(controller_hid, 2)
                    self._choices.append(choice)
                    self._decoders.append(decoder)
                else:
                    size = len(self.search_space[key])
                    decoder = torch.nn.Linear(controller_hid, size)
                    choice = None  # will not be used
                    self._decoders.append(decoder)
                    self._choices.append(choice)

        self.reset_parameters()

    def forward(self,
                inputs,
                hidden,
                action_name,
                is_embed):
        # get the hidden embedding of operator values that processed by RNN and MLP
        embed = inputs

        hx, cx = self.lstm(embed, hidden)
        choices = self._choices[action_name](hx)
        logits = self._decoders[action_name](hx)

        choices /= self.softmax_temperature
        logits /= self.softmax_temperature

        # exploration
        if self.training:
            logits = (self.tanh_c * torch.tanh(logits))
            choices = (self.tanh_c * torch.tanh(choices))

        return (choices, logits), (hx, cx)

    def sample(self, batch_size=1, with_details=False):

        if batch_size < 1:
            raise Exception(f'Wrong batch_size: {batch_size} < 1')
        # build inputs of RNN
        device = self.encoder.weight.device
        inputs = torch.zeros([batch_size, self.controller_hid])
        hidden = (torch.zeros([batch_size, self.controller_hid]), torch.zeros([batch_size, self.controller_hid]))
        inputs = inputs.to(device)
        hidden = (hidden[0].to(device), hidden[1].to(device))

        entropies = []
        log_probs = []
        actions = []
        for block_idx, action_name in enumerate(self.action_list):
            decoder_index = self.action_index(action_name) if self.integrate else block_idx
            if "aggr" not in action_name:  # operators
                inputs = self.sample_operator(action_name, actions, block_idx, decoder_index, device, entropies,
                                              hidden, inputs, log_probs)
            else:
                inputs = self.sample_aggr(action_name, actions, block_idx, decoder_index, device, entropies,
                                          hidden, inputs, log_probs)

        actions = torch.stack(actions).transpose(0, 1)
        dags = self.construct_action(actions)

        if with_details:
            # dags: [N_GNN, N_actions]
            # log_probs: [N_GNN, N_actions]
            # entropies: [N_GNN, N_actions]
            return dags, \
                   torch.cat(log_probs).view(batch_size, -1), \
                   torch.cat(entropies).view(batch_size, -1)

        return dags

    def sample_aggr(self, action_name, actions, block_idx, decoder_index, device, entropies, hidden, inputs,
                    log_probs):
        # get hidden embedding
        logits, hidden = super(HierarchicalNASController, self).forward(inputs,
                                                                        hidden,
                                                                        action_name if self.integrate else block_idx,
                                                                        is_embed=(block_idx == 0))

        probs = F.softmax(logits, dim=-1)
        log_prob = F.log_softmax(logits, dim=-1)

        entropy = -(log_prob * probs).sum(1, keepdim=False)
        # sampling
        action = probs.multinomial(num_samples=1).data
        # recording entropies and log_probs
        selected_log_prob = log_prob.gather(
            1, utils.get_variable(action, requires_grad=False))

        entropies.append(entropy)
        log_probs.append(selected_log_prob[:, 0])

        # current action is fed as inputs of the new iteration
        inputs = utils.get_variable(
            action[:, 0] + sum(self.num_tokens[:decoder_index]),
            self.is_cuda,
            requires_grad=False)
        inputs = inputs.to(device)
        inputs = self.encoder(inputs)

        actions.append(action[:, 0])
        return inputs

    def sample_operator(self, action_name, actions, block_idx, decoder_index, device, entropies, hidden, inputs,
                        log_probs):
        # get hidden embedding
        (choices, logits), hidden = self.forward(inputs,
                                                 hidden,
                                                 action_name if self.integrate else block_idx,
                                                 is_embed=(block_idx == 0))
        choices_prob = F.softmax(choices, dim=-1)
        probs = F.softmax(logits, dim=-1) * choices_prob[:, 1]
        log_prob = torch.log(probs)
        # the entropy of zeroize operator and other operators
        entropy = - (torch.log(choices_prob[:, 0]) * choices_prob[:, 0]).sum(dim=-1, keepdim=False) \
                  - (log_prob * probs).sum(1, keepdim=False)
        entropies.append(entropy)
        # sampling
        choices_action = choices_prob.multinomial(num_samples=1).data  # here is Bernoulli distribution
        action = probs.multinomial(num_samples=1).data
        choices_value = choices_action.item()
        # recording entropies and log_probs

        # TODO: works only when batch=1
        selected_choice_log_prob = choices_prob.gather(1, utils.get_variable(choices_action, requires_grad=False))
        selected_log_prob = log_prob.gather(1, utils.get_variable(action, requires_grad=False))
        if choices_value == 0:
            log_probs.append(selected_choice_log_prob[:, 0])
            action = choices_action
        else:
            log_probs.append(selected_log_prob[:, 0])
            # current action is fed as inputs of the new iteration
            action = action + 1
        inputs = utils.get_variable(
            action[:, 0] + sum(self.num_tokens[:decoder_index]),
            self.is_cuda,
            requires_grad=False)
        inputs = inputs.to(device)
        inputs = self.encoder(inputs)
        actions.append(action[:, 0])
        return inputs

        # # modify actions
        # modified_index = choices_action[:, 0] > 0
        # action[modified_index, 0] += 1
        # other_index = choices_action[:, 0] == 0
        # action[other_index, 0] = 0


class SparseNASController(HierarchicalNASController):

    def sample_inter_layer(self, action_name, actions, block_idx, decoder_index, device, entropies, hidden, inputs,
                           log_probs):
        if hasattr(self.args, "sparse_rate"):
            sparse_factor = self.args.sparse_rate - 1e-5
        else:
            sparse_factor = 0.5 - 1e-5
        # get hidden embedding
        (choices, logits), hidden = self.forward(inputs,
                                                 hidden,
                                                 action_name if self.integrate else block_idx,
                                                 is_embed=(block_idx == 0))
        choices_prob = F.softmax(choices, dim=-1)
        # choices_prob = choices_prob * torch.Tensor([[1-sparse_factor, sparse_factor]]).to(device)
        choices_prob = choices_prob @ torch.Tensor([[sparse_factor, 1 - sparse_factor], [1, 0]]).to(device)
        # choices_prob = F.softmax(choices_prob, dim=-1)
        probs = F.softmax(logits, dim=-1) * choices_prob[:, 1]
        log_prob = torch.log(probs)
        # the entropy of zeroize operator and other operators
        entropy = - (torch.log(choices_prob[:, 0]) * choices_prob[:, 0]).sum(dim=-1, keepdim=False) \
                  - (log_prob * probs).sum(1, keepdim=False)
        entropies.append(entropy)
        # sampling
        choices_action = choices_prob.multinomial(num_samples=1).data  # here is Bernoulli distribution
        action = probs.multinomial(num_samples=1).data
        choices_value = choices_action.item()
        # recording entropies and log_probs

        # TODO: works only when batch=1
        selected_choice_log_prob = choices_prob.gather(1, utils.get_variable(choices_action, requires_grad=False))
        selected_log_prob = log_prob.gather(1, utils.get_variable(action, requires_grad=False))
        if choices_value == 0:
            log_probs.append(selected_choice_log_prob[:, 0])
            action = choices_action
        else:
            log_probs.append(selected_log_prob[:, 0])
            # current action is fed as inputs of the new iteration
            action = action + 1
        inputs = utils.get_variable(
            action[:, 0] + sum(self.num_tokens[:decoder_index]),
            self.is_cuda,
            requires_grad=False)
        inputs = inputs.to(device)
        inputs = self.encoder(inputs)
        actions.append(action[:, 0])
        return inputs

    def sample_operator(self, action_name, actions, block_idx, decoder_index, device, entropies, hidden, inputs,
                        log_probs):
        if isinstance(action_name, tuple) and len(action_name) > 3:
            return self.sample_inter_layer(action_name, actions, block_idx, decoder_index, device, entropies, hidden,
                                           inputs, log_probs)
        else:
            return super(SparseNASController, self).sample_operator(action_name, actions, block_idx, decoder_index,
                                                                    device,
                                                                    entropies, hidden, inputs, log_probs)
