import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
import math
import os

from .replay_memory import ReplayMemory, Transition
from ..base import BaseSolver
from .model import DQN

class Solver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)

    def initialize_state(self):
        # extract args
        args = self.args
        # initialize DQN
        self.policy_net = DQN(args)
        self.target_net = DQN(args)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # initialize optimizer
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=args.lr, amsgrad=True)
        self.memory = ReplayMemory(10000)

    def optimize_model(self):
        # extract args
        args = self.args
        #
        if len(self.memory) < args.batch_size:
            return
        # sample one transitions only, since they have different length
        assert args.batch_size == 1
        transitions = self.memory.sample(args.batch_size)
        batch = Transition(*zip(*transitions))
        # concatenate batch
        b_o = torch.cat(batch.observation)
        b_m = torch.cat(batch.mf_observation)
        b_a = torch.cat(batch.action).to(torch.long)
        b_o = torch.cat(batch.observation)
        b_r = torch.cat(batch.reward)
        # compute estimated Q
        q_values = self.policy_net(b_o, b_m).gather(1, b_a[:, None])
        # compute next Q
        # next_q_values = self.target_net(b_n_o, b_n_m).max(1).values
        # expected_q_values = next_q_values * args.gamma + b_r
        expected_q_values = b_r.unsqueeze(1)
        # loss function
        criterion = nn.SmoothL1Loss()
        loss = criterion(q_values, expected_q_values)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        return loss.item()

    def choose_arm(self):
        # extract args
        args       = self.args
        Q          = args.n_arm
        idx        = self.active_indices
        n_active   = len(idx)
        policy_net = self.policy_net
        # extract observation
        self.observation    = self.temporary_g_db[idx]
        self.mf_observation = self.population_profile / n_active
        # compute exploration rate
        eps = args.eps_end + (args.eps_start - args.eps_end) * math.exp(-1 * self.global_step / args.eps_decay)
        # select arm
        if np.random.rand() > eps:
            self.arms[idx] = torch.randint(low=0, high=Q, size=(n_active, ), dtype=torch.int)
        else:
            self.arms[idx] = policy_net(self.observation, self.mf_observation).max(1).indices.to(dtype=torch.int)
        self.action = self.arms[idx]

    def update_state(self):
        # extract args
        args     = self.args
        idx      = self.active_indices
        n_active = len(idx)
        # store transition in memory
        if args.mode == 'train':
            # store the transition in memory
            observation         = self.observation
            mf_observation      = self.mf_observation
            action              = self.action
            reward              = self.rewards[idx]
            self.memory.push(observation, mf_observation, action, reward)

            # perform one step of optimization
            self.loss = self.optimize_model()
            # soft update of the target network's weights
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * args.tau +\
                                             target_net_state_dict[key] * (1 - args.tau)
            self.target_net.load_state_dict(target_net_state_dict)
