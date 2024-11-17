import torch

from .base import BaseSolver

class GreedySolver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)

    def initialize_state(self):
        pass

    def choose_arm(self):
        # extract args
        args           = self.args
        Q              = args.n_arm
        active_indices = self.active_indices
        # select arms randomly
        self.arms[active_indices] = Q - 1

    def update_state(self):
        pass
