import torch

from .base import BaseSolver

class WeightedRandomSolver(BaseSolver):

    def __init__(self, args):
        super().__init__(args)

    def initialize_state(self):
        pass

    def choose_arm(self):
        # extract args
        args           = self.args
        Q              = args.n_arm
        active_indices = self.active_indices
        n_active       = len(active_indices)
        # select arms randomly
        if Q == 3:
            p = torch.tensor([0.1, 0.25, 0.65])
            random_choice = torch.multinomial(p,
                                              num_samples=n_active,
                                              replacement=True).to(dtype=torch.int)
            self.arms[active_indices] = random_choice
        elif Q == 5:
            p = torch.tensor([0.025, 0.05,  0.125, 0.15, 0.65])
            random_choice = torch.multinomial(p,
                                              num_samples=n_active,
                                              replacement=True).to(dtype=torch.int)
            self.arms[active_indices] = random_choice
        else:
            raise NotImplementedError

    def update_state(self):
        pass
