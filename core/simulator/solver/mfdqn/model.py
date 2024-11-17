import torch.nn.functional as F
import torch.nn as nn
import torch

class DQN(nn.Module):

    def __init__(self, args):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(args.n_observation, args.n_hidden)
        self.layer2 = nn.Linear(args.n_hidden, args.n_hidden)
        self.layer3 = nn.Linear(args.n_hidden, args.n_action)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
