import torch.nn.functional as F
import torch.nn as nn
import torch

class DQN(nn.Module):

    def __init__(self, args):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(args.n_observation, args.n_hidden)
        self.layer2 = nn.Linear(args.n_hidden, args.n_hidden)
        self.layer3 = nn.Linear(args.n_hidden, args.n_action)

    def forward(self, x, m):
        x = torch.cat((x[:, None], m[None, :].repeat(x.shape[0], 1)), dim=1)
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
