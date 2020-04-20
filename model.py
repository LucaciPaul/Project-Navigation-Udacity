import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, inLayer_output_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.inpt = nn.Linear(state_size, inLayer_output_units)
        self.fc2 = nn.Linear(inLayer_output_units, fc2_units)
        self.out = nn.Linear(fc2_units, action_size)
        
    def forward(self, state):
        out_actions = F.relu(self.inpt(state))
        out_actions = F.relu(self.fc2(out_actions))
        out_actions = self.out(out_actions)
        
        return out_actions