import torch
import numpy as np
import torch.nn as nn

class RegressionModel(nn.Module):
    def __init__(self, p):
        super(RegressionModel, self).__init__()
        self.linear = nn.Linear(p, 1)

    def forward(self, x):
        # x * w + b
        x.view(x.size(0), -1)
        return self.linear(x)


