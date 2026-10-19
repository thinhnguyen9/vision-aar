import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

class DrogueTrajectoryPredictor(nn.Module):
    def __init__(self, in_dim, out_dim, horizon, hidden, layers, dropout, residual):
        super().__init__()
        if residual:
            assert in_dim == out_dim, "residual mode needs input and output to have the same features"
        self.lstm = nn.LSTM(in_dim, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, horizon * out_dim)
        self.horizon, self.out_dim, self.residual = horizon, out_dim, residual

    def forward(self, x):                              # x: (B, T, in_dim)
        _, (h, _) = self.lstm(x)                       # h: (layers, B, hidden)
        y = self.head(h[-1])                           # last layer's final hidden state
        y = y.view(-1, self.horizon, self.out_dim)     # (B, K, out_dim)
        if self.residual:
            y = y + x[:, -1:, :]                       # add last observed step to every future step
        return y
