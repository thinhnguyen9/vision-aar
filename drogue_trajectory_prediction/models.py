import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class DrogueLSTM_20261019(nn.Module):
    """
    Simple network with 1 LSTM layer and 1 linear output layer.

    Input: (batch, in_horizon, in_features).
    Output: (batch, out_horizon, out_features).
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=kwargs['in_features'],
            hidden_size=kwargs['lstm_hidden_size'],
            batch_first=True    # Puts batch dimension at the front
        )
        self.head = nn.Linear(
            in_features=kwargs['lstm_hidden_size'],
            out_features=kwargs['out_horizon'] * kwargs['out_features']
        )
        self.out_horizon, self.out_features = kwargs['out_horizon'], kwargs['out_features']

    def forward(self, x):                              # x: (batch, in_horizon, in_features)
        _, (h, _) = self.lstm(x)                       # h: (layers=1, batch, hidden)
        y = self.head(h[-1])                           # last layer's final hidden state
        y = y.view(-1, self.out_horizon, self.out_features)     # reshape to (batch, out_horizon, out_features)
        return y


class DrogueLSTM_20261021(nn.Module):
    """LSTM trajectory predictor with a state bypass to the output layer.

    Input:  (batch, time, num_features); the first `num_bypass` features
            are the target's current absolute state (x, y, vx, vy).
    Output: (batch, time, num_outputs), a prediction at every time step.
    """

    def __init__(self, num_features, num_outputs, lstm_units=256,
                 dense1=256, dense2=128, num_bypass=4):
        super().__init__()
        self.num_bypass = num_bypass
        self.lstm = nn.LSTM(num_features, lstm_units, batch_first=True)
        self.dense1 = nn.Linear(lstm_units, dense1)
        self.dense2 = nn.Linear(dense1, dense2)
        self.out = nn.Linear(dense2 + num_bypass, num_outputs)  # linear
        self.act = nn.Tanh()

    def forward(self, x, state=None):
        bypass = x[..., :self.num_bypass]
        h, state = self.lstm(x, state)         # (B, T, lstm_units)
        h = self.act(self.dense1(h))           # Linear acts on last dim,
        h = self.act(self.dense2(h))           # i.e. per time step
        h = torch.cat([h, bypass], dim=-1)     # skip connection
        return self.out(h), state              # state enables streaming
