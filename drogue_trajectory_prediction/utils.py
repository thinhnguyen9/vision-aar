import numpy as np
import importlib
import torch
import pandas as pd
import matplotlib.pyplot as plt
import json


class DrogueTrajectoryPredictor():
    """
    Wrapper for some prediction models, used for online inference on CPU
    w/ batch=1 (predict one step at a time). Also handles normalization.
    """
    def __init__(self, model_path, method, **kwargs):
        if method not in ['lstm', 'rigid-drogue', 'const-vel']:
            raise ValueError(f"Unsupported method: {method}. Only accept lstm/rigid-drogue/const-vel.")
        self.method = method

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.config = ckpt['config']
        self.mean = np.asarray(ckpt['mean'], dtype=np.float32)
        self.std = np.asarray(ckpt['std'], dtype=np.float32)
        self.x_shape = (self.config['lstm']['in_horizon'], self.config['lstm']['in_features'])

        if self.method == 'lstm':
            module = importlib.import_module(self.config['lstm']['model']['module'])
            model_class = getattr(module, self.config['lstm']['model']['class'])
            self.model = model_class(
                in_features=self.config['lstm']['in_features'],
                out_features=self.config['lstm']['out_features'],
                out_horizon=self.config['lstm']['out_horizon'],
                lstm_hidden_size=self.config['lstm']['hidden_size']
            )
            self.model.load_state_dict(ckpt['state_dict'])
            self.model.eval()
            self.device = torch.device("cpu")
            self.model.to(self.device)
    
    def predict(self, x):
        """
        x: (in_horizon, in_features), original units (NOT normalized)!!
        y: (out_horizon, out_features), original units
        """
        if x.shape != self.x_shape:
            raise ValueError(f"Input x has shape {x.shape}, but expected {self.x_shape}")
        x = np.asarray(x, dtype=np.float32)

        if self.method == 'lstm':   # only LSTM needs normalization
            x = (x - self.mean) / self.std
            xb = torch.from_numpy(x).unsqueeze(0).to(self.device)   # (1, in_horizon, in_features)
            with torch.no_grad():
                y = self.model(xb).squeeze(0).cpu().numpy()   # (out_horizon, out_features)
            y = y * self.std + self.mean        # denormalize

        elif self.method == 'rigid-drogue':
            y = np.tile(self.mean, (self.config['lstm']['out_horizon'], 1))

        elif self.method == 'const-vel':
            # constant velocity model: assume last observed velocity continues
            last_pos = x[-1:]
            last_vel = x[-1:] - x[-2:-1]
            preds = []
            for _ in range(self.config['lstm']['out_horizon']):
                next_pos = last_pos + last_vel
                preds.append(next_pos)
                last_pos = next_pos
            y = np.concatenate(preds, axis=0)

        return y


def resample_csv_log(csv_path, dt, time_col, data_cols, max_time_gap=1.0):
    """
    Resample a CSV log with linear interpolation at a uniform time step.

    When large gaps are detected, the time series is split into separate
    segments to prevent linear interpolation from generating fake data.
    """
    time = []
    time_series = []
    try:
        df = pd.read_csv(csv_path).sort_values(time_col)
    except Exception as e:
        print(f"[resample_csv_log][Error] Failed to read {csv_path}: {e}")
        return time, time_series
    df = df.drop_duplicates(subset=time_col)       # interpolation needs strictly increasing t
    if df.empty:
        return time, time_series

    t = df[time_col].to_numpy()
    gap_indices = np.flatnonzero(np.diff(t) > max_time_gap)
    # if gap_indices.size:
    #     print(f"[resample_csv_log][Warning] Large time gap ({np.max(np.diff(t)):.4f} s) detected in {csv_path}!")

    segment_starts = np.r_[0, gap_indices + 1]
    segment_ends = np.r_[gap_indices + 1, len(t)]
    for start, end in zip(segment_starts, segment_ends):
        segment_t = t[start:end]
        # print(f"[resample_csv_log] Segment {len(time_series) + 1}: [{segment_t[0]:.2f}, {segment_t[-1]:.2f}] s")
        if segment_t[-1] <= segment_t[0]:
            continue

        t_new = np.arange(segment_t[0], segment_t[-1], dt)
        if t_new.size < 2:
            continue

        segment_data = df.iloc[start:end]
        out = np.empty((t_new.size, len(data_cols)))
        for i, c in enumerate(data_cols):
            out[:, i] = np.interp(t_new, segment_t, segment_data[c].to_numpy())
        time.append(t_new)
        time_series.append(out)

    return time, time_series


def make_data_windows(time_series, Nin, Nout, sampling_interval=5):
    """
    Dimensions:
        time_series: (N, F) resampled data, N time steps, F features
        Nin: # of input time steps
        Nout: # of output time steps
        X: (num_windows, Nin, F) input sequences for LSTM
        Y: (num_windows, Nout, F) output sequences for LSTM
    """
    X, Y = [], []
    for i in range(0, len(time_series) - Nin - Nout + 1, sampling_interval):
        X.append(time_series[i : i+Nin])
        Y.append(time_series[i+Nin : i+Nin+Nout])
    if not X:
        return (
            np.empty((0, Nin, time_series.shape[1]), dtype=time_series.dtype),
            np.empty((0, Nout, time_series.shape[1]), dtype=time_series.dtype),
        )
    return np.stack(X), np.stack(Y)


def rmse_per_horizon_step(predictions, ground_truths):
    """
    Args:
        predictions      : (N, out_horizon, out_features) predicted sequences
        ground_truths    : (N, out_horizon, out_features) ground truth sequences
    
    Returns:
        rmse_per_feature : (out_horizon, out_features) RMSE for each horizon step, for each output feature
        rmse_total       : (out_horizon,) RMSE for each horizon step
    """
    # Remove NaN time steps
    valid_rows = (
        np.isfinite(predictions).all(axis=(1, 2))
        & np.isfinite(ground_truths).all(axis=(1, 2))
    )
    predictions = predictions[valid_rows]
    ground_truths = ground_truths[valid_rows]

    # Compute RMSEs
    err = predictions - ground_truths
    err_norm = np.linalg.norm(err, axis=2)     # (N, out_horizon)
    rmse_per_feature = np.sqrt((err ** 2).mean(axis=0))     # (out_horizon, out_features)
    rmse_total = np.sqrt((err_norm ** 2).mean(axis=0))      # (out_horizon,)
    return rmse_per_feature, rmse_total
