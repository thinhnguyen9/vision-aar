import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import DrogueTrajectoryPredictor
from make_drogue_trajectory_dataset import resample_log, LSTM_PAST_STEPS, LSTM_TIME_STEP, CSV_DROGUE_COLS

# ------------------------- config -------------------------
MODEL_PATH = "drogue_trajectory_prediction/drogue_trajectory_lstm.pt"
CSV_LOG_PATH = "logs/lqr_gt_1789775576.csv"
# LSTM_PAST_STEPS = 40                    # history window length: must match the LSTM_PAST_STEPS used to build the training windows
# LSTM_TIME_STEP = 0.05                 # resample period in seconds (only used for plot time axes)
AXIS_NAMES = ["x", "y", "z"]


# ------------------------- load model -------------------------
ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)  # file also holds numpy mean/std
cfg, mean, std = ckpt["config"], ckpt["mean"], ckpt["std"]
LSTM_FUTURE_STEPS = cfg["horizon"]
model = DrogueTrajectoryPredictor(cfg["in_dim"], cfg["out_dim"], LSTM_FUTURE_STEPS, cfg["hidden"],
                        cfg["layers"], cfg["dropout"], cfg["residual"])
model.load_state_dict(ckpt["state_dict"])
model.eval()

# ------------------------- load run -------------------------
resampled_df = resample_log(CSV_LOG_PATH)
resampled_data = resampled_df[CSV_DROGUE_COLS].to_numpy()     # time steps x features (x/y/z)
pos = resampled_data.astype(np.float32)          # (N, 3)
N, F = pos.shape
assert F == cfg["in_dim"], f"run has {F} features but the model expects {cfg['in_dim']}"
assert N >= LSTM_PAST_STEPS + LSTM_FUTURE_STEPS, f"run is too short: need at least LSTM_PAST_STEPS+LSTM_FUTURE_STEPS = {LSTM_PAST_STEPS + LSTM_FUTURE_STEPS} steps, got {N}"

pos_norm = (pos - mean) / std                         # same normalization as training

# ------------------------- predict at every time step -------------------------
# t = index of the last observed step. Window = steps t-LSTM_PAST_STEPS+1 .. t, future = steps t+1 .. t+LSTM_FUTURE_STEPS.
# Only t values with a full history AND full ground-truth future are used.
t_idx = np.arange(LSTM_PAST_STEPS - 1, N - LSTM_FUTURE_STEPS)
windows = np.stack([pos_norm[t - LSTM_PAST_STEPS + 1: t + 1] for t in t_idx])      # (M, LSTM_PAST_STEPS, F)

with torch.no_grad():
    pred_norm = model(torch.from_numpy(windows)).numpy()              # (M, LSTM_FUTURE_STEPS, F)

pred = pred_norm * std[:cfg["out_dim"]] + mean[:cfg["out_dim"]]       # back to physical units
truth = np.stack([pos[t + 1: t + LSTM_FUTURE_STEPS + 1] for t in t_idx])              # (M, LSTM_FUTURE_STEPS, F)

# baseline: rigid drogue
rigid_pred = np.tile(mean, (LSTM_FUTURE_STEPS, 1))
baseline = np.tile(rigid_pred, (windows.shape[0], 1, 1))

# ------------------------- errors -------------------------
err = pred - truth                                                    # (M, LSTM_FUTURE_STEPS, F)
rmse_axis = np.sqrt((err ** 2).mean(axis=0))                          # (LSTM_FUTURE_STEPS, F) per future step and axis
rmse_total = np.sqrt((err ** 2).sum(axis=2).mean(axis=0))             # (LSTM_FUTURE_STEPS,)   3D position error
rmse_baseline = np.sqrt(((baseline - truth) ** 2).sum(axis=2).mean(axis=0)) # (LSTM_FUTURE_STEPS,)   baseline 3D error

print(f"evaluated {len(t_idx)} time steps")
print(f"{'step':>5} {'LSTM 3D RMSE':>14} {'rigid-drogue':>12}")
for k in sorted(set([1, max(1, LSTM_FUTURE_STEPS // 4), max(1, LSTM_FUTURE_STEPS // 2), LSTM_FUTURE_STEPS])):
    print(f"{k:5d} {rmse_total[k - 1]:14.4f} {rmse_baseline[k - 1]:12.4f}")

# ------------------------- plot 1: predicted vs ground truth at four time steps -------------------------
plot_count = 4
if len(t_idx) < plot_count:
    raise ValueError(f"need at least {plot_count} valid prediction time steps to plot")
plot_indices = np.linspace(0, len(t_idx) - 1, plot_count, dtype=int)
plot_steps = t_idx[plot_indices]

fig, axes = plt.subplots(F, plot_count, figsize=(4.5 * plot_count, 2.8 * F),
                         sharex=False, squeeze=False)
for col, (m, t_plot) in enumerate(zip(plot_indices, plot_steps)):
    hist_t = (np.arange(t_plot - LSTM_PAST_STEPS + 1, t_plot + 1)) * LSTM_TIME_STEP
    fut_t = (np.arange(t_plot + 1, t_plot + LSTM_FUTURE_STEPS + 1)) * LSTM_TIME_STEP

    for i in range(F):
        ax = axes[i, col]
        ax.plot(hist_t, pos[t_plot - LSTM_PAST_STEPS + 1: t_plot + 1, i],
                "k-", label="observed history")
        ax.plot(fut_t, truth[m, :, i], "g-", label="ground truth")
        ax.plot(fut_t, pred[m, :, i], "r--", label="prediction (LSTM)")
        ax.plot(fut_t, baseline[m, :, i], ":", color="gray",
                label="prediction (rigid-drogue)")
        ax.axvline(t_plot * LSTM_TIME_STEP, color="b", alpha=0.3)
        if col == 0:
            ax.set_ylabel(AXIS_NAMES[i] if i < len(AXIS_NAMES) else f"f{i}")
        if i == 0:
            ax.set_title(f"step {t_plot}\nt = {t_plot * LSTM_TIME_STEP:.2f} s")
        if i == F - 1:
            ax.set_xlabel("time (s)")
        ax.grid(alpha=0.3)

axes[0, 0].legend(loc="best")
fig.suptitle("Predictions at some time steps")
fig.tight_layout()

# ------------------------- plot 2: RMSE by future step -------------------------
fig2, ax2 = plt.subplots(figsize=(8, 4.5))
future_s = np.arange(1, LSTM_FUTURE_STEPS + 1) * LSTM_TIME_STEP
ax2.plot(future_s, rmse_total, "r-o", markersize=3, label="LSTM (3D)")
ax2.plot(future_s, rmse_baseline, "-s", color="gray", markersize=3, label="rigid-drogue (3D)")
for i in range(F):
    ax2.plot(future_s, rmse_axis[:, i], "--", alpha=0.6,
             label=f"LSTM ({AXIS_NAMES[i] if i < len(AXIS_NAMES) else i})")
ax2.set_xlabel("prediction horizon (s)")
ax2.set_ylabel("RMSE (m)")
ax2.set_title("Prediction error vs. prediction horizon")
ax2.grid(alpha=0.3)
ax2.legend()
fig2.tight_layout()

plt.show()