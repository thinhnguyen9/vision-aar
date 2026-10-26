import numpy as np
import importlib
import json
from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from utils import resample_csv_log, make_data_windows, rmse_per_horizon_step, DrogueTrajectoryPredictor


# ----------------------------- config ----------------------------- #
CONFIG_JSON = "drogue_trajectory_prediction/20261023_01_config.json"
CSV_LOG_PATH = "logs/lqr_gt_1789769555.csv"
# ------------------------------------------------------------------ #

with open(CONFIG_JSON, "r") as f:
    config = json.load(f)
AXIS_NAMES = ['x', 'y', 'z']
CONFIG_NAME = Path(CONFIG_JSON).stem[:11]   # config file version (date & number)
CSV_LOG_ID = Path(CSV_LOG_PATH).stem[-10:]  # UNIX timestamp of log
PREDICTIONS_FIGURE_PATH = f"drogue_trajectory_prediction/images/{CONFIG_NAME}_{CSV_LOG_ID}_predictions.png"
RMSE_FIGURE_PATH = f"drogue_trajectory_prediction/images/{CONFIG_NAME}_{CSV_LOG_ID}_rmse.png"
HISTORY_FIGURE_PATH = f"drogue_trajectory_prediction/images/{CONFIG_NAME}_{CSV_LOG_ID}_pred_history.png"
PLOT_PREDICTION_EVERY = config['lstm']['out_horizon']

print(f"[test_lstm] config file: {CONFIG_JSON}")
print(f"[test_lstm] model: {config['lstm']['model_path']}")
print(f"[test_lstm] CSV log to test: {CSV_LOG_PATH}")

# ------------------------- load model ------------------------- #
predictor_lstm = DrogueTrajectoryPredictor(
    model_path=config['lstm']['model_path'],
    method='lstm'
)
predictor_rigid = DrogueTrajectoryPredictor(
    model_path=config['lstm']['model_path'],
    method='rigid-drogue'
)
predictor_constvel = DrogueTrajectoryPredictor(
    model_path=config['lstm']['model_path'],
    method='const-vel'
)


# ------------------------- load CSV log ------------------------- #
time_segments, resampled_segments = resample_csv_log(
    csv_path=CSV_LOG_PATH,
    dt=config['time_step'],
    time_col=config['csv_col_names']['time'],
    data_cols=config['csv_col_names']['drogue_pos'],
    max_time_gap=3*config['time_step']
)
tvec = time_segments[0] - time_segments[0][0]   # (N,), starts at 0
pvec = resampled_segments[0]                    # (N, 3)
N, F = pvec.shape
assert F == config['lstm']['in_features'], f"run has {F} features but the model expects {config['lstm']['in_features']}"
assert N >= config['lstm']['in_horizon'] + config['lstm']['out_horizon'], \
       f"run is too short: need at least Nin+Nout = {config['lstm']['in_horizon'] + config['lstm']['out_horizon']} steps, got {N}"


# ------------------------- predict at each time step ------------------------- #
t_idx = np.arange(config['lstm']['in_horizon'] - 1, N - config['lstm']['out_horizon'])
N_pred = len(t_idx)
groundtruths, preds_rigid, preds_constvel, preds_lstm = [], [], [], []
for k in t_idx:
    x = pvec[k - config['lstm']['in_horizon'] + 1: k + 1]
    y = pvec[k + 1: k + 1 + config['lstm']['out_horizon']]
    preds_rigid.append(predictor_rigid.predict(x)[np.newaxis, ...])
    preds_constvel.append(predictor_constvel.predict(x)[np.newaxis, ...])
    preds_lstm.append(predictor_lstm.predict(x)[np.newaxis, ...])
    groundtruths.append(y[np.newaxis, ...])

preds_rigid = np.concatenate(preds_rigid)           # (N_pred, out_horizon, out_features)
preds_constvel = np.concatenate(preds_constvel)     # (N_pred, out_horizon, out_features)
preds_lstm = np.concatenate(preds_lstm)             # (N_pred, out_horizon, out_features)
groundtruths = np.concatenate(groundtruths)         # (N_pred, out_horizon, out_features)

# ------------------------- errors ------------------------- #
rmse_rigid,    rmse3d_rigid    = rmse_per_horizon_step(preds_rigid,    groundtruths)
rmse_constvel, rmse3d_constvel = rmse_per_horizon_step(preds_constvel, groundtruths)
rmse_lstm,     rmse3d_lstm     = rmse_per_horizon_step(preds_lstm,     groundtruths)

print(f"[test_lstm] evaluated {N_pred} time steps")
print(f"[test_lstm] RMSE (3D, meters):")
print(f"[test_lstm] {'step':>5} {'LSTM':>15} {'const-vel':>15}  {'rigid-drogue':>15}")
for k in sorted(set([1, max(1, config['lstm']['out_horizon'] // 4), max(1, config['lstm']['out_horizon'] // 2), config['lstm']['out_horizon']])):
    print(f"[test_lstm] {k:5d} {rmse3d_lstm[k - 1]:15.4f} {rmse3d_constvel[k - 1]:15.4f}  {rmse3d_rigid[k - 1]:15.4f}")


# ------------------------- fig 1: trajectory and predictions ------------------------- #
fig, axes = plt.subplots(F, 1, figsize=(11, 3 * F), sharex=True, squeeze=False)

for i, ax in enumerate(axes[:, 0]):
    ax.plot(tvec, pvec[:, i], "k-", linewidth=2, label="ground truth")
    for j in range(t_idx[0], t_idx[-1] + 1, PLOT_PREDICTION_EVERY):
        ax.axvline(tvec[j], color='gray', linestyle=':', linewidth=0.8)
        # predictions are j+1,...,j+config['lstm']['out_horizon']
        ax.plot(
            tvec[j+1 : j+config['lstm']['out_horizon']+1],
            preds_rigid[j-t_idx[0], :, i],
            "g:", linewidth=1.2, label='rigid-drogue'
        )
        ax.plot(
            tvec[j+1 : j+config['lstm']['out_horizon']+1],
            preds_constvel[j-t_idx[0], :, i],
            "b-.", linewidth=1.2, label='const-vel'
        )
        ax.plot(
            tvec[j+1 : j+config['lstm']['out_horizon']+1],
            preds_lstm[j-t_idx[0], :, i],
            "r--", linewidth=1.2, label='LSTM'
        )
        ax.plot(tvec[j+1], preds_rigid[j-t_idx[0], 0, i], 'gx', markersize=5)
        ax.plot(tvec[j+1], preds_constvel[j-t_idx[0], 0, i], 'b+', markersize=5)
        ax.plot(tvec[j+1], preds_lstm[j-t_idx[0], 0, i], 'ro', markersize=5)
        if i==0 and j==t_idx[0]:
            ax.legend(loc="best")
    ax.set_ylabel(f"{AXIS_NAMES[i] if i < len(AXIS_NAMES) else f'f{i}'} (m)")
    ax.grid(alpha=0.3)

# axes[0, 0].legend(loc="best")
axes[-1, 0].set_xlabel("Time (s)")
fig.suptitle(f"Ground truth vs. predicted drogue trajectory: {CSV_LOG_PATH}")
fig.tight_layout()
fig.savefig(PREDICTIONS_FIGURE_PATH, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[test_lstm] saved figure {PREDICTIONS_FIGURE_PATH}")


# ------------------------- fig 2: RMSE by future step ------------------------- #
fig2, axes2 = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
future_s = np.arange(1, config['lstm']['out_horizon'] + 1) * config['time_step']
model_curves = (
    (rmse_rigid, "rigid-drogue", "green", "-x"),
    (rmse_constvel, "const-vel", "blue", "-+"),
    (rmse_lstm, "LSTM", "red", "-o"),
)
for i, ax2 in enumerate(axes2[:3]):
    for rmse, label, color, style in model_curves:
        ax2.plot(future_s, rmse[:, i], style, color=color, markersize=5, label=label)
    axis_name = AXIS_NAMES[i] if i < len(AXIS_NAMES) else f"feature {i}"
    ax2.set_ylabel(f"{axis_name} RMSE (m)")
    ax2.grid(alpha=0.3)
axes2[0].legend(loc="best")

ax2 = axes2[3]
for rmse, label, color, style in (
    (rmse3d_rigid, "rigid-drogue", "green", "-x"),
    (rmse3d_constvel, "const-vel", "blue", "-+"),
    (rmse3d_lstm, "LSTM", "red", "-o"),
):
    ax2.plot(future_s, rmse, style, color=color, markersize=5, label=label)
ax2.grid(alpha=0.3)
ax2.set_xlabel("prediction horizon (s)")
ax2.set_ylabel("3D RMSE (m)")
fig2.suptitle(f"RMSE per future step: {CSV_LOG_PATH}")
fig2.tight_layout()
fig2.savefig(RMSE_FIGURE_PATH, dpi=150, bbox_inches="tight")
plt.close(fig2)
print(f"[test_lstm] saved figure {RMSE_FIGURE_PATH}")


# --------------------- fig 3: same predictions at different times --------------------- #
curr_truths, hist_rigid, hist_constvel, hist_lstm = [], [], [], []
for k in range(t_idx[0], t_idx[-1] + 1 - config['lstm']['out_horizon']):
    # ground truths : P(k+10)
    # predictions   : p(k+10|k), ..., p(k+10|k+9)
    curr_truths.append(np.tile(
        pvec[k + config['lstm']['out_horizon']],
        (config['lstm']['out_horizon'], 1)
    )[np.newaxis, ...])
    curr_rigid, curr_constvel, curr_lstm = [], [], []
    for i in range(config['lstm']['out_horizon']):
        curr_rigid.append(preds_rigid[k - t_idx[0] + i, -i-1, :])
        curr_constvel.append(preds_constvel[k - t_idx[0] + i, -i-1, :])
        curr_lstm.append(preds_lstm[k - t_idx[0] + i, -i-1, :])
    hist_rigid.append(np.array(curr_rigid)[np.newaxis, ...])
    hist_constvel.append(np.array(curr_constvel)[np.newaxis, ...])
    hist_lstm.append(np.array(curr_lstm)[np.newaxis, ...])
curr_truths = np.concatenate(curr_truths)       # (N_pred - out_horizon, out_horizon, out_features)
hist_rigid = np.concatenate(hist_rigid)         # (N_pred - out_horizon, out_horizon, out_features)
hist_constvel = np.concatenate(hist_constvel)   # (N_pred - out_horizon, out_horizon, out_features)
hist_lstm = np.concatenate(hist_lstm)           # (N_pred - out_horizon, out_horizon, out_features)

fig3, axes3 = plt.subplots(F, 1, figsize=(11, 3 * F), sharex=True, squeeze=False)
for i, ax in enumerate(axes3[:, 0]):
    for j in range(t_idx[0], t_idx[-1] + 1 - config['lstm']['out_horizon'], PLOT_PREDICTION_EVERY):
        ax.axvline(tvec[j + config['lstm']['out_horizon']], color='gray', linestyle=':', linewidth=0.8)
        ax.plot(
            tvec[j : j + config['lstm']['out_horizon']],
            curr_truths[j-t_idx[0], :, i],
            "k-", linewidth=2
        )
        ax.plot(
            tvec[j + config['lstm']['out_horizon']],
            curr_truths[j - t_idx[0], 0, i],
            'ko', markersize=5, label='ground truth(T)'
        )
        # predictions are p(j+10|j), ..., p(j+10|j+9)
        ax.plot(
            tvec[j : j + config['lstm']['out_horizon']],
            hist_rigid[j-t_idx[0], :, i],
            "g:", linewidth=1.2, label='rigid-drogue(T|T-k)'
        )
        ax.plot(
            tvec[j : j + config['lstm']['out_horizon']],
            hist_constvel[j-t_idx[0], :, i],
            "b-.", linewidth=1.2, label='const-vel(T|T-k)'
        )
        ax.plot(
            tvec[j : j + config['lstm']['out_horizon']],
            hist_lstm[j-t_idx[0], :, i],
            "r--", linewidth=1.2, label='LSTM(T|T-k)'
        )
        if i==0 and j==t_idx[0]:
            ax.legend(loc="best")
    ax.set_ylabel(f"{AXIS_NAMES[i] if i < len(AXIS_NAMES) else f'f{i}'} (m)")
    ax.grid(alpha=0.3)

# axes3[0, 0].legend(loc="best")
axes3[-1, 0].set_xlabel("Time (s)")
fig3.suptitle(f"Prediction history: {CSV_LOG_PATH}")
fig3.tight_layout()
fig3.savefig(HISTORY_FIGURE_PATH, dpi=150, bbox_inches="tight")
plt.close(fig3)
print(f"[test_lstm] saved figure {HISTORY_FIGURE_PATH}")
