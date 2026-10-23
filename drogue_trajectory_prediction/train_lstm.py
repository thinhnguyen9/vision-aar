import numpy as np
import importlib
import json
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from utils import rmse_per_horizon_step


# ----------------------------- config ----------------------------- #
CONFIG_JSON = "drogue_trajectory_prediction/20261023_01_config.json"
PATIENCE = 20        # stop if validation loss doesn't improve for this many epochs
SEED = 0
# ------------------------------------------------------------------ #

with open(CONFIG_JSON, "r") as f:
    config = json.load(f)
print(f"[train_lstm] config file: {CONFIG_JSON}")
print(f"[train_lstm] dataset: {config['lstm']['data_path']}")
print(f"[train_lstm] model: {config['lstm']['model']['module']}.{config['lstm']['model']['class']}")

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)


# ----------------------------- load data ----------------------------- #
d = np.load(config['lstm']['data_path'])
X_train, Y_train = d['X_train'].astype(np.float32), d['Y_train'].astype(np.float32)
X_val,   Y_val   = d['X_val'].astype(np.float32),   d['Y_val'].astype(np.float32)
X_test,  Y_test  = d['X_test'].astype(np.float32),  d['Y_test'].astype(np.float32)
mean, std = d['mean'].astype(np.float32), d['std'].astype(np.float32)

for name, arr in [("X_train", X_train), ("Y_train", Y_train),
                  ("X_val", X_val), ("Y_val", Y_val),
                  ("X_test", X_test), ("Y_test", Y_test)]:
    assert np.isfinite(arr).all(), f"{name} contains NaN or inf"

print(f"[train_lstm] trajectories: train={len(X_train)} val={len(X_val)} test={len(X_test)} | "
      f"in_horizon={config['lstm']['in_horizon']} out_horizon={config['lstm']['out_horizon']} "
      f"in_features={config['lstm']['in_features']} out_features={config['lstm']['out_features']}")

def make_loader(X, Y, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    return DataLoader(ds, batch_size=config['lstm']['training']['batch'], shuffle=shuffle)

train_loader = make_loader(X_train, Y_train, shuffle=True)   # shuffle only training
val_loader = make_loader(X_val, Y_val, shuffle=False)
# test_loader = make_loader(X_test, Y_test, shuffle=False)


# ----------------------------- train ----------------------------- #
module = importlib.import_module(config['lstm']['model']['module'])
model_class = getattr(module, config['lstm']['model']['class'])
model = model_class(
    in_features=config['lstm']['in_features'],
    out_features=config['lstm']['out_features'],
    out_horizon=config['lstm']['out_horizon'],
    lstm_hidden_size=config['lstm']['hidden_size']
).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config['lstm']['training']['lr'])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
print(f"[train_lstm] parameters: {sum(p.numel() for p in model.parameters()):,} | device={device}")


def run_epoch(loader, train):
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # guards against exploding gradients
                optimizer.step()
            total += loss.item() * len(xb)
            n += len(xb)
    return total / n


best_val, best_state, bad_epochs = float("inf"), None, 0
for epoch in range(1, config['lstm']['training']['epochs'] + 1):
    train_loss = run_epoch(train_loader, train=True)
    val_loss = run_epoch(val_loader, train=False)
    scheduler.step(val_loss)
    print(f"[train_lstm] epoch {epoch:3d} | train {train_loss:.5f} | val {val_loss:.5f} | "
          f"lr {optimizer.param_groups[0]['lr']:.1e}")

    if val_loss < best_val:
        best_val, bad_epochs = val_loss, 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print("[train_lstm] early stopping")
            break

model.load_state_dict(best_state)
# Save config so inference doesn't need to load config.json
torch.save({"state_dict": best_state, "mean": mean, "std": std, "config": config}, config['lstm']['model_path'])
print(f"[train_lstm] saved best model (val loss {best_val:.5f}) to {config['lstm']['model_path']}")


# ----------------------------- test ----------------------------- #
model.eval()
# preds = []
# with torch.no_grad():
#     for xb, _ in test_loader:   # xb: (batch, in_horizon, in_features), target: (batch, out_horizon, out_features)
#         preds.append(model(xb.to(device)).cpu().numpy())
# pred = np.concatenate(preds)    # (N, out_horizon, out_features), normalized units
preds_base, preds_lstm = [], []
with torch.no_grad():
    for x in X_test:    # x: (in_horizon, in_features)
        xb = torch.from_numpy(x[np.newaxis, ...]).to(device)   # xb: (batch=1, in_horizon, in_features)
        preds_base.append(np.tile(mean, (config['lstm']['out_horizon'], 1))[np.newaxis, ...]) # baseline: rigid drogue (predictions are just the mean)
        preds_lstm.append(model(xb).cpu().numpy())
preds_base = np.concatenate(preds_base)    # (N, out_horizon, out_features), normalized units
preds_lstm = np.concatenate(preds_lstm)    # (N, out_horizon, out_features), normalized units

# Convert normalized to original units
preds_base = preds_base * std + mean
preds_lstm = preds_lstm * std + mean
Y_test = Y_test * std + mean

# Compute RMSE per horizon step
rmse_lstm, rmse_lstm_total = rmse_per_horizon_step(preds_lstm, Y_test)
rmse_base, rmse_base_total = rmse_per_horizon_step(preds_base, Y_test)

print("\n[train_lstm] Test RMSE (same units as CSV):")
print("[train_lstm] RMSE (3D):")
print(f"[train_lstm] {'step':>5} {'LSTM':>15} {'rigid-drogue':>15}")
for k in sorted(set([1, max(1, config['lstm']['out_horizon'] // 4), max(1, config['lstm']['out_horizon'] // 2), config['lstm']['out_horizon']])):
    print(f"[train_lstm] {k:5d} {rmse_lstm_total[k - 1]:15.4f} {rmse_base_total[k - 1]:15.4f}")
print("[train_lstm] RMSE (per x/y/z):")
print(f"[train_lstm] {'step':>5} {'LSTM':>24} {'rigid-drogue':>24}")
for k in sorted(set([1, max(1, config['lstm']['out_horizon'] // 4), max(1, config['lstm']['out_horizon'] // 2), config['lstm']['out_horizon']])):
    print(f"[train_lstm] {k:5d} {rmse_lstm[k - 1, 0]:8.4f} {rmse_lstm[k - 1, 1]:8.4f} {rmse_lstm[k - 1, 2]:8.4f} "
          f"{rmse_base[k - 1, 0]:8.4f} {rmse_base[k - 1, 1]:8.4f} {rmse_base[k - 1, 2]:8.4f}")