import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import DrogueTrajectoryPredictor

# ------------------------- config -------------------------
DATA_PATH = "drogue_trajectory_prediction/drogue_trajectory_dataset.npz"
MODEL_PATH = "drogue_trajectory_prediction/drogue_trajectory_lstm.pt"
HIDDEN = 256          # size of the LSTM memory vector
LAYERS = 2           # stacked LSTM layers
DROPOUT = 0.1        # regularization between stacked layers
RESIDUAL = True      # predict change from the last observed step (usually trains better)
BATCH = 16
EPOCHS = 100
LR = 1e-3
PATIENCE = 50        # stop if validation loss doesn't improve for this many epochs
SEED = 0

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED)
np.random.seed(SEED)

# ------------------------- data -------------------------
d = np.load(DATA_PATH)
X_train, Y_train = d["X_train"].astype(np.float32), d["Y_train"].astype(np.float32)
X_val, Y_val = d["X_val"].astype(np.float32), d["Y_val"].astype(np.float32)
X_test, Y_test = d["X_test"].astype(np.float32), d["Y_test"].astype(np.float32)
mean, std = d["mean"].astype(np.float32), d["std"].astype(np.float32)

for name, arr in [("X_train", X_train), ("Y_train", Y_train), ("X_val", X_val),
                  ("Y_val", Y_val), ("X_test", X_test), ("Y_test", Y_test)]:
    assert np.isfinite(arr).all(), f"{name} contains NaN or inf"

_, T, IN_DIM = X_train.shape
_, K, OUT_DIM = Y_train.shape
print(f"windows: train={len(X_train)} val={len(X_val)} test={len(X_test)} | "
      f"T={T} K={K} in_dim={IN_DIM} out_dim={OUT_DIM} | device={device}")


def make_loader(X, Y, shuffle):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Y))
    return DataLoader(ds, batch_size=BATCH, shuffle=shuffle)


train_loader = make_loader(X_train, Y_train, shuffle=True)   # shuffle only training
val_loader = make_loader(X_val, Y_val, shuffle=False)
test_loader = make_loader(X_test, Y_test, shuffle=False)

model = DrogueTrajectoryPredictor(IN_DIM, OUT_DIM, K, HIDDEN, LAYERS, DROPOUT, RESIDUAL).to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3)
print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")


# ------------------------- training -------------------------
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
for epoch in range(1, EPOCHS + 1):
    train_loss = run_epoch(train_loader, train=True)
    val_loss = run_epoch(val_loader, train=False)
    scheduler.step(val_loss)
    print(f"epoch {epoch:3d} | train {train_loss:.5f} | val {val_loss:.5f} | "
          f"lr {optimizer.param_groups[0]['lr']:.1e}")

    if val_loss < best_val:
        best_val, bad_epochs = val_loss, 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    else:
        bad_epochs += 1
        if bad_epochs >= PATIENCE:
            print("early stopping")
            break

model.load_state_dict(best_state)
torch.save({"state_dict": best_state,
            "config": dict(in_dim=IN_DIM, out_dim=OUT_DIM, horizon=K, hidden=HIDDEN,
                           layers=LAYERS, dropout=DROPOUT, residual=RESIDUAL),
            "mean": mean, "std": std}, MODEL_PATH)
print(f"saved best model (val loss {best_val:.5f}) to {MODEL_PATH}")


# ------------------------- test evaluation -------------------------
model.eval()
preds = []
with torch.no_grad():
    for xb, _ in test_loader:
        preds.append(model(xb.to(device)).cpu().numpy())
pred = np.concatenate(preds)                                   # (N, K, out_dim), normalized units

# baseline: rigid drogue (predictions are just the mean)
# (linear extrapolation is unchanged by normalization, so it can be done here)
rigid_pred = np.tile(mean, (K, 1))
baseline = np.tile(rigid_pred, (len(X_test), 1, 1))


def rmse_per_horizon(p, y):
    err = (p - y) * std[:OUT_DIM]                              # convert to physical units
    return np.sqrt((err ** 2).mean(axis=(0, 2)))               # (K,)


rmse_lstm = rmse_per_horizon(pred, Y_test)
rmse_baseline = rmse_per_horizon(baseline, Y_test)

print("\nTest RMSE in physical units (same units as CSV):")
print(f"{'step':>5} {'LSTM':>10} {'rigid-drogue':>10}")
for k in sorted(set([1, max(1, K // 4), max(1, K // 2), K])):
    print(f"{k:5d} {rmse_lstm[k - 1]:10.4f} {rmse_baseline[k - 1]:10.4f}")