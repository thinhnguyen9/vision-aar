import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

# ------------------------- config -------------------------
LOG_LIST = "drogue_trajectory_prediction/drogue_trajectory_logs.json"
CSV_TIME_COL = "t"
CSV_DROGUE_COLS = ["sway_dx_m", "sway_dy_m", "sway_dz_m"]
LSTM_TIME_STEP = 0.5
LSTM_PAST_STEPS = 20
LSTM_FUTURE_STEPS = 10
LSTM_DATASET_PATH = "drogue_trajectory_prediction/drogue_trajectory_dataset.npz"


def resample_log(csv_path, dt=LSTM_TIME_STEP, time_col=CSV_TIME_COL, cols=CSV_DROGUE_COLS, max_time_gap=2*LSTM_TIME_STEP):
    """
    Csv logs doesn't have a fixed time step. Resample with linear interpolation for a uniform time step.
    """
    df = pd.read_csv(csv_path).sort_values(time_col)
    df = df.drop_duplicates(subset=time_col)       # interpolation needs strictly increasing t
    t = df[time_col].to_numpy()
    if np.any(np.diff(t) > max_time_gap):
        print(f"[resample_log][Warning] Large time gap ({np.max(np.diff(t)):.4f} s) detected in {csv_path}!")
    t_new = np.arange(t[0], t[-1], dt)             # uniform grid
    out = {time_col: t_new}
    for c in cols:
        out[c] = np.interp(t_new, t, df[c].to_numpy())   # linear interpolation
    # plt.plot(t, df[cols[0]], 'o', label='original')
    # plt.plot(t_new, out[cols[0]], '-x', label='resampled')
    # plt.legend()
    # plt.show()
    return pd.DataFrame(out)


def make_windows(data, T=LSTM_PAST_STEPS, K=LSTM_FUTURE_STEPS, stride=1):
    """
    Dimensions:
        data: (N, F) resampled log, N time steps, F=3 features (x/y/z)
        X: (num_windows, T, F) input sequences for LSTM
        Y: (num_windows, K, F) output sequences for LSTM
    """
    X, Y = [], []
    for i in range(0, len(data) - T - K + 1, stride):
        X.append(data[i : i+T])
        Y.append(data[i+T : i+T+K])
    return np.stack(X), np.stack(Y)


def main():
    
    # Load list of csv logs
    with open(LOG_LIST, "r") as f:
        logs = json.load(f)

    # Make dataset
    drogue_dataset = {}
    for split in ["train", "test", "val"]:
        raw_data = np.empty((0, len(CSV_DROGUE_COLS)))
        X = np.empty((0, LSTM_PAST_STEPS, len(CSV_DROGUE_COLS)))
        Y = np.empty((0, LSTM_FUTURE_STEPS, len(CSV_DROGUE_COLS)))
        for csv_path in logs[split]:
            resampled_df = resample_log(csv_path)
            resampled_data = resampled_df[CSV_DROGUE_COLS].to_numpy()     # time steps x features (x/y/z)
            raw_data = np.concatenate((raw_data, resampled_data), axis=0)
            X_per_log, Y_per_log = make_windows(resampled_data)
            X = np.concatenate((X, X_per_log), axis=0)
            Y = np.concatenate((Y, Y_per_log), axis=0)

        # Normalize using training mean/stdev
        if split == "train":
            train_mean = np.mean(raw_data, axis=0)
            train_std = np.std(raw_data, axis=0)
        drogue_dataset[split] = {
            "X": (X - train_mean) / train_std,
            "Y": (Y - train_mean) / train_std
        }
        print(f"[make_drogue_trajectory_dataset] Processed {split} split: X shape = {X.shape}, Y shape = {Y.shape}.")

    # Save dataset to .npz
    np.savez(
        LSTM_DATASET_PATH,
        X_train=drogue_dataset["train"]["X"], Y_train=drogue_dataset["train"]["Y"],
        X_val=drogue_dataset["val"]["X"],     Y_val=drogue_dataset["val"]["Y"],
        X_test=drogue_dataset["test"]["X"],   Y_test=drogue_dataset["test"]["Y"],
        mean=train_mean, std=train_std,      # normalization stats, shape (F,)
        dt=LSTM_TIME_STEP,                   # resample period
    )
    print(f"[make_drogue_trajectory_dataset] Saved dataset to {LSTM_DATASET_PATH}.")


if __name__ == "__main__":
    main()