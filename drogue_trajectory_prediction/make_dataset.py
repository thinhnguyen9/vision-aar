import numpy as np
import json
from utils import resample_csv_log, make_data_windows

# ----------------------------- config ----------------------------- #
CONFIG_JSON = "drogue_trajectory_prediction/20261023_01_config.json"
# ------------------------------------------------------------------ #


def main():

    # Load configs
    print(f"[make_dataset] Using config: {CONFIG_JSON}")
    with open(CONFIG_JSON, "r") as f:
        config = json.load(f)

    # Make dataset
    drogue_dataset = {}
    for split in config['csv_log_path'].keys():     # 'train', 'val', 'test'
        raw_data = np.empty((0, config['lstm']['in_features']))  # x, y, z
        X = np.empty((0, config['lstm']['in_horizon'], config['lstm']['in_features']))
        Y = np.empty((0, config['lstm']['out_horizon'], config['lstm']['out_features']))
        for csv_path in config['csv_log_path'][split]:
            time_segments, resampled_segments = resample_csv_log(
                csv_path=csv_path,
                dt=config['time_step'],
                time_col=config['csv_col_names']['time'],
                data_cols=config['csv_col_names']['drogue_pos'],
                max_time_gap=3*config['time_step']
            )
            for resampled_data in resampled_segments:
                if len(resampled_data) < 1:
                    continue
                raw_data = np.concatenate((raw_data, resampled_data), axis=0)
                X_per_log, Y_per_log = make_data_windows(
                    time_series=resampled_data,
                    Nin=config['lstm']['in_horizon'],
                    Nout=config['lstm']['out_horizon'],
                    sampling_interval=config['sampling_interval']
                )
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
        print(f"[make_dataset] Processed {split} split: X shape = {X.shape}, Y shape = {Y.shape}.")

    # Save dataset to .npz
    np.savez(
        config['lstm']['data_path'],
        X_train=drogue_dataset['train']['X'], Y_train=drogue_dataset['train']['Y'],
        X_val=drogue_dataset['val']['X'],     Y_val=drogue_dataset['val']['Y'],
        X_test=drogue_dataset['test']['X'],   Y_test=drogue_dataset['test']['Y'],
        mean=train_mean, std=train_std,      # normalization stats, shape (F,)
        dt=config['time_step'],              # resample period
    )
    print(f"[make_dataset] Saved dataset to {config['lstm']['data_path']}.")


if __name__ == "__main__":
    main()