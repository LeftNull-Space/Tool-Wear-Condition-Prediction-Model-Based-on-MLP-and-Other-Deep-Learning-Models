"""Simple stacking/averaging ensemble over trained regressors.
Loads trained weights (if available) for MLP, ResMLP, AttnMLP, CNN1D, LSTM and averages their predictions.
Uses same preprocessing: C1 80/20 split, PCA=15, y standardization.
"""
import os
import random
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# --------------------
# Config
# --------------------
DATA_DIR = "data"
WINDOW_SIZE = 1024
STEP = 512
CORR_THRESHOLD = 0.3
N_COMPONENTS = 15
BATCH_SIZE = 128
HIDDEN_SIZE = 192
DROPOUT = 0.20
CNN_C1 = 64
CNN_C2 = 128
EPOCHS = 2000  # not used (only inference)
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# checkpoint names
CKPTS = {
    "mlp": "best_mlp_regressor.pth",
    "resmlp": "best_resmlp_regressor.pth",
    "attn": "best_mlp_attn_regressor.pth",
    "cnn": "best_cnn_regressor.pth",
    "lstm": "best_lstm_regressor.pth",
}


# safe load to silence future warning (weights_only) with fallback
def safe_load(ckpt_path):
    try:
        return torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    except TypeError:
        return torch.load(ckpt_path, map_location=DEVICE)


# --------------------
# Utils
# --------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_signal_file(filepath: str) -> np.ndarray:
    data = np.loadtxt(filepath, delimiter=",")
    return data.reshape(-1, 7)


def time_domain_features(x: np.ndarray) -> list:
    mean = np.mean(x)
    std = np.std(x)
    pp = np.max(x) - np.min(x)
    rms = np.sqrt(np.mean(x ** 2))
    skew = np.mean(((x - mean) / (std + 1e-8)) ** 3)
    kurt = np.mean(((x - mean) / (std + 1e-8)) ** 4)
    return [mean, std, pp, rms, skew, kurt]


def freq_domain_features(x: np.ndarray, fs: int = 15385) -> list:
    n = len(x)
    X = np.fft.rfft(x) / n
    amp = np.abs(X)
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    energy = np.sum(amp ** 2)
    main_freq = freqs[np.argmax(amp[1:]) + 1] if len(amp) > 1 else 0.0
    total_amp = np.sum(amp)
    if total_amp > 0:
        mean_freq = np.sum(amp * freqs) / total_amp
        var_freq = np.sum(amp * (freqs - mean_freq) ** 2) / total_amp
    else:
        mean_freq = var_freq = 0.0
    return [energy, main_freq, mean_freq, var_freq]


def extract_features_from_signal(signal: np.ndarray, window_size: int = WINDOW_SIZE, step: int = STEP) -> np.ndarray:
    n = signal.shape[0]
    features = []
    for start in range(0, n - window_size + 1, step):
        seg = signal[start : start + window_size]
        seg_feat = []
        for ch in range(7):
            x = seg[:, ch]
            seg_feat.extend(time_domain_features(x))
            seg_feat.extend(freq_domain_features(x))
        features.append(seg_feat)
    if len(features) > 0:
        return np.mean(features, axis=0)
    return np.zeros(7 * (6 + 4))


def build_raw_dataset(data_folder: str, wear_file: str, max_files: int = 315) -> Tuple[np.ndarray, np.ndarray]:
    wear_df = pd.read_csv(os.path.join(DATA_DIR, wear_file))
    wear_df["VB"] = wear_df[["flute_1", "flute_2", "flute_3"]].max(axis=1)

    all_feats, all_labels = [], []
    for i in range(1, max_files + 1):
        file_path = os.path.join(DATA_DIR, data_folder, f"c_{data_folder[-1]}_{i:03d}.csv")
        if not os.path.exists(file_path):
            continue
        try:
            signal = load_signal_file(file_path)
            feats = extract_features_from_signal(signal)
            vb = wear_df.iloc[i - 1]["VB"]
            all_feats.append(feats)
            all_labels.append(vb)
        except Exception:
            continue

    X = np.array(all_feats)
    y = np.array(all_labels)
    return X, y


def select_features_by_correlation(X: np.ndarray, y: np.ndarray, threshold: float = CORR_THRESHOLD):
    selected = []
    for i in range(X.shape[1]):
        r, _ = pearsonr(X[:, i], y)
        if abs(r) >= threshold:
            selected.append(i)
    return X[:, selected], selected


def prepare_data():
    if os.path.exists("X_c1.npy") and os.path.exists("y_c1.npy"):
        X_c1 = np.load("X_c1.npy")
        y_c1 = np.load("y_c1.npy")
    else:
        X_c1, y_c1 = build_raw_dataset("c1", "c1_wear.csv")
        np.save("X_c1.npy", X_c1)
        np.save("y_c1.npy", y_c1)

    initial_vb = y_c1.min()
    y_relative_c1 = y_c1 - initial_vb

    X_c1_sel, selected_idx = select_features_by_correlation(X_c1, y_relative_c1, CORR_THRESHOLD)

    idx = np.arange(len(X_c1_sel))
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    split = int(0.8 * len(idx))
    train_idx, test_idx = idx[:split], idx[split:]

    X_train_raw = X_c1_sel[train_idx]
    y_train_raw = y_relative_c1[train_idx]
    X_test_raw = X_c1_sel[test_idx]
    y_test_raw = y_relative_c1[test_idx]

    y_scaler = StandardScaler()
    y_train_std = y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    y_test_std = y_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    pca = PCA(n_components=N_COMPONENTS)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    return {
        "X_train": X_train_pca,
        "y_train": y_train_std,
        "X_test": X_test_pca,
        "y_test": y_test_std,
        "scaler": scaler,
        "pca": pca,
        "selected_idx": selected_idx,
        "y_scaler": y_scaler,
    }


# --------------------
# Models (matching existing scripts)
# --------------------
class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = HIDDEN_SIZE, dropout: float = DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return x + self.block(x)


class ResMLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = HIDDEN_SIZE, num_blocks: int = 3, dropout: float = DROPOUT):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_size, dropout) for _ in range(num_blocks)])
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        h = self.stem(x)
        h = self.blocks(h)
        return self.head(h)


class AttnMLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = HIDDEN_SIZE, dropout: float = DROPOUT):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.Sigmoid(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        w = self.attn(x)
        return self.mlp(x * w)


class CNNRegressor(nn.Module):
    def __init__(self, input_len: int, c1: int = CNN_C1, c2: int = CNN_C2, dropout: float = DROPOUT):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(c2, c2 // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(c2 // 2, 1),
        )

    def forward(self, x):
        feats = self.conv(x)
        gap = feats.mean(dim=2)
        return self.head(gap)


class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = HIDDEN_SIZE, num_layers: int = 2, dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


# --------------------
# Inference helper
# --------------------
def load_and_predict(models: List[str], X: np.ndarray, y_scaler, seq_mode=False, input_len=None):
    preds_all = []
    for name in models:
        ckpt = CKPTS.get(name)
        if ckpt is None or (not os.path.exists(ckpt)):
            print(f"[Skip] {name} checkpoint not found: {ckpt}")
            continue

        if name == "mlp":
            net = MLPRegressor(input_dim=X.shape[1], hidden_size=HIDDEN_SIZE, dropout=DROPOUT)
            try:
                net.load_state_dict(safe_load(ckpt))
            except RuntimeError as e:
                print(f"[Skip] {name} load mismatch: {e}")
                continue
            net.to(DEVICE).eval()
            x_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                p = net(x_t).cpu().numpy().flatten()
        elif name == "resmlp":
            net = ResMLPRegressor(input_dim=X.shape[1], hidden_size=HIDDEN_SIZE, dropout=DROPOUT)
            try:
                net.load_state_dict(safe_load(ckpt))
            except RuntimeError as e:
                print(f"[Skip] {name} load mismatch: {e}")
                continue
            net.to(DEVICE).eval()
            x_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                p = net(x_t).cpu().numpy().flatten()
        elif name == "attn":
            net = AttnMLPRegressor(input_dim=X.shape[1], hidden_size=HIDDEN_SIZE, dropout=DROPOUT)
            try:
                net.load_state_dict(safe_load(ckpt))
            except RuntimeError as e:
                print(f"[Skip] {name} load mismatch: {e}")
                continue
            net.to(DEVICE).eval()
            x_t = torch.tensor(X, dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                p = net(x_t).cpu().numpy().flatten()
        elif name == "cnn":
            net = CNNRegressor(input_len=X.shape[1], c1=CNN_C1, c2=CNN_C2, dropout=DROPOUT)
            try:
                net.load_state_dict(safe_load(ckpt))
            except RuntimeError as e:
                print(f"[Skip] {name} load mismatch: {e}")
                continue
            net.to(DEVICE).eval()
            x_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(DEVICE)
            with torch.no_grad():
                p = net(x_t).cpu().numpy().flatten()
        elif name == "lstm":
            net = LSTMRegressor(input_dim=X.shape[1], hidden_size=HIDDEN_SIZE, dropout=DROPOUT)
            try:
                net.load_state_dict(safe_load(ckpt))
            except RuntimeError as e:
                print(f"[Skip] {name} load mismatch: {e}")
                continue
            net.to(DEVICE).eval()
            x_t = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(DEVICE)
            with torch.no_grad():
                p = net(x_t).cpu().numpy().flatten()
        else:
            continue

        # inverse transform to raw scale
        p_raw = y_scaler.inverse_transform(p.reshape(-1, 1)).flatten()
        preds_all.append(p_raw)
        print(f"[Use] {name} with ckpt {ckpt}, got {len(p_raw)} preds")

    if not preds_all:
        raise RuntimeError("No model predictions available. Check checkpoints.")

    # simple mean ensemble
    stacked = np.stack(preds_all, axis=0)  # [M, N]
    return stacked.mean(axis=0)


# --------------------
# Main
# --------------------
def main():
    set_seed(SEED)
    data = prepare_data()
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    y_scaler = data["y_scaler"]

    # Ensemble on test set (using trained checkpoints)
    models_to_use = ["mlp", "resmlp", "attn", "lstm"]  # cnn 效果较差，移除
    out_dir = "figs_ensemble"
    os.makedirs(out_dir, exist_ok=True)
    try:
        preds_test = load_and_predict(models_to_use, X_test, y_scaler)
    except Exception as e:
        print(f"Ensemble failed: {e}")
        return

    targets_raw = y_scaler.inverse_transform(np.array(y_test).reshape(-1, 1)).flatten()

    mae = mean_absolute_error(targets_raw, preds_test)
    rmse = np.sqrt(mean_squared_error(targets_raw, preds_test))
    r2 = r2_score(targets_raw, preds_test)

    print(f"\nEnsemble (mean) Test -> MAE: {mae:.3f} um | RMSE: {rmse:.3f} um | R2: {r2:.3f}")
    print(
        f"Pred mean: {np.mean(preds_test):.4f}, var: {np.var(preds_test):.4f} | "
        f"Target mean: {np.mean(targets_raw):.4f}, var: {np.var(targets_raw):.4f}"
    )
    print(
        f"Pred range: {np.min(preds_test):.4f} - {np.max(preds_test):.4f} | "
        f"Target range: {np.min(targets_raw):.4f} - {np.max(targets_raw):.4f}"
    )

    residuals = preds_test - targets_raw
    abs_err = np.abs(residuals)

    # 可视化：两个子图放在一张图里
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.scatter(preds_test, targets_raw, alpha=0.6)
    line_min = min(targets_raw.min(), preds_test.min())
    line_max = max(targets_raw.max(), preds_test.max())
    plt.plot([line_min, line_max], [line_min, line_max], "r--", label="Ideal")
    plt.xlabel("Predicted (ensemble)")
    plt.ylabel("True")
    plt.title(f"Pred vs True (Ensemble)\nR2={r2:.3f}, MAE={mae:.3f} um")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(targets_raw, abs_err, alpha=0.6, color="tab:orange")
    plt.xlabel("True")
    plt.ylabel("|Error|")
    plt.title("Absolute Error vs True")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_and_scatter.png"), dpi=200)
    plt.show()

    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.hist(residuals, bins=40, alpha=0.7, color="tab:blue")
    plt.axvline(residuals.mean(), color="r", linestyle="--", label=f"Mean={residuals.mean():.4f}")
    plt.title("Residuals (Pred - True)")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.scatter(targets_raw, abs_err, alpha=0.6, color="tab:orange")
    plt.xlabel("True")
    plt.ylabel("|Error|")
    plt.title("Absolute Error vs True")

    plt.subplot(1, 3, 3)
    plt.hist(targets_raw, bins=40, alpha=0.5, label="True", color="tab:green")
    plt.hist(preds_test, bins=40, alpha=0.5, label="Pred", color="tab:purple")
    plt.title("Distribution: True vs Pred")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "residuals_abs_err_dist.png"), dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
