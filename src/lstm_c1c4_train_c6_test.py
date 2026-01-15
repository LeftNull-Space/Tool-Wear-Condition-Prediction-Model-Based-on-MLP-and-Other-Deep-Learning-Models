"""LSTM regressor: train on C1+C4, test on C6.
Preprocessing: feature extraction cache, correlation filter, standardization, PCA=15, y standardization.
Notes: treats PCA features as length-1 sequence for simplicity.
"""
import os
import random
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
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
NUM_LAYERS = 2
EPOCHS = 2000
LR = 7e-5
WEIGHT_DECAY = 5e-5
PATIENCE = 200
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    if os.path.exists("X_c4.npy") and os.path.exists("y_c4.npy"):
        X_c4 = np.load("X_c4.npy")
        y_c4 = np.load("y_c4.npy")
    else:
        X_c4, y_c4 = build_raw_dataset("c4", "c4_wear.csv")
        np.save("X_c4.npy", X_c4)
        np.save("y_c4.npy", y_c4)

    if os.path.exists("X_c6.npy") and os.path.exists("y_c6.npy"):
        X_c6 = np.load("X_c6.npy")
        y_c6 = np.load("y_c6.npy")
    else:
        X_c6, y_c6 = build_raw_dataset("c6", "c6_wear.csv")
        np.save("X_c6.npy", X_c6)
        np.save("y_c6.npy", y_c6)

    initial_vb = min(y_c1.min(), y_c4.min(), y_c6.min())
    y_rel_c1 = y_c1 - initial_vb
    y_rel_c4 = y_c4 - initial_vb
    y_rel_c6 = y_c6 - initial_vb

    X_train_raw = np.vstack([X_c1, X_c4])
    y_train_raw = np.hstack([y_rel_c1, y_rel_c4])
    X_test_raw = X_c6
    y_test_raw = y_rel_c6

    X_train_sel, selected_idx = select_features_by_correlation(X_train_raw, y_train_raw, CORR_THRESHOLD)
    X_test_sel = X_test_raw[:, selected_idx]

    y_scaler = StandardScaler()
    y_train_std = y_scaler.fit_transform(y_train_raw.reshape(-1, 1)).flatten()
    y_test_std = y_scaler.transform(y_test_raw.reshape(-1, 1)).flatten()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)

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
# Model: LSTM
# --------------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = HIDDEN_SIZE, num_layers: int = NUM_LAYERS, dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, input_dim) -> treat as sequence len=1
        seq = x.unsqueeze(1)
        out, _ = self.lstm(seq)
        last = out[:, -1, :]
        return self.head(last)


# --------------------
# Training
# --------------------
def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, y_scaler):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    best_mae = float("inf")
    no_improve = 0
    train_losses, val_maes = [], []
    best_preds = None
    best_targets = None

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred.squeeze(), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                pred = model(x)
                preds.extend(pred.cpu().numpy().flatten())
                targets.extend(y.cpu().numpy())

        targets_raw = y_scaler.inverse_transform(np.array(targets).reshape(-1, 1)).flatten()
        preds_raw = y_scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()

        mae = mean_absolute_error(targets_raw, preds_raw)
        rmse = np.sqrt(mean_squared_error(targets_raw, preds_raw))
        r2 = r2_score(targets_raw, preds_raw)
        val_maes.append(mae)

        scheduler.step(mae)

        if mae < best_mae:
            best_mae = mae
            no_improve = 0
            best_preds = preds_raw.copy()
            best_targets = targets_raw.copy()
            torch.save(model.state_dict(), "best_lstm_c1c4_to_c6.pth")
        else:
            no_improve += 1

        if epoch % 20 == 0 or epoch == EPOCHS - 1:
            print(
                f"Epoch {epoch + 1:3d} | Train Loss: {train_losses[-1]:.4f} | "
                f"Val MAE: {mae:.3f} um | R2: {r2:.3f}"
            )

        if no_improve >= PATIENCE:
            print(f"Early stop at epoch {epoch + 1}. Best MAE: {best_mae:.3f} um")
            break

    return {
        "best_mae": best_mae,
        "train_losses": train_losses,
        "val_maes": val_maes,
        "best_preds": best_preds,
        "best_targets": best_targets,
    }


# --------------------
# Main
# --------------------
def main():
    set_seed(SEED)
    print(f"Using device: {DEVICE}")

    out_dir = "figs_lstm_c1c4_c6"
    os.makedirs(out_dir, exist_ok=True)

    data = prepare_data()
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    y_scaler = data["y_scaler"]

    input_dim = X_train.shape[1]
    model = LSTMRegressor(input_dim=input_dim, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    results = train_model(model, train_loader, test_loader, y_scaler)

    mae = results["best_mae"]
    preds = results["best_preds"]
    targets = results["best_targets"]
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2 = r2_score(targets, preds)

    print(
        f"\nTest on C6 -> MAE: {mae:.3f} um | RMSE: {rmse:.3f} um | R2: {r2:.3f}"
    )
    print(
        f"Pred mean: {np.mean(preds):.4f}, var: {np.var(preds):.4f} | "
        f"Target mean: {np.mean(targets):.4f}, var: {np.var(targets):.4f}"
    )
    print(
        f"Pred range: {np.min(preds):.4f} - {np.max(preds):.4f} | "
        f"Target range: {np.min(targets):.4f} - {np.max(targets):.4f}"
    )

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(results["train_losses"], label="Train Loss")
    plt.plot(results["val_maes"], label="Val MAE")
    plt.xlabel("Epoch")
    plt.title("Training Curve (LSTM C1+C4 train)")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(preds, targets, alpha=0.6)
    line_min = min(targets.min(), preds.min())
    line_max = max(targets.max(), preds.max())
    plt.plot([line_min, line_max], [line_min, line_max], "r--", label="Ideal")
    plt.xlabel("Predicted VB on C6 (raw)")
    plt.ylabel("True VB on C6 (raw)")
    plt.title(f"Pred vs True (R2={r2:.3f}, MAE={mae:.3f} um)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "training_and_scatter.png"), dpi=200)
    plt.show()

    residuals = preds - targets
    abs_err = np.abs(residuals)

    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.hist(residuals, bins=40, alpha=0.7, color="tab:blue")
    plt.axvline(residuals.mean(), color="r", linestyle="--", label=f"Mean={residuals.mean():.4f}")
    plt.title("Residuals (Pred - True)")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.scatter(targets, abs_err, alpha=0.6, color="tab:orange")
    plt.xlabel("True VB on C6 (raw)")
    plt.ylabel("|Error|")
    plt.title("Absolute Error vs True")

    plt.subplot(1, 3, 3)
    plt.hist(targets, bins=40, alpha=0.5, label="True", color="tab:green")
    plt.hist(preds, bins=40, alpha=0.5, label="Pred", color="tab:purple")
    plt.title("Distribution: True vs Pred (C6)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "residuals_abs_err_dist.png"), dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
