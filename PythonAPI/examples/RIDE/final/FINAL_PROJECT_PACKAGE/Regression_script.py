import json
import math
import random
from collections import defaultdict
from typing import Dict, Any

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# -------------------------
# Config
# -------------------------

INPUT_JSON = "complete_dataset.json"
BEST_MODEL_PATH = "best_model.pt"
STATS_PATH = "normalization_stats.json"

RANDOM_SEED = 42

TRAIN_POOL_RATIO = 0.9          # 90% train pool, 10% final test
VAL_FRACTION_IN_TRAIN = 0.11111 # fraction of train pool used as val each epoch

BATCH_SIZE = 64
MAX_EPOCHS = 200
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 10          # stop if val loss not improved for 5 epochs

# Numeric feature fields (inputs)
NUMERIC_FEATURES_INPUT = [
    "total_length",
    "avg_curvature",
    "max_curvature",
    "std_curvature",
    "tight_corners_pct",
    "straight_pct",
    "avg_slope",
    "max_slope",
    "elevation_range",
    "max_elevation",
    "min_elevation",
    "gear_ratio",
    "tire_friction",
]

# Target column
TARGET_COL = "avg_speed_ms"


# -------------------------
# Utility functions
# -------------------------

def compute_mean_std(rows, cols):
    """
    Compute mean and std for each column in cols over given rows.
    rows: list of dicts with numeric values for these cols.
    Returns: dict[col] = (mean, std)
    """
    stats = {}
    for col in cols:
        values = [r[col] for r in rows]
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(var)
        if std == 0.0:
            std = 1.0
        stats[col] = (mean, std)
    return stats


def normalize_rows(rows, stats, cols):
    """
    In-place normalization of rows for given cols using stats[col] = (mean, std).
    """
    for r in rows:
        for col in cols:
            mean, std = stats[col]
            r[col] = (r[col] - mean) / std


class CarlaDataset(Dataset):
    def __init__(self, rows, feature_cols, target_col):
        self.rows = rows
        self.feature_cols = feature_cols
        self.target_col = target_col

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        x = [r[c] for c in self.feature_cols]
        y = r[self.target_col]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def create_model(input_dim: int) -> nn.Module:
    """
    Simple MLP for regression.
    """
    model = nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )
    return model


def save_stats(
    stats: Dict[str, Any],
    numeric_features_input,
    target_col,
    numeric_to_normalize,
    feature_cols,
    route_types,
    path: str = STATS_PATH,
):
    """
    Save normalization stats + metadata to JSON so we can use them at inference.
    """
    out = {
        "numeric_features_input": numeric_features_input,
        "target_col": target_col,
        "numeric_to_normalize": numeric_to_normalize,
        "feature_cols": feature_cols,
        "route_types": route_types,
        "stats": {
            col: {"mean": stats[col][0], "std": stats[col][1]}
            for col in stats
        },
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved normalization stats + metadata to {path}")


def load_stats(path: str = STATS_PATH) -> Dict[str, Any]:
    with open(path, "r") as f:
        meta = json.load(f)
    return meta


# -------------------------
# Inference helper
# -------------------------

def infer(
    track_features: Dict[str, float],
    gear_ratio: float,
    tire_friction: float,
    route_type: str,
    model_path: str = BEST_MODEL_PATH,
    stats_path: str = STATS_PATH,
    device: str = "cpu",
) -> float:
    """
    Infer avg_speed_ms for a new (track, gear_ratio, tire_friction, route_type).
    """
    meta = load_stats(stats_path)

    numeric_features_input = meta["numeric_features_input"]
    target_col = meta["target_col"]
    feature_cols = meta["feature_cols"]
    route_types = meta["route_types"]
    stats = meta["stats"]  # col -> {"mean": ..., "std": ...}

    # Build row
    row = {}
    for col in numeric_features_input:
        if col == "gear_ratio":
            val = gear_ratio
        elif col == "tire_friction":
            val = tire_friction
        else:
            if col not in track_features:
                raise KeyError(f"Missing track feature '{col}' for inference.")
            val = track_features[col]
        row[col] = float(val)

    # Normalize numeric inputs
    for col in numeric_features_input:
        if col not in stats:
            continue
        mean = stats[col]["mean"]
        std = stats[col]["std"] if stats[col]["std"] != 0 else 1.0
        row[col] = (row[col] - mean) / std

    # One-hot route_type
    if route_type not in route_types:
        raise ValueError(f"Unknown route_type '{route_type}'. Known: {route_types}")
    for rt in route_types:
        row[f"rt_{rt}"] = 1.0 if route_type == rt else 0.0

    x_vec = [row[c] for c in feature_cols]
    x_tensor = torch.tensor(x_vec, dtype=torch.float32).unsqueeze(0)

    input_dim = len(feature_cols)
    model = create_model(input_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        x_tensor = x_tensor.to(device)
        y_norm = model(x_tensor).squeeze(0).item()

    # Denormalize target
    mean_y = stats[target_col]["mean"]
    std_y = stats[target_col]["std"] if stats[target_col]["std"] != 0 else 1.0
    y_real = y_norm * std_y + mean_y

    return y_real


# -------------------------
# Main training pipeline
# -------------------------

def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    # 1. Load JSON
    with open(INPUT_JSON, "r") as f:
        raw_data = json.load(f)

    # Discover route types dynamically
    route_types = sorted(set(item["route_type"] for item in raw_data))
    print("Found route_types:", route_types)
    one_hot_cols = [f"rt_{rt}" for rt in route_types]

    # 2. Build rows
    rows = []
    for item in raw_data:
        row = {}
        row["track_id"] = item["track_id"]
        for col in NUMERIC_FEATURES_INPUT:
            row[col] = float(item[col])
        row[TARGET_COL] = float(item[TARGET_COL])

        rt_val = item["route_type"]
        for rt in route_types:
            col_name = f"rt_{rt}"
            row[col_name] = 1.0 if rt_val == rt else 0.0
        row["route_type"] = rt_val
        rows.append(row)

    # 3. Stratified split at track level
    rt_to_tracks = defaultdict(set)
    for r in rows:
        rt_to_tracks[r["route_type"]].add(r["track_id"])

    for rt in rt_to_tracks:
        rt_to_tracks[rt] = sorted(rt_to_tracks[rt])

    track_split = {}  # track_id -> "train_pool" / "test"
    for rt, track_ids in rt_to_tracks.items():
        track_ids = list(track_ids)
        random.shuffle(track_ids)
        n = len(track_ids)
        n_train_pool = int(TRAIN_POOL_RATIO * n)
        train_pool_ids = track_ids[:n_train_pool]
        test_ids = track_ids[n_train_pool:]

        for tid in train_pool_ids:
            track_split[tid] = "train_pool"
        for tid in test_ids:
            track_split[tid] = "test"

        print(
            f"Route_type={rt}: tracks={n}, "
            f"train_pool={len(train_pool_ids)}, test={len(test_ids)}"
        )

    train_pool_rows, test_rows = [], []
    for r in rows:
        split = track_split[r["track_id"]]
        if split == "train_pool":
            train_pool_rows.append(r)
        else:
            test_rows.append(r)

    print(
        f"Total rows: {len(rows)} | "
        f"train_pool rows: {len(train_pool_rows)}, test rows: {len(test_rows)}"
    )

    # 4. Normalization stats from train_pool only
    numeric_to_normalize = NUMERIC_FEATURES_INPUT + [TARGET_COL]
    stats = compute_mean_std(train_pool_rows, numeric_to_normalize)

    normalize_rows(train_pool_rows, stats, numeric_to_normalize)
    normalize_rows(test_rows, stats, numeric_to_normalize)

    feature_cols = NUMERIC_FEATURES_INPUT + one_hot_cols

    save_stats(
        stats,
        numeric_features_input=NUMERIC_FEATURES_INPUT,
        target_col=TARGET_COL,
        numeric_to_normalize=numeric_to_normalize,
        feature_cols=feature_cols,
        route_types=route_types,
        path=STATS_PATH,
    )

    # Fixed test loader
    test_ds = CarlaDataset(test_rows, feature_cols, TARGET_COL)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 7. Model
    input_dim = len(feature_cols)
    model = create_model(input_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 8. Training with dynamic val + early stopping
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        random.shuffle(train_pool_rows)
        n_train_pool = len(train_pool_rows)
        n_val = max(1, int(VAL_FRACTION_IN_TRAIN * n_train_pool))
        n_train = n_train_pool - n_val

        current_train_rows = train_pool_rows[:n_train]
        current_val_rows = train_pool_rows[n_train:]

        train_ds = CarlaDataset(current_train_rows, feature_cols, TARGET_COL)
        val_ds = CarlaDataset(current_val_rows, feature_cols, TARGET_COL)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        # train
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(x_batch).squeeze(-1)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x_batch.size(0)
        train_loss /= len(train_ds)

        # val
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(x_batch).squeeze(-1)
                loss = criterion(preds, y_batch)
                val_loss += loss.item() * x_batch.size(0)
        val_loss /= len(val_ds)

        print(
            f"Epoch {epoch:03d} | "
            f"Train MSE: {train_loss:.4f} | "
            f"Val MSE (dynamic split): {val_loss:.4f}"
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"  -> New best model saved to {BEST_MODEL_PATH}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(no val improvement for {EARLY_STOP_PATIENCE} epochs)."
                )
                break

    # 9. Load best model & evaluate on train_pool + test, with R²
    best_model = create_model(len(feature_cols))
    best_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    best_model.to(device)
    best_model.eval()

    def eval_r2(dloader, n_samples):
        """Return (mse, r2) on normalized target."""
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for x_batch, y_batch in dloader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                preds = best_model(x_batch).squeeze(-1)
                all_preds.append(preds.cpu())
                all_targets.append(y_batch.cpu())
        y_pred = torch.cat(all_preds, dim=0)
        y_true = torch.cat(all_targets, dim=0)

        mse = torch.mean((y_true - y_pred) ** 2).item()
        y_mean = torch.mean(y_true).item()
        ss_res = torch.sum((y_true - y_pred) ** 2).item()
        ss_tot = torch.sum((y_true - y_mean) ** 2).item()
        r2 = float("nan") if ss_tot == 0 else 1.0 - ss_res / ss_tot
        return mse, r2

    # train_pool eval
    train_full_ds = CarlaDataset(train_pool_rows, feature_cols, TARGET_COL)
    train_full_loader = DataLoader(train_full_ds, batch_size=BATCH_SIZE, shuffle=False)
    train_mse, train_r2 = eval_r2(train_full_loader, len(train_full_ds))

    # test eval
    test_mse, test_r2 = eval_r2(test_loader, len(test_ds))

    print("\n=== Final Evaluation (on NORMALIZED avg_speed_ms) ===")
    print(f"Train pool: MSE = {train_mse:.4f}, R^2 = {train_r2:.4f}")
    print(f"Test set : MSE = {test_mse:.4f}, R^2 = {test_r2:.4f}")

    # Target normalization info
    mean_y, std_y = stats[TARGET_COL]
    print("\nNormalization stats for target (avg_speed_ms):")
    print(f"mean={mean_y:.6f}, std={std_y:.6f}")
    print("To denormalize: y_real = y_pred_norm * std + mean")


if __name__ == "__main__":
    main()
    # Example of how you might call infer() later (in a separate script or REPL):
    # track_feats = {
    #     "total_length": 880.0,
    #     "avg_curvature": 0.01,
    #     "max_curvature": 0.04,
    #     "std_curvature": 0.015,
    #     "tight_corners_pct": 0.2,
    #     "straight_pct": 0.6,
    #     "avg_slope": 0.0,
    #     "max_slope": 0.0,
    #     "elevation_range": 0.0,
    #     "max_elevation": 0.0,
    #     "min_elevation": 0.0,
    # }
    # pred_speed = infer(track_feats, gear_ratio=4.0, tire_friction=2.0,
    #                    route_type="straight_speed")
    # print("Predicted avg_speed_ms:", pred_speed)