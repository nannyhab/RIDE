import json
import csv
import math

INPUT_JSON = "complete_dataset.json"
RAW_CSV = "dataset_raw.csv"
NORM_CSV = "dataset_normalized.csv"

# The fields you specified
NUMERIC_FIELDS = [
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
    "avg_speed_ms",  # target
]

# Exact route_type values we found in the dataset
ROUTE_TYPES = [
    "elevation",
    "mixed",
    "straight_speed",
    "tight_curves",
    "tire_friction_test",
]

ONE_HOT_COLS = [f"rt_{rt}" for rt in ROUTE_TYPES]


def compute_mean_std(rows, numeric_cols):
    """
    Compute mean and std for each numeric column over the given rows.
    rows: list of dicts, each dict has all numeric fields + one-hot
    numeric_cols: list of column names to normalize
    Returns: dict col -> (mean, std)
    """
    stats = {}
    for col in numeric_cols:
        values = [row[col] for row in rows]
        mean = sum(values) / len(values)
        # compute variance
        var = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(var)
        # avoid division by zero
        if std == 0:
            std = 1.0
        stats[col] = (mean, std)
    return stats


def main():
    # 1. Load JSON
    with open(INPUT_JSON, "r") as f:
        data = json.load(f)

    # 2. Build raw rows (numeric + one-hot)
    rows = []
    for item in data:
        row = {}

        # Copy numeric fields
        for col in NUMERIC_FIELDS:
            if col not in item:
                raise KeyError(f"Column '{col}' missing in JSON entry: {item}")
            row[col] = float(item[col])

        # One-hot for route_type
        rt_value = item.get("route_type", None)
        if rt_value not in ROUTE_TYPES:
            raise ValueError(f"Unexpected route_type '{rt_value}' in data")
        for rt in ROUTE_TYPES:
            row[f"rt_{rt}"] = 1.0 if rt_value == rt else 0.0

        rows.append(row)

    # 3. Prepare header: numeric fields + one-hot cols
    header = NUMERIC_FIELDS + ONE_HOT_COLS

    # 4. Write RAW CSV (no normalization)
    with open(RAW_CSV, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row[col] for col in header])

    print(f"Saved raw dataset to {RAW_CSV}")

    # 5. Compute normalization stats for ALL columns except one-hot
    numeric_to_normalize = NUMERIC_FIELDS  # all numeric, including avg_speed_ms
    stats = compute_mean_std(rows, numeric_to_normalize)

    # 6. Build normalized rows
    norm_rows = []
    for row in rows:
        norm_row = {}
        # normalize numeric columns
        for col in numeric_to_normalize:
            mean, std = stats[col]
            norm_row[col] = (row[col] - mean) / std
        # copy one-hot cols unchanged
        for col in ONE_HOT_COLS:
            norm_row[col] = row[col]
        norm_rows.append(norm_row)

    # 7. Write NORMALIZED CSV
    with open(NORM_CSV, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)
        for row in norm_rows:
            writer.writerow([row[col] for col in header])

    print(f"Saved normalized dataset to {NORM_CSV}")
    print("Note: all numeric columns (including avg_speed_ms) were normalized; one-hot columns kept as 0/1.")


if __name__ == "__main__":
    main()