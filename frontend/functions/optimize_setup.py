import json
import math
from typing import Dict, List, Tuple

import torch
from torch import nn

# -------------------------
# Paths
# -------------------------

BEST_MODEL_PATH = "best_model.pt"
STATS_PATH = "normalization_stats.json"

# Gear/tire grids (same as training data)
GEAR_RATIOS = [1.5, 2.5, 4.0, 6.0]
TIRE_FRICTIONS = [0.5, 1.0, 2.0, 3.0]

# -------------------------
# Helper: load stats/meta
# -------------------------

def load_meta(stats_path: str = STATS_PATH) -> Dict:
    with open(stats_path, "r") as f:
        meta = json.load(f)
    return meta


# -------------------------
# Model definition
# (must match training)
# -------------------------

def create_model(input_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )


# -------------------------
# Track feature synthesis
# -------------------------

# Global raw-data min/max to keep synthesized values in-range
RAW_MIN = {
    "total_length": 861.8068857178112,
    "avg_curvature": 0.0030865551763776377,
    "max_curvature": 0.0048649695770353555,
    "std_curvature": 0.001960150645148558,
    "tight_corners_pct": 0.0,
    "straight_pct": 0.35,
    "avg_slope": 0.0,
    "max_slope": 0.0,
    "elevation_range": 0.0,
    "max_elevation": 0.0,
    "min_elevation": 0.0,
}

RAW_MAX = {
    "total_length": 930.5335371008955,
    "avg_curvature": 0.016462962209355513,
    "max_curvature": 0.04807396744081191,
    "std_curvature": 0.02033278133377408,
    "tight_corners_pct": 0.5,
    "straight_pct": 1.0,
    "avg_slope": 0.011939938450975941,
    "max_slope": 0.05779095935569062,
    "elevation_range": 11.0,
    "max_elevation": 11.0,
    "min_elevation": 0.0,
}

# Route-type-specific *mean* feature profiles (from your dataset)
BASE_MEANS = {
    "elevation": {
        "total_length": 874.9674528288151,
        "avg_curvature": 0.003381853812289305,
        "max_curvature": 0.018977307227006864,
        "std_curvature": 0.005289018603516639,
        "tight_corners_pct": 0.010000000000000002,
        "straight_pct": 0.700000000000001,
        "avg_slope": 0.011815732008076367,
        "max_slope": 0.05593031686871559,
        "elevation_range": 11.0,
        "max_elevation": 11.0,
        "min_elevation": 0.0,
    },
    "mixed": {
        "total_length": 880.6855305616396,
        "avg_curvature": 0.01646296220935551,
        "max_curvature": 0.04566138226786209,
        "std_curvature": 0.020332781333774082,
        "tight_corners_pct": 0.4000000000000001,
        "straight_pct": 0.5999999999999999,
        "avg_slope": 0.0,
        "max_slope": 0.0,
        "elevation_range": 0.0,
        "max_elevation": 0.0,
        "min_elevation": 0.0,
    },
    "straight_speed": {
        "total_length": 903.8624699851277,
        "avg_curvature": 0.009339882759635355,
        "max_curvature": 0.03205113930638572,
        "std_curvature": 0.011989863655287623,
        "tight_corners_pct": 0.24999999999999978,
        "straight_pct": 0.6600000000000001,
        "avg_slope": 9.645591372253547e-05,
        "max_slope": 0.0009041993576998253,
        "elevation_range": 0.039178466796875,
        "max_elevation": 0.039178466796875,
        "min_elevation": 0.0,
    },
    "tight_curves": {
        "total_length": 920.4307975075171,
        "avg_curvature": 0.013988226571267841,
        "max_curvature": 0.03742921088486738,
        "std_curvature": 0.014793246712394065,
        "tight_corners_pct": 0.3800000000000002,
        "straight_pct": 0.4699999999999999,
        "avg_slope": 0.0,
        "max_slope": 0.0,
        "elevation_range": 0.0,
        "max_elevation": 0.0,
        "min_elevation": 0.0,
    },
    "tire_friction_test": {
        "total_length": 926.7876757549495,
        "avg_curvature": 0.014471593967694058,
        "max_curvature": 0.03571446727888195,
        "std_curvature": 0.013535455357858791,
        "tight_corners_pct": 0.5,
        "straight_pct": 0.4,
        "avg_slope": 0.0,
        "max_slope": 0.0,
        "elevation_range": 0.0,
        "max_elevation": 0.0,
        "min_elevation": 0.0,
    },
}

def clip(val: float, name: str) -> float:
    return max(RAW_MIN[name], min(RAW_MAX[name], val))


def choose_base_route_type(flags: List[str]) -> str:
    flags_set = set(flags)
    # names we expect:
    # "intersections_90", "steep_elevation", "high_speed_sections",
    # "tight_curves", "long_straightaways", "narrow_lanes", "wide_multi_lane"
    if "steep_elevation" in flags_set:
        return "elevation"
    if "tight_curves" in flags_set and not (
        "high_speed_sections" in flags_set or "long_straightaways" in flags_set
    ):
        return "tight_curves"
    if "high_speed_sections" in flags_set or "long_straightaways" in flags_set:
        return "straight_speed"
    if "intersections_90" in flags_set:
        return "mixed"
    # fallback
    return "straight_speed"


def synthesize_track_features(flags: List[str]) -> Tuple[Dict[str, float], str]:
    """
    Given a list of selected UI flags, return:
      - track_features: dict of raw (unnormalized) global features
      - route_type: one of your 5 route types
    """
    flags_set = set(flags)
    route_type = choose_base_route_type(flags)
    base = BASE_MEANS[route_type]

    # Start from base means
    feat = {k: float(v) for k, v in base.items()}

    # 90° intersections -> more tight corners, grid-like
    if "intersections_90" in flags_set:
        feat["tight_corners_pct"] = clip(0.35, "tight_corners_pct")
        feat["straight_pct"] = clip(0.55, "straight_pct")
        feat["avg_curvature"] = clip(feat["avg_curvature"] * 1.1, "avg_curvature")
        feat["max_curvature"] = clip(feat["max_curvature"] * 1.1, "max_curvature")
        feat["std_curvature"] = clip(feat["std_curvature"] * 1.1, "std_curvature")

    # High-speed sections -> lots of straights, lower curvature
    if "high_speed_sections" in flags_set:
        feat["straight_pct"] = clip(0.80, "straight_pct")
        feat["tight_corners_pct"] = clip(0.10, "tight_corners_pct")
        feat["avg_curvature"] = clip(0.005, "avg_curvature")
        feat["max_curvature"] = clip(0.025, "max_curvature")
        feat["std_curvature"] = clip(0.007, "std_curvature")
        feat["total_length"] = clip(feat["total_length"] + 15.0, "total_length")

    # Tight curves / S-bends -> more corners, higher curvature
    if "tight_curves" in flags_set:
        feat["tight_corners_pct"] = clip(0.45, "tight_corners_pct")
        feat["straight_pct"] = clip(0.45, "straight_pct")
        feat["avg_curvature"] = clip(0.014, "avg_curvature")
        feat["max_curvature"] = clip(0.038, "max_curvature")
        feat["std_curvature"] = clip(0.015, "std_curvature")

    # Long straightaways -> extreme straight dominance
    if "long_straightaways" in flags_set:
        feat["straight_pct"] = clip(0.95, "straight_pct")
        feat["tight_corners_pct"] = clip(0.05, "tight_corners_pct")
        feat["avg_curvature"] = clip(0.004, "avg_curvature")
        feat["max_curvature"] = clip(0.022, "max_curvature")
        feat["std_curvature"] = clip(0.005, "std_curvature")
        feat["total_length"] = clip(feat["total_length"] + 25.0, "total_length")

    # Steep elevation changes -> copy elevation profile
    if "steep_elevation" in flags_set:
        feat["avg_slope"] = 0.0118
        feat["max_slope"] = 0.056
        feat["elevation_range"] = 11.0
        feat["max_elevation"] = 11.0
        feat["min_elevation"] = 0.0

    # Narrow lanes -> slightly more 'difficult'
    if "narrow_lanes" in flags_set:
        feat["tight_corners_pct"] = clip(feat["tight_corners_pct"] + 0.05,
                                         "tight_corners_pct")
        feat["avg_curvature"] = clip(feat["avg_curvature"] * 1.05, "avg_curvature")
        feat["std_curvature"] = clip(feat["std_curvature"] * 1.05, "std_curvature")

    # Wide multi-lane roads -> slightly easier
    if "wide_multi_lane" in flags_set:
        feat["tight_corners_pct"] = clip(feat["tight_corners_pct"] - 0.05,
                                         "tight_corners_pct")
        feat["straight_pct"] = clip(feat["straight_pct"] + 0.05, "straight_pct")
        feat["avg_curvature"] = clip(feat["avg_curvature"] * 0.9, "avg_curvature")
        feat["std_curvature"] = clip(feat["std_curvature"] * 0.9, "std_curvature")

    return feat, route_type


# -------------------------
# Main optimization routine
# -------------------------

def optimize_setup_for_flags(selected_flags: List[str]) -> Dict:
    """
    Given a list of UI flags, synthesize a track, then run the trained model
    over all gear_ratio / tire_friction combinations and print the best one.
    Returns a dictionary with the best parameters.
    """
    meta = load_meta(STATS_PATH)
    numeric_features_input = meta["numeric_features_input"]
    target_col = meta["target_col"]          # "avg_speed_ms"
    feature_cols = meta["feature_cols"]      # numeric + one-hot (order)
    route_types = meta["route_types"]
    stats = meta["stats"]                    # col -> {"mean", "std"}

    # Build track features (raw)
    track_features, route_type = synthesize_track_features(selected_flags)
    print("Using base route_type:", route_type)
    
    # Create model
    input_dim = len(feature_cols)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(input_dim)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    def norm(col, value):
        if col not in stats:
            return value
        mean = stats[col]["mean"]
        std = stats[col]["std"] if stats[col]["std"] != 0 else 1.0
        return (value - mean) / std

    mean_y = stats[target_col]["mean"]
    std_y = stats[target_col]["std"] if stats[target_col]["std"] != 0 else 1.0

    best_speed = -1e9
    best_combo = None
    
    # We can also calculate a "baseline" speed if we assume some default parameters
    # e.g. gear_ratio=4.0, tire_friction=2.0 (just as an example baseline)
    baseline_combo = (4.0, 2.0) 
    baseline_speed = None

    for gr in GEAR_RATIOS:
        for tf in TIRE_FRICTIONS:
            # Build raw row for this combination
            row = {}
            for col in numeric_features_input:
                if col == "gear_ratio":
                    val = gr
                elif col == "tire_friction":
                    val = tf
                else:
                    val = track_features[col]
                row[col] = float(val)

            # Normalize numeric inputs
            for col in numeric_features_input:
                row[col] = norm(col, row[col])

            # One-hot route_type
            for rt in route_types:
                row[f"rt_{rt}"] = 1.0 if route_type == rt else 0.0

            # Build feature vector
            x_vec = [row[c] for c in feature_cols]
            x_tensor = torch.tensor(x_vec, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                y_norm = model(x_tensor).squeeze(0).item()

            # Denormalize to real avg_speed_ms
            y_real = y_norm * std_y + mean_y
            
            if y_real > best_speed:
                best_speed = y_real
                best_combo = (gr, tf)

    # Calculate predicted time (assuming some arbitrary distance, or just return speed)
    # The UI expects "predicted_time". 
    # Speed is in m/s. Let's assume a standard distance for the track, e.g., track_features["total_length"]
    track_length = track_features.get("total_length", 1000.0)
    
    predicted_time = track_length / best_speed if best_speed > 0 else 0.0
    
    return {
        "tire_friction": best_combo[1],
        "gear_ratio": best_combo[0],
        "predicted_time": predicted_time,
        "predicted_speed_ms": best_speed,
        "profile_summary": f"Route Type: {route_type}, Length: {track_length:.1f}m"
    }


# -------------------------
# Example usage
# -------------------------

if __name__ == "__main__":
    # Example: 90° intersections + high-speed sections
    example_flags = [
        "intersections_90",
        "high_speed_sections",
        # "steep_elevation",
        # "tight_curves",
        # "long_straightaways",
        # "narrow_lanes",
        # "wide_multi_lane",
    ]
    optimize_setup_for_flags(example_flags)