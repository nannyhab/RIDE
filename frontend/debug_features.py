from optimize_setup import synthesize_track_features, load_meta, STATS_PATH

def print_features(flags):
    print(f"Flags: {flags}")
    feats, r_type = synthesize_track_features(flags)
    print(f"Route Type: {r_type}")
    for k, v in feats.items():
        print(f"  {k}: {v:.4f}")
    print("-" * 20)

print_features(["high_speed_sections"])
print_features(["high_speed_sections", "tight_curves", "wide_multi_lane"])
