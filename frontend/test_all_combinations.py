import itertools
import json
import pandas as pd
import sys
import os
from contextlib import contextmanager
from optimize_setup import optimize_setup_for_flags

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

ALL_FLAGS = [
    "intersections_90",
    "steep_elevation",
    "high_speed_sections",
    "tight_curves",
    "long_straightaways",
    "narrow_lanes",
    "wide_multi_lane"
]

results = []

print(f"Testing all combinations of {len(ALL_FLAGS)} flags...")

# Iterate through all possible combinations length 1 to 7
count = 0
total_combinations = 2**len(ALL_FLAGS) - 1

for r in range(1, len(ALL_FLAGS) + 1):
    for combo in itertools.combinations(ALL_FLAGS, r):
        flags = list(combo)
        count += 1
        if count % 10 == 0:
            sys.stdout.write(f"\rProcessing {count}/{total_combinations}...")
            sys.stdout.flush()
            
        try:
            with suppress_stdout():
                res = optimize_setup_for_flags(flags)
            
            # Flatten the result for easier CSV
            entry = {
                "selected_flags": ", ".join(flags),
                "tire_friction": res.get("tire_friction"),
                "gear_ratio": res.get("gear_ratio"),
                "predicted_speed_ms": res.get("predicted_speed_ms"),
                "predicted_time": res.get("predicted_time"),
                "profile_summary": res.get("profile_summary")
            }
            results.append(entry)
        except Exception as e:
            print(f"\nError with flags {flags}: {e}")

print(f"\nDone! Tested {len(results)} combinations.")

# Save to JSON
with open("all_combinations_results.json", "w") as f:
    json.dump(results, f, indent=2)

# Save to CSV
df = pd.DataFrame(results)
df.to_csv("all_combinations_results.csv", index=False)

print("Results saved to 'all_combinations_results.json' and 'all_combinations_results.csv'")
