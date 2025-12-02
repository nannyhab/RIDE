import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# Load data
with open('track_optimization_data_final/dataset_checkpoint.pkl', 'rb') as f:
    data = pickle.load(f)

print(f"Total points: {len(data)}")
print(f"Scenarios: {sorted(set(d['track_id'] for d in data))}")

# Prepare features
exclude = {
    'track_id', 'map_name', 'route_type', 'complexity', 'travel_time', 
    'obey_traffic', 'obey_signs', 'obey_lights',
    'avg_speed_ms', 'max_speed_ms', 'min_speed_ms', 'speed_std',
    'avg_speed_kmh', 'max_speed_kmh',
    'avg_lateral_accel', 'max_lateral_accel',
    'avg_longitudinal_accel', 'max_longitudinal_accel',
    'avg_steering', 'max_steering', 'smoothness',
    'actual_distance_m', 'avg_lateral_offset', 'max_lateral_offset',
    'stop_count', 'collision_count'
}

feat_names = [k for k in data[0].keys() if k not in exclude]
X = np.array([[d[k] for k in feat_names] for d in data])
y = np.array([d['travel_time'] for d in data])
groups = np.array([d['track_id'] for d in data])

print("\n" + "="*60)
print("LEAVE-ONE-SCENARIO-OUT CROSS-VALIDATION")
print("="*60)

# Leave-one-scenario-out CV
logo = LeaveOneGroupOut()
r2_scores = []
mae_scores = []

for train_idx, test_idx in logo.split(X, y, groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    test_scenario = groups[test_idx][0]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    r2_scores.append(r2)
    mae_scores.append(mae)
    
    print(f"Scenario {test_scenario} held out: R²={r2:.3f}, MAE={mae:.2f}s")

print("\n" + "="*60)
print("CROSS-VALIDATION SUMMARY")
print("="*60)
print(f"Mean R²: {np.mean(r2_scores):.3f} (+/- {np.std(r2_scores):.3f})")
print(f"Mean MAE: {np.mean(mae_scores):.2f}s (+/- {np.std(mae_scores):.2f}s)")
print(f"Worst R²: {np.min(r2_scores):.3f} (scenario {groups[test_idx][0]})")

if np.mean(r2_scores) > 0.90:
    print("\n✓ MODEL IS ROBUST - Good generalization to unseen scenarios!")
elif np.mean(r2_scores) > 0.80:
    print("\n⚠ MODEL IS DECENT - Acceptable but room for improvement")
else:
    print("\n✗ MODEL HAS ISSUES - Poor generalization to new scenarios")

# Percentage error
mean_time = np.mean(y)
pct_error = (np.mean(mae_scores) / mean_time) * 100
print(f"\nRelative error: {pct_error:.1f}% of average travel time")

