"""
Track-Based Physical Parameter Optimizer
Predicts optimal gear ratio and tire friction from track features
"""

import carla
import numpy as np
import math
import time
import pickle
import json
import os
from collections import defaultdict
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = "track_optimization_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Physical parameters to test
GEAR_RATIOS = [2.5, 3.0, 3.5, 4.0]
TIRE_FRICTIONS = [0.6, 0.8, 1.0, 1.2]

# Track generation parameters
NUM_RANDOM_TRACKS = 20  # Generate 20 random tracks
TRACK_SEGMENTS = 20  # Divide each track into 20 segments

MAX_RUN_TIME = 120  # seconds


# ============================================================================
# RANDOM TRACK GENERATOR
# ============================================================================

class RandomTrackGenerator:
    """
    Generate random tracks with varying characteristics
    """
    
    def __init__(self):
        self.track_types = [
            'straight', 'gentle_curves', 'tight_curves', 
            'mixed', 's_curve', 'spiral'
        ]
    
    def generate_random_track(self, track_id):
        """
        Generate a random track configuration
        Returns XODR string
        """
        track_type = np.random.choice(self.track_types)
        
        if track_type == 'straight':
            return self._generate_straight(track_id)
        elif track_type == 'gentle_curves':
            return self._generate_gentle_curves(track_id)
        elif track_type == 'tight_curves':
            return self._generate_tight_curves(track_id)
        elif track_type == 'mixed':
            return self._generate_mixed(track_id)
        elif track_type == 's_curve':
            return self._generate_s_curve(track_id)
        elif track_type == 'spiral':
            return self._generate_spiral(track_id)
    
    def _generate_straight(self, track_id):
        """Straight track"""
        length = np.random.uniform(150, 250)
        segments = [{"type": "line", "L": length}]
        return self._build_xodr(segments, 0, 0, 0, track_id)
    
    def _generate_gentle_curves(self, track_id):
        """Gentle curves"""
        R = np.random.uniform(60, 100)
        arc_length = np.random.uniform(40, 80)
        straight_length = np.random.uniform(20, 40)
        
        segments = [
            {"type": "arc", "L": arc_length, "k": 1.0/R},
            {"type": "line", "L": straight_length},
            {"type": "arc", "L": arc_length, "k": -1.0/R},
            {"type": "line", "L": straight_length},
        ]
        return self._build_xodr(segments, 0, 0, 0, track_id)
    
    def _generate_tight_curves(self, track_id):
        """Tight curves"""
        R = np.random.uniform(20, 40)
        arc_length = np.random.uniform(30, 60)
        straight_length = np.random.uniform(10, 25)
        
        segments = [
            {"type": "arc", "L": arc_length, "k": 1.0/R},
            {"type": "line", "L": straight_length},
            {"type": "arc", "L": arc_length, "k": -1.0/R},
            {"type": "line", "L": straight_length},
        ]
        return self._build_xodr(segments, 0, 0, 0, track_id)
    
    def _generate_mixed(self, track_id):
        """Mixed: straight + gentle + tight"""
        segments = [
            {"type": "line", "L": np.random.uniform(30, 50)},
            {"type": "arc", "L": 40, "k": 1.0/80},  # Gentle
            {"type": "line", "L": np.random.uniform(20, 30)},
            {"type": "arc", "L": 35, "k": -1.0/30},  # Tight
            {"type": "line", "L": np.random.uniform(25, 40)},
        ]
        return self._build_xodr(segments, 0, 0, 0, track_id)
    
    def _generate_s_curve(self, track_id):
        """S-shaped curves"""
        R1 = np.random.uniform(40, 70)
        R2 = np.random.uniform(40, 70)
        
        segments = [
            {"type": "arc", "L": 60, "k": 1.0/R1},
            {"type": "line", "L": 25},
            {"type": "arc", "L": 60, "k": -1.0/R2},
        ]
        return self._build_xodr(segments, 0, 0, 0, track_id)
    
    def _generate_spiral(self, track_id):
        """Gradually tightening curve"""
        segments = [
            {"type": "arc", "L": 40, "k": 1.0/90},
            {"type": "arc", "L": 40, "k": 1.0/60},
            {"type": "arc", "L": 40, "k": 1.0/40},
            {"type": "line", "L": 30},
        ]
        return self._build_xodr(segments, 0, 0, 0, track_id)
    
    def _build_xodr(self, segments, x0, y0, hdg0, track_id):
        """Build XODR from segments"""
        header = f'''<?xml version="1.0" standalone="yes"?>
<OpenDRIVE revMajor="1" revMinor="4" name="track_{track_id}" version="1.5">
  <header revMajor="1" revMinor="4" name="track_{track_id}" version="1.5"
          date="{datetime.datetime.utcnow().isoformat()}" 
          north="0" south="0" east="0" west="0" vendor="generated"/>'''
        
        planview, total_length = self._planview_from_segments(segments, x0, y0, hdg0)
        
        lanes = f'''
    <elevationProfile/>
    <lateralProfile/>
    <lanes>
      <laneOffset s="0.0" a="0" b="0" c="0" d="0"/>
      <laneSection s="0.0">
        <center>
          <lane id="0" type="none" level="false"><link/></lane>
        </center>
        <right>
          <lane id="-1" type="driving" level="false">
            <link/>
            <width sOffset="0.0" a="3.5" b="0" c="0" d="0"/>
            <roadMark sOffset="0.0" type="broken" weight="standard" color="standard" width="0.13"/>
          </lane>
        </right>
      </laneSection>
    </lanes>
    <objects/>
    <signals/>'''
        
        road = f'''
  <road name="R0" length="{total_length:.6f}" id="0" junction="-1">
    <link/>{planview}{lanes}
  </road>'''
        
        return header + road + '\n</OpenDRIVE>\n'
    
    def _planview_from_segments(self, segments, x0, y0, hdg0):
        """Generate planView XML"""
        x, y, hdg = x0, y0, hdg0
        s_cum = 0.0
        xml_parts = ['    <planView>\n']
        
        for seg in segments:
            L = float(seg["L"])
            
            if seg["type"] == "line":
                xml_parts.append(
                    f'      <geometry s="{s_cum:.6f}" x="{x:.6f}" y="{y:.6f}" '
                    f'hdg="{hdg:.12f}" length="{L:.6f}">\n'
                    f'        <line/>\n'
                    f'      </geometry>\n'
                )
                x += L * math.cos(hdg)
                y += L * math.sin(hdg)
            else:  # arc
                k = float(seg["k"])
                xml_parts.append(
                    f'      <geometry s="{s_cum:.6f}" x="{x:.6f}" y="{y:.6f}" '
                    f'hdg="{hdg:.12f}" length="{L:.6f}">\n'
                    f'        <arc curvature="{k:.12f}"/>\n'
                    f'      </geometry>\n'
                )
                dpsi = k * L
                if abs(k) > 1e-9:
                    r = 1.0 / k
                    x += r * (math.sin(hdg + dpsi) - math.sin(hdg))
                    y -= r * (math.cos(hdg + dpsi) - math.cos(hdg))
                    hdg += dpsi
                else:
                    x += L * math.cos(hdg)
                    y += L * math.sin(hdg)
            
            s_cum += L
        
        xml_parts.append('    </planView>\n')
        return ''.join(xml_parts), s_cum


# ============================================================================
# TRACK FEATURE EXTRACTION
# ============================================================================

class TrackFeatureExtractor:
    """
    Extract numerical features from track waypoints
    """
    
    def __init__(self, num_segments=TRACK_SEGMENTS):
        self.num_segments = num_segments
    
    def extract_features(self, waypoints):
        """
        Extract features from waypoints
        
        Returns:
            feature_dict: Dictionary of track features
        """
        if len(waypoints) < 3:
            return None
        
        # Convert to arrays
        x = np.array([wp.transform.location.x for wp in waypoints])
        y = np.array([wp.transform.location.y for wp in waypoints])
        z = np.array([wp.transform.location.z for wp in waypoints])
        
        # Total length
        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)
        segment_lengths = np.sqrt(dx**2 + dy**2 + dz**2)
        total_length = segment_lengths.sum()
        
        # Resample into N equal segments
        cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        target_lengths = np.linspace(0, total_length, self.num_segments + 1)
        
        # Per-segment features
        curvatures = []
        slopes = []
        
        for i in range(self.num_segments):
            start_s = target_lengths[i]
            end_s = target_lengths[i + 1]
            
            # Find waypoints in this segment
            mask = (cumulative_length >= start_s) & (cumulative_length <= end_s)
            if mask.sum() < 3:
                curvatures.append(0.0)
                slopes.append(0.0)
                continue
            
            seg_x = x[mask]
            seg_y = y[mask]
            seg_z = z[mask]
            
            # Curvature (rate of heading change)
            curvature = self._compute_curvature(seg_x, seg_y)
            curvatures.append(curvature)
            
            # Slope (elevation change)
            if len(seg_z) > 1:
                slope = abs(seg_z[-1] - seg_z[0]) / max(end_s - start_s, 1e-6)
            else:
                slope = 0.0
            slopes.append(slope)
        
        # Global features
        curvature_array = np.array(curvatures)
        
        features = {
            # Global track features
            'total_length': total_length,
            'avg_curvature': np.mean(np.abs(curvature_array)),
            'max_curvature': np.max(np.abs(curvature_array)),
            'std_curvature': np.std(curvature_array),
            'avg_slope': np.mean(slopes),
            'max_slope': np.max(slopes),
            
            # Tight corners percentage (curvature > threshold)
            'tight_corners_pct': np.sum(np.abs(curvature_array) > 0.02) / len(curvature_array),
            
            # Straight percentage (curvature < threshold)
            'straight_pct': np.sum(np.abs(curvature_array) < 0.005) / len(curvature_array),
            
            # Per-segment features (flattened)
            **{f'curvature_{i}': curvatures[i] for i in range(self.num_segments)},
            **{f'slope_{i}': slopes[i] for i in range(self.num_segments)},
        }
        
        return features
    
    def _compute_curvature(self, x, y):
        """
        Compute curvature from points
        Curvature = 1/R where R is radius
        """
        if len(x) < 3:
            return 0.0
        
        # Use finite differences
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        # Curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = np.power(dx**2 + dy**2, 1.5) + 1e-6
        
        curvature = numerator / denominator
        
        return np.mean(curvature)


# ============================================================================
# DATA COLLECTION WITH AUTOPILOT
# ============================================================================

def collect_data_point(client, track_xodr, track_id, gear_ratio, tire_friction):
    """
    Run autopilot on a track with specific parameters and measure travel time
    
    Returns:
        track_features: dict
        travel_time: float
        success: bool
    """
    print(f"  Track {track_id}, gear={gear_ratio:.2f}, friction={tire_friction:.2f}")
    
    try:
        # Generate world from XODR
        params = carla.OpendriveGenerationParameters(
            vertex_distance=2.0,
            max_road_length=50.0,
            wall_height=1.0,
            additional_width=0.6,
            smooth_junctions=True,
            enable_mesh_visibility=True,
            enable_pedestrian_navigation=True
        )
        
        world = client.generate_opendrive_world(track_xodr, params)
        
        # Sync mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        # Get waypoints
        waypoints = get_track_waypoints(world)
        
        if len(waypoints) < 10:
            print("    ERROR: Not enough waypoints")
            return None, None, False
        
        # Extract track features
        extractor = TrackFeatureExtractor()
        track_features = extractor.extract_features(waypoints)
        
        if track_features is None:
            print("    ERROR: Could not extract features")
            return None, None, False
        
        # Spawn vehicle
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        
        spawn_point = waypoints[0].transform
        spawn_point.location.z += 2.0
        
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        
        # Set physical parameters
        physics = vehicle.get_physics_control()
        
        # Set gear ratio (affects speed)
        # CARLA doesn't have direct gear ratio, so we modify max RPM and torque curve
        physics.max_rpm = 5000.0 * gear_ratio / 3.0  # Scale with gear ratio
        
        # Set tire friction
        for wheel in physics.wheels:
            wheel.tire_friction = tire_friction
        
        vehicle.apply_physics_control(physics)
        
        # Enable AUTOPILOT
        vehicle.set_autopilot(True)
        
        # Run and measure time
        start_time = time.time()
        start_location = vehicle.get_location()
        
        max_steps = int(MAX_RUN_TIME / 0.05)
        last_progress_time = time.time()
        prev_distance = 0.0
        
        for step in range(max_steps):
            world.tick()
            
            # Check progress
            current_location = vehicle.get_location()
            distance_traveled = math.sqrt(
                (current_location.x - start_location.x)**2 +
                (current_location.y - start_location.y)**2
            )
            
            # Check if stuck
            if distance_traveled - prev_distance < 0.1:
                if time.time() - last_progress_time > 10.0:
                    print("    STUCK")
                    break
            else:
                last_progress_time = time.time()
                prev_distance = distance_traveled
            
            # Check if completed (near start again)
            if distance_traveled > 50 and distance_traveled < prev_distance:
                # Completed lap
                break
            
            # Check if traveled far enough
            if distance_traveled > track_features['total_length'] * 0.9:
                break
        
        travel_time = time.time() - start_time
        
        # Cleanup
        vehicle.destroy()
        
        print(f"    Travel time: {travel_time:.2f}s")
        
        return track_features, travel_time, True
        
    except Exception as e:
        print(f"    ERROR: {e}")
        return None, None, False


def get_track_waypoints(world):
    """Get waypoints for the track"""
    map_obj = world.get_map()
    
    # Try to get waypoint at road 0
    wp = None
    for lane_id in (-1, 1):
        for s0 in (0.5, 1.0, 2.0, 5.0):
            try:
                wp = map_obj.get_waypoint_xodr(0, lane_id, s0)
                if wp:
                    break
            except:
                pass
        if wp:
            break
    
    if wp is None:
        wp = map_obj.get_waypoint(
            carla.Location(0, 0, 0),
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )
    
    if wp is None:
        return []
    
    # Walk forward
    waypoints = []
    seen = set()
    
    for _ in range(5000):
        key = (int(wp.transform.location.x * 100),
               int(wp.transform.location.y * 100))
        
        if key in seen:
            break
        
        seen.add(key)
        waypoints.append(wp)
        
        next_wps = wp.next(0.5)
        if not next_wps:
            break
        
        wp = next_wps[0]
    
    return waypoints


def collect_all_data():
    """
    Main data collection loop
    """
    client = carla.Client('localhost', 2000)
    client.set_timeout(30.0)
    
    generator = RandomTrackGenerator()
    
    dataset = []
    
    print("="*60)
    print("COLLECTING DATA")
    print("="*60)
    
    for track_id in range(NUM_RANDOM_TRACKS):
        print(f"\nTrack {track_id + 1}/{NUM_RANDOM_TRACKS}")
        
        # Generate random track
        track_xodr = generator.generate_random_track(track_id)
        
        # Test all parameter combinations
        for gear_ratio in GEAR_RATIOS:
            for tire_friction in TIRE_FRICTIONS:
                
                features, travel_time, success = collect_data_point(
                    client, track_xodr, track_id, gear_ratio, tire_friction
                )
                
                if success and features is not None:
                    data_point = {
                        'track_id': track_id,
                        'gear_ratio': gear_ratio,
                        'tire_friction': tire_friction,
                        'travel_time': travel_time,
                        **features
                    }
                    dataset.append(data_point)
                
                # Small delay between runs
                time.sleep(1)
    
    # Save dataset
    with open(f'{DATA_DIR}/dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    
    with open(f'{DATA_DIR}/dataset.json', 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n✓ Collected {len(dataset)} data points")
    print(f"✓ Saved to {DATA_DIR}/")
    
    return dataset


# ============================================================================
# MODEL TRAINING
# ============================================================================

class TravelTimePredictor:
    """
    Regression model to predict travel time from track features + physical params
    """
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.scaler = StandardScaler()
        
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                random_state=42
            )
        elif model_type == 'neural_network':
            self.model = None  # Will be created in train()
    
    def train(self, dataset):
        """
        Train regression model
        
        dataset: list of dicts with track features + params + travel_time
        """
        # Prepare features and targets
        X = []
        y = []
        
        # Get feature names (exclude metadata)
        exclude_keys = {'track_id', 'travel_time'}
        feature_names = [k for k in dataset[0].keys() if k not in exclude_keys]
        
        for point in dataset:
            features = [point[k] for k in feature_names]
            X.append(features)
            y.append(point['travel_time'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train
        if self.model_type == 'random_forest':
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_pred = self.model.predict(X_train_scaled)
            test_pred = self.model.predict(X_test_scaled)
            
            train_mse = mean_squared_error(y_train, train_pred)
            test_mse = mean_squared_error(y_test, test_pred)
            train_r2 = r2_score(y_train, train_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            print(f"\n✓ Random Forest Results:")
            print(f"  Train MSE: {train_mse:.2f}, R²: {train_r2:.3f}")
            print(f"  Test MSE: {test_mse:.2f}, R²: {test_r2:.3f}")
            
            # Feature importance
            importances = self.model.feature_importances_
            top_features = sorted(zip(feature_names, importances), 
                                 key=lambda x: x[1], reverse=True)[:10]
            print(f"\n  Top 10 Important Features:")
            for feat, imp in top_features:
                print(f"    {feat}: {imp:.3f}")
        
        self.feature_names = feature_names
        
        return train_r2, test_r2
    
    def predict(self, track_features, gear_ratio, tire_friction):
        """
        Predict travel time for given track and parameters
        """
        # Build feature vector
        features = []
        for key in self.feature_names:
            if key == 'gear_ratio':
                features.append(gear_ratio)
            elif key == 'tire_friction':
                features.append(tire_friction)
            else:
                features.append(track_features[key])
        
        X = np.array([features])
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        return prediction
    
    def save(self, path):
        """Save model"""
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_type': self.model_type
            }, f)
    
    def load(self, path):
        """Load model"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_names = data['feature_names']
            self.model_type = data['model_type']


# ============================================================================
# GRID SEARCH FOR OPTIMAL PARAMETERS
# ============================================================================

def find_optimal_parameters(predictor, track_features):
    """
    Grid search over physical parameters to find minimum predicted time
    
    Args:
        predictor: Trained TravelTimePredictor
        track_features: Dict of track features
    
    Returns:
        best_params: (gear_ratio, tire_friction)
        best_time: predicted travel time
    """
    print("\n" + "="*60)
    print("GRID SEARCH FOR OPTIMAL PARAMETERS")
    print("="*60)
    
    # Fine-grained grid
    gear_grid = np.linspace(2.0, 4.5, 20)
    friction_grid = np.linspace(0.5, 1.3, 20)
    
    best_time = float('inf')
    best_params = None
    
    results = []
    
    for gear in gear_grid:
        for friction in friction_grid:
            predicted_time = predictor.predict(track_features, gear, friction)
            results.append((gear, friction, predicted_time))
            
            if predicted_time < best_time:
                best_time = predicted_time
                best_params = (gear, friction)
    
    print(f"\nOptimal Parameters:")
    print(f"  Gear Ratio: {best_params[0]:.3f}")
    print(f"  Tire Friction: {best_params[1]:.3f}")
    print(f"  Predicted Time: {best_time:.2f}s")
    
    # Show top 5
    results.sort(key=lambda x: x[2])
    print(f"\nTop 5 Configurations:")
    for i, (g, f, t) in enumerate(results[:5]):
        print(f"  {i+1}. Gear={g:.3f}, Friction={f:.3f}, Time={t:.2f}s")
    
    return best_params, best_time


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("TRACK-BASED PARAMETER OPTIMIZER")
    print("Gear Ratio + Tire Friction Optimization")
    print("="*60)
    
    # Step 1: Collect data
    print("\n[1/3] Collecting data from random tracks...")
    dataset = collect_all_data()
    
    if len(dataset) == 0:
        print("ERROR: No data collected!")
        return
    
    # Step 2: Train model
    print("\n[2/3] Training regression model...")
    predictor = TravelTimePredictor(model_type='random_forest')
    train_r2, test_r2 = predictor.train(dataset)
    predictor.save(f'{DATA_DIR}/travel_time_predictor.pkl')
    
    # Step 3: Test on new track
    print("\n[3/3] Testing on new random track...")
    
    client = carla.Client('localhost', 2000)
    generator = RandomTrackGenerator()
    test_track_xodr = generator.generate_random_track(999)
    
    # Generate world and extract features
    params = carla.OpendriveGenerationParameters(2.0, 50.0, 1.0, 0.6, True, True, True)
    world = client.generate_opendrive_world(test_track_xodr, params)
    
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    
    waypoints = get_track_waypoints(world)
    extractor = TrackFeatureExtractor()
    test_track_features = extractor.extract_features(waypoints)
    
    if test_track_features:
        print(f"\nTest Track Features:")
        print(f"  Length: {test_track_features['total_length']:.1f}m")
        print(f"  Avg Curvature: {test_track_features['avg_curvature']:.4f}")
        print(f"  Tight Corners: {test_track_features['tight_corners_pct']*100:.1f}%")
        print(f"  Straight: {test_track_features['straight_pct']*100:.1f}%")
        
        # Find optimal parameters
        best_params, best_time = find_optimal_parameters(predictor, test_track_features)
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
