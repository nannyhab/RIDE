"""
Track-Based Parameter Optimizer V2
Using manual controller instead of autopilot (more reliable)
"""

import carla
import numpy as np
import math
import time
import pickle
import json
import os
from collections import defaultdict, deque
import datetime

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = "track_optimization_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Physical parameters to test (reduced for faster testing)
GEAR_RATIOS = [2.5, 3.0, 3.5]
TIRE_FRICTIONS = [0.7, 1.0, 1.2]

# Track generation
NUM_RANDOM_TRACKS = 10
TRACK_SEGMENTS = 20

MAX_RUN_TIME = 60  # seconds
TARGET_SPEED = 30.0  # km/h


# ============================================================================
# SIMPLE CONTROLLER (instead of autopilot)
# ============================================================================

class SimpleController:
    """
    Basic pure pursuit controller - more reliable than autopilot
    """
    
    def __init__(self, target_speed_kmh=TARGET_SPEED):
        self.target_speed = target_speed_kmh / 3.6  # m/s
        self.lookahead = 15.0
        
    def control(self, vehicle, waypoints):
        """Get control for vehicle"""
        if len(waypoints) < 2:
            return 0.0, 0.0
        
        transform = vehicle.get_transform()
        location = transform.location
        
        # Find nearest waypoint
        min_dist = float('inf')
        nearest_idx = 0
        
        for i, wp in enumerate(waypoints):
            dist = location.distance(wp.transform.location)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        # Get target waypoint (lookahead)
        target_idx = min(nearest_idx + 20, len(waypoints) - 1)
        target = waypoints[target_idx].transform.location
        
        # Calculate steering
        yaw = math.radians(transform.rotation.yaw)
        dx = target.x - location.x
        dy = target.y - location.y
        
        # Transform to vehicle frame
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        
        # Pure pursuit
        ld = math.sqrt(local_x**2 + local_y**2) or 1e-6
        steer = math.atan2(2.0 * 2.88 * local_y, ld * ld)
        steer = np.clip(steer / 0.6, -1.0, 1.0)
        
        # Speed control
        velocity = vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2)
        
        if speed < self.target_speed * 0.5:
            throttle = 1.0
        elif speed < self.target_speed:
            throttle = 0.7
        else:
            throttle = 0.3
        
        return throttle, steer


# ============================================================================
# RANDOM TRACK GENERATOR
# ============================================================================

class RandomTrackGenerator:
    """Generate random tracks"""
    
    def generate_random_track(self, track_id):
        """Generate random track XODR"""
        track_types = [
            self._straight,
            self._gentle_s,
            self._tight_s,
            self._mixed,
        ]
        
        track_func = np.random.choice(track_types)
        return track_func(track_id)
    
    def _straight(self, track_id):
        """Straight track"""
        length = np.random.uniform(150, 200)
        segments = [{"type": "line", "L": length}]
        return self._build_xodr(segments, 0, 0, 0, track_id)
    
    def _gentle_s(self, track_id):
        """Gentle S curves"""
        R = np.random.uniform(60, 90)
        segments = [
            {"type": "arc", "L": 50, "k": 1.0/R},
            {"type": "line", "L": 30},
            {"type": "arc", "L": 50, "k": -1.0/R},
        ]
        return self._build_xodr(segments, 0, 0, 0, track_id)
    
    def _tight_s(self, track_id):
        """Tight S curves"""
        R = np.random.uniform(30, 50)
        segments = [
            {"type": "arc", "L": 40, "k": 1.0/R},
            {"type": "line", "L": 20},
            {"type": "arc", "L": 40, "k": -1.0/R},
        ]
        return self._build_xodr(segments, 0, 0, 0, track_id)
    
    def _mixed(self, track_id):
        """Mixed track"""
        segments = [
            {"type": "line", "L": 40},
            {"type": "arc", "L": 35, "k": 1.0/70},
            {"type": "line", "L": 30},
            {"type": "arc", "L": 35, "k": -1.0/50},
            {"type": "line", "L": 25},
        ]
        return self._build_xodr(segments, 0, 0, 0, track_id)
    
    def _build_xodr(self, segments, x0, y0, hdg0, track_id):
        """Build XODR from segments"""
        header = f'''<?xml version="1.0" standalone="yes"?>
<OpenDRIVE revMajor="1" revMinor="4" name="track_{track_id}">
  <header revMajor="1" revMinor="4" name="track_{track_id}"
          date="{datetime.datetime.utcnow().isoformat()}" 
          north="0" south="0" east="0" west="0" vendor="generated"/>'''
        
        planview, total_length = self._planview(segments, x0, y0, hdg0)
        
        lanes = '''
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
    
    def _planview(self, segments, x0, y0, hdg0):
        """Generate planView"""
        x, y, hdg = x0, y0, hdg0
        s = 0.0
        xml = ['    <planView>\n']
        
        for seg in segments:
            L = float(seg["L"])
            
            if seg["type"] == "line":
                xml.append(f'      <geometry s="{s:.6f}" x="{x:.6f}" y="{y:.6f}" hdg="{hdg:.12f}" length="{L:.6f}">\n')
                xml.append(f'        <line/>\n')
                xml.append(f'      </geometry>\n')
                x += L * math.cos(hdg)
                y += L * math.sin(hdg)
            else:
                k = float(seg["k"])
                xml.append(f'      <geometry s="{s:.6f}" x="{x:.6f}" y="{y:.6f}" hdg="{hdg:.12f}" length="{L:.6f}">\n')
                xml.append(f'        <arc curvature="{k:.12f}"/>\n')
                xml.append(f'      </geometry>\n')
                dpsi = k * L
                if abs(k) > 1e-9:
                    r = 1.0 / k
                    x += r * (math.sin(hdg + dpsi) - math.sin(hdg))
                    y -= r * (math.cos(hdg + dpsi) - math.cos(hdg))
                    hdg += dpsi
                else:
                    x += L * math.cos(hdg)
                    y += L * math.sin(hdg)
            s += L
        
        xml.append('    </planView>\n')
        return ''.join(xml), s


# ============================================================================
# TRACK FEATURE EXTRACTION
# ============================================================================

def extract_track_features(waypoints):
    """Extract numerical features from waypoints"""
    if len(waypoints) < 3:
        return None
    
    x = np.array([wp.transform.location.x for wp in waypoints])
    y = np.array([wp.transform.location.y for wp in waypoints])
    
    # Distances
    dx = np.diff(x)
    dy = np.diff(y)
    lengths = np.sqrt(dx**2 + dy**2)
    total_length = lengths.sum()
    
    # Curvatures (rate of heading change)
    headings = np.arctan2(dy, dx)
    curvatures = np.abs(np.diff(headings))
    curvatures = np.concatenate([[0], curvatures])
    
    # Features
    features = {
        'total_length': total_length,
        'avg_curvature': np.mean(curvatures),
        'max_curvature': np.max(curvatures),
        'std_curvature': np.std(curvatures),
        'tight_corners_pct': np.sum(curvatures > 0.1) / len(curvatures),
        'straight_pct': np.sum(curvatures < 0.01) / len(curvatures),
    }
    
    return features


def get_waypoints(world):
    """Get track waypoints"""
    map_obj = world.get_map()
    
    # Get starting waypoint
    wp = None
    for lane_id in (-1, 1):
        for s in (1.0, 5.0, 10.0):
            try:
                wp = map_obj.get_waypoint_xodr(0, lane_id, s)
                if wp:
                    break
            except:
                pass
        if wp:
            break
    
    if not wp:
        return []
    
    # Collect waypoints
    waypoints = []
    seen = set()
    
    for _ in range(3000):
        key = (int(wp.transform.location.x * 10), int(wp.transform.location.y * 10))
        if key in seen:
            break
        seen.add(key)
        waypoints.append(wp)
        
        next_wps = wp.next(1.0)
        if not next_wps:
            break
        wp = next_wps[0]
    
    return waypoints


# ============================================================================
# DATA COLLECTION
# ============================================================================

def collect_data_point(client, xodr, track_id, gear_ratio, tire_friction):
    """Run vehicle and measure travel time"""
    print(f"  Track {track_id}, gear={gear_ratio:.1f}, friction={tire_friction:.1f}", end=" ")
    
    try:
        # Generate world
        params = carla.OpendriveGenerationParameters(2.0, 50.0, 1.0, 0.6, True, True, True)
        world = client.generate_opendrive_world(xodr, params)
        
        # Sync mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        # Get waypoints
        waypoints = get_waypoints(world)
        if len(waypoints) < 10:
            print("- No waypoints")
            return None, None, False
        
        # Extract features
        features = extract_track_features(waypoints)
        if not features:
            print("- No features")
            return None, None, False
        
        # Spawn vehicle
        bp = world.get_blueprint_library().filter('vehicle.tesla.model3')[0]
        spawn = waypoints[5].transform
        spawn.location.z += 2.0
        
        vehicle = None
        for i in range(3):
            try:
                vehicle = world.spawn_actor(bp, spawn)
                break
            except:
                spawn.location.z += 2.0
        
        if not vehicle:
            print("- Spawn failed")
            return None, None, False
        
        # Set physics
        physics = vehicle.get_physics_control()
        physics.max_rpm = 5000.0 * (gear_ratio / 3.0)
        for wheel in physics.wheels:
            wheel.tire_friction = tire_friction
        vehicle.apply_physics_control(physics)
        
        # Settle
        for _ in range(10):
            world.tick()
        
        # Drive with controller
        controller = SimpleController(target_speed_kmh=TARGET_SPEED)
        
        start_time = time.time()
        start_loc = vehicle.get_location()
        
        positions = deque(maxlen=50)
        max_steps = 1200  # 60 seconds
        
        for step in range(max_steps):
            throttle, steer = controller.control(vehicle, waypoints)
            vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))
            world.tick()
            
            # Track position
            loc = vehicle.get_location()
            positions.append((loc.x, loc.y))
            
            # Check if stuck (not moving)
            if len(positions) == 50:
                movement = sum(
                    math.sqrt((positions[i][0] - positions[i-10][0])**2 +
                             (positions[i][1] - positions[i-10][1])**2)
                    for i in range(10, 50, 10)
                )
                if movement < 5.0:  # Less than 5m in 2.5 seconds
                    print("- Stuck")
                    vehicle.destroy()
                    return None, None, False
            
            # Check distance
            distance = math.sqrt((loc.x - start_loc.x)**2 + (loc.y - start_loc.y)**2)
            
            # Completed track
            if distance > features['total_length'] * 0.8:
                break
        
        travel_time = time.time() - start_time
        
        vehicle.destroy()
        time.sleep(0.5)
        
        print(f"- {travel_time:.1f}s")
        
        return features, travel_time, True
        
    except Exception as e:
        print(f"- Error: {e}")
        return None, None, False


def collect_all_data():
    """Collect dataset"""
    client = carla.Client('localhost', 2000)
    client.set_timeout(30.0)
    
    generator = RandomTrackGenerator()
    dataset = []
    
    print("="*60)
    print("COLLECTING DATA")
    print("="*60)
    
    for track_id in range(NUM_RANDOM_TRACKS):
        print(f"\nTrack {track_id + 1}/{NUM_RANDOM_TRACKS}")
        
        xodr = generator.generate_random_track(track_id)
        
        for gear in GEAR_RATIOS:
            for friction in TIRE_FRICTIONS:
                features, travel_time, success = collect_data_point(
                    client, xodr, track_id, gear, friction
                )
                
                if success:
                    dataset.append({
                        'track_id': track_id,
                        'gear_ratio': gear,
                        'tire_friction': friction,
                        'travel_time': travel_time,
                        **features
                    })
    
    # Save
    with open(f'{DATA_DIR}/dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"\n✓ Collected {len(dataset)} data points")
    return dataset


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(dataset):
    """Train regression model"""
    print("\n" + "="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    # Prepare data
    exclude = {'track_id', 'travel_time'}
    feature_names = [k for k in dataset[0].keys() if k not in exclude]
    
    X = np.array([[d[k] for k in feature_names] for d in dataset])
    y = np.array([d['travel_time'] for d in dataset])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    
    print(f"\nResults:")
    print(f"  Train R²: {train_r2:.3f}")
    print(f"  Test R²: {test_r2:.3f}")
    print(f"  Test RMSE: {np.sqrt(mean_squared_error(y_test, test_pred)):.2f}s")
    
    # Feature importance
    importances = model.feature_importances_
    top = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:5]
    print(f"\nTop 5 Features:")
    for name, imp in top:
        print(f"  {name}: {imp:.3f}")
    
    # Save
    with open(f'{DATA_DIR}/model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'features': feature_names}, f)
    
    return model, scaler, feature_names


def find_optimal_params(model, scaler, feature_names, track_features):
    """Grid search for optimal parameters"""
    print("\n" + "="*60)
    print("FINDING OPTIMAL PARAMETERS")
    print("="*60)
    
    print(f"\nTrack: length={track_features['total_length']:.0f}m, "
          f"curves={track_features['tight_corners_pct']*100:.0f}%")
    
    best_time = float('inf')
    best_params = None
    
    # Grid search
    gear_grid = np.linspace(2.0, 4.0, 15)
    friction_grid = np.linspace(0.6, 1.3, 15)
    
    for gear in gear_grid:
        for friction in friction_grid:
            features_dict = {**track_features, 'gear_ratio': gear, 'tire_friction': friction}
            X = np.array([[features_dict[k] for k in feature_names]])
            X_scaled = scaler.transform(X)
            pred_time = model.predict(X_scaled)[0]
            
            if pred_time < best_time:
                best_time = pred_time
                best_params = (gear, friction)
    
    print(f"\nOptimal:")
    print(f"  Gear Ratio: {best_params[0]:.2f}")
    print(f"  Tire Friction: {best_params[1]:.2f}")
    print(f"  Predicted Time: {best_time:.1f}s")
    
    return best_params, best_time


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*60)
    print("TRACK-BASED PARAMETER OPTIMIZER V2")
    print("="*60)
    
    # Collect data
    dataset = collect_all_data()
    
    if len(dataset) < 10:
        print("\nERROR: Not enough data collected")
        return
    
    # Train model
    model, scaler, features = train_model(dataset)
    
    # Test on new track
    print("\n" + "="*60)
    print("TESTING ON NEW TRACK")
    print("="*60)
    
    client = carla.Client('localhost', 2000)
    generator = RandomTrackGenerator()
    test_xodr = generator.generate_random_track(999)
    
    params = carla.OpendriveGenerationParameters(2.0, 50.0, 1.0, 0.6, True, True, True)
    world = client.generate_opendrive_world(test_xodr, params)
    
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    
    waypoints = get_waypoints(world)
    test_features = extract_track_features(waypoints)
    
    if test_features:
        find_optimal_params(model, scaler, features, test_features)
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
