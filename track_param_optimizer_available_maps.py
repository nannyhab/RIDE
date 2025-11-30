"""
Track Parameter Optimizer - AVAILABLE CARLA MAPS
Uses 7 maps with multiple routes each = 20 diverse scenarios
"""

import carla
import numpy as np
import math
import time
import pickle
import json
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

DATA_DIR = "track_optimization_data_real_maps"
os.makedirs(DATA_DIR, exist_ok=True)

GEAR_RATIOS = [2.5, 3.0, 3.5, 4.0]
TIRE_FRICTIONS = [0.6, 0.8, 1.0, 1.2]
TRACK_SEGMENTS = 20


# ============================================================================
# MAP & ROUTE SELECTOR - 20 Scenarios from 7 Maps
# ============================================================================

class MapRouteSelector:
    """
    Create 20 diverse scenarios from available maps
    Each map tested with multiple different routes
    """
    
    def __init__(self):
        # Define 20 scenarios: (map_name, spawn_seed, route_type, complexity)
        self.scenarios = [
            # Town01 - Small town (3 routes)
            {'id': 0, 'map': 'Town01', 'seed': 0, 'type': 'town_main_street', 'complexity': 'easy'},
            {'id': 1, 'map': 'Town01', 'seed': 1, 'type': 'town_curves', 'complexity': 'medium'},
            {'id': 2, 'map': 'Town01', 'seed': 2, 'type': 'town_complex', 'complexity': 'medium'},
            
            # Town02 - Residential (3 routes)
            {'id': 3, 'map': 'Town02', 'seed': 0, 'type': 'residential_straight', 'complexity': 'easy'},
            {'id': 4, 'map': 'Town02', 'seed': 1, 'type': 'residential_turns', 'complexity': 'medium'},
            {'id': 5, 'map': 'Town02', 'seed': 2, 'type': 'residential_loop', 'complexity': 'medium'},
            
            # Town03 - Suburban (3 routes)
            {'id': 6, 'map': 'Town03', 'seed': 0, 'type': 'suburban_highway', 'complexity': 'easy'},
            {'id': 7, 'map': 'Town03', 'seed': 1, 'type': 'suburban_mixed', 'complexity': 'medium'},
            {'id': 8, 'map': 'Town03', 'seed': 2, 'type': 'suburban_roundabout', 'complexity': 'hard'},
            
            # Town04 - Highway (3 routes)
            {'id': 9, 'map': 'Town04', 'seed': 0, 'type': 'highway_straight', 'complexity': 'easy'},
            {'id': 10, 'map': 'Town04', 'seed': 1, 'type': 'highway_exits', 'complexity': 'medium'},
            {'id': 11, 'map': 'Town04', 'seed': 2, 'type': 'highway_merge', 'complexity': 'medium'},
            
            # Town05 - Urban (3 routes)
            {'id': 12, 'map': 'Town05', 'seed': 0, 'type': 'urban_highway', 'complexity': 'easy'},
            {'id': 13, 'map': 'Town05', 'seed': 1, 'type': 'urban_streets', 'complexity': 'medium'},
            {'id': 14, 'map': 'Town05', 'seed': 2, 'type': 'urban_downtown', 'complexity': 'hard'},
            
            # Town10HD - Dense city (5 routes - most complex)
            {'id': 15, 'map': 'Town10HD', 'seed': 0, 'type': 'city_main_avenue', 'complexity': 'medium'},
            {'id': 16, 'map': 'Town10HD', 'seed': 1, 'type': 'city_intersections', 'complexity': 'hard'},
            {'id': 17, 'map': 'Town10HD', 'seed': 2, 'type': 'city_tight_turns', 'complexity': 'hard'},
            {'id': 18, 'map': 'Town10HD', 'seed': 3, 'type': 'city_stop_and_go', 'complexity': 'hard'},
            {'id': 19, 'map': 'Town10HD', 'seed': 4, 'type': 'city_maximum_density', 'complexity': 'hard'},
        ]
    
    def get_scenario(self, track_id):
        """Get scenario info for track_id"""
        if track_id < len(self.scenarios):
            return self.scenarios[track_id]
        return None
    
    def get_total_scenarios(self):
        """Total number of scenarios"""
        return len(self.scenarios)


# ============================================================================
# TRACK FEATURE EXTRACTION
# ============================================================================

class TrackFeatureExtractor:
    """Extract numerical features from track waypoints"""
    
    def __init__(self, num_segments=TRACK_SEGMENTS):
        self.num_segments = num_segments
    
    def extract_features(self, waypoints):
        """Extract features from waypoints"""
        if len(waypoints) < 3:
            return None
        
        x = np.array([wp.transform.location.x for wp in waypoints])
        y = np.array([wp.transform.location.y for wp in waypoints])
        z = np.array([wp.transform.location.z for wp in waypoints])
        
        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)
        segment_lengths = np.sqrt(dx**2 + dy**2 + dz**2)
        total_length = segment_lengths.sum()
        
        cumulative_length = np.concatenate([[0], np.cumsum(segment_lengths)])
        target_lengths = np.linspace(0, total_length, self.num_segments + 1)
        
        curvatures = []
        slopes = []
        
        for i in range(self.num_segments):
            start_s = target_lengths[i]
            end_s = target_lengths[i + 1]
            
            mask = (cumulative_length >= start_s) & (cumulative_length <= end_s)
            if mask.sum() < 3:
                curvatures.append(0.0)
                slopes.append(0.0)
                continue
            
            seg_x = x[mask]
            seg_y = y[mask]
            seg_z = z[mask]
            
            curvature = self._compute_curvature(seg_x, seg_y)
            curvatures.append(curvature)
            
            if len(seg_z) > 1:
                slope = abs(seg_z[-1] - seg_z[0]) / max(end_s - start_s, 1e-6)
            else:
                slope = 0.0
            slopes.append(slope)
        
        curvature_array = np.array(curvatures)
        
        features = {
            'total_length': total_length,
            'avg_curvature': np.mean(np.abs(curvature_array)),
            'max_curvature': np.max(np.abs(curvature_array)),
            'std_curvature': np.std(curvature_array),
            'avg_slope': np.mean(slopes),
            'max_slope': np.max(slopes),
            'tight_corners_pct': np.sum(np.abs(curvature_array) > 0.02) / len(curvature_array),
            'straight_pct': np.sum(np.abs(curvature_array) < 0.005) / len(curvature_array),
            **{f'curvature_{i}': curvatures[i] for i in range(self.num_segments)},
            **{f'slope_{i}': slopes[i] for i in range(self.num_segments)},
        }
        
        return features
    
    def _compute_curvature(self, x, y):
        """Compute curvature from points"""
        if len(x) < 3:
            return 0.0
        
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        
        numerator = np.abs(dx * ddy - dy * ddx)
        denominator = np.power(dx**2 + dy**2, 1.5) + 1e-6
        
        curvature = numerator / denominator
        
        return np.mean(curvature)


# ============================================================================
# WAYPOINT EXTRACTION
# ============================================================================

def get_route_waypoints(world, seed=0, route_length=500):
    """
    Get a route through the map using seed for reproducibility
    """
    np.random.seed(seed)  # Reproducible routes
    
    map_obj = world.get_map()
    spawn_points = map_obj.get_spawn_points()
    
    if not spawn_points:
        return []
    
    # Pick spawn point based on seed
    spawn_idx = seed % len(spawn_points)
    start_transform = spawn_points[spawn_idx]
    start_wp = map_obj.get_waypoint(start_transform.location)
    
    if not start_wp:
        return []
    
    # Follow road forward
    waypoints = [start_wp]
    current_wp = start_wp
    distance = 0.0
    
    for _ in range(route_length):
        next_wps = current_wp.next(2.0)  # 2m spacing
        
        if not next_wps:
            break
        
        # If multiple choices, use seed to pick
        if len(next_wps) > 1:
            choice_idx = (seed + len(waypoints)) % len(next_wps)
            current_wp = next_wps[choice_idx]
        else:
            current_wp = next_wps[0]
        
        waypoints.append(current_wp)
        distance += 2.0
        
        if distance > 1000:  # Max 1km route
            break
    
    return waypoints


# ============================================================================
# CONTROLLER
# ============================================================================

def pure_pursuit_controller(vehicle, waypoints):
    """Pure pursuit controller"""
    tf = vehicle.get_transform()
    loc = tf.location
    
    min_dist = float('inf')
    nearest_idx = 0
    for i, wp in enumerate(waypoints):
        d = loc.distance(wp.transform.location)
        if d < min_dist:
            min_dist = d
            nearest_idx = i
    
    lookahead = 15
    target_idx = min(nearest_idx + lookahead, len(waypoints) - 1)
    target = waypoints[target_idx].transform.location
    
    yaw = math.radians(tf.rotation.yaw)
    dx = target.x - loc.x
    dy = target.y - loc.y
    
    local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
    local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
    
    ld = math.sqrt(local_x**2 + local_y**2) or 1e-6
    curvature = 2.0 * local_y / (ld * ld)
    wheelbase = 2.88
    steer = math.atan(wheelbase * curvature)
    steer = np.clip(steer / 0.7, -1.0, 1.0)
    
    vel = vehicle.get_velocity()
    speed = math.sqrt(vel.x**2 + vel.y**2)
    
    target_speed = 10.0
    
    if speed < target_speed * 0.7:
        throttle = 0.8
        brake = 0.0
    elif speed < target_speed:
        throttle = 0.5
        brake = 0.0
    elif speed < target_speed * 1.2:
        throttle = 0.3
        brake = 0.0
    else:
        throttle = 0.0
        brake = 0.3
    
    return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)


# ============================================================================
# TESTING
# ============================================================================

def test_one_config(world, waypoints, features, scenario, gear, friction):
    """Test ONE config on route"""
    print(f"    g={gear:.1f} f={friction:.1f}", end=" ", flush=True)
    
    vehicle = None
    
    try:
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        
        # Spawn at start of route
        sp = waypoints[0].transform
        sp.location.z += 2.0
        vehicle = world.spawn_actor(vehicle_bp, sp)
        
        # Settle
        vehicle.set_simulate_physics(False)
        world.tick()
        sp.location.z = waypoints[0].transform.location.z + 0.5
        vehicle.set_transform(sp)
        world.tick()
        vehicle.set_simulate_physics(True)
        for _ in range(15):
            world.tick()
        
        # Physics
        phys = vehicle.get_physics_control()
        phys.max_rpm = 5000.0 * gear / 3.0
        for w in phys.wheels:
            w.tire_friction = friction
        vehicle.apply_physics_control(phys)
        
        for _ in range(10):
            world.tick()
        
        # Drive
        t0 = time.time()
        start_loc = vehicle.get_location()
        stuck = 0
        done = False
        
        for step in range(2400):
            control = pure_pursuit_controller(vehicle, waypoints)
            vehicle.apply_control(control)
            
            world.tick()
            
            loc = vehicle.get_location()
            vel = vehicle.get_velocity()
            dist = math.sqrt((loc.x - start_loc.x)**2 + (loc.y - start_loc.y)**2)
            speed = math.sqrt(vel.x**2 + vel.y**2)
            
            if speed < 0.5:
                stuck += 1
                if stuck > 100:
                    vehicle.destroy()
                    world.tick()
                    print("STUCK")
                    return None
            else:
                stuck = 0
            
            if dist > features['total_length'] * 0.8:
                done = True
                break
        
        elapsed = time.time() - t0
        
        vehicle.destroy()
        world.tick()
        
        if not done:
            print("TIMEOUT")
            return None
        
        print(f"{elapsed:.1f}s ✓")
        
        return {
            'track_id': scenario['id'],
            'map_name': scenario['map'],
            'route_type': scenario['type'],
            'complexity': scenario['complexity'],
            'seed': scenario['seed'],
            'gear_ratio': gear,
            'tire_friction': friction,
            'travel_time': elapsed,
            **features
        }
        
    except Exception as e:
        print(f"ERR: {e}")
        if vehicle:
            try:
                vehicle.destroy()
                world.tick()
            except:
                pass
        return None


# ============================================================================
# DATA COLLECTION
# ============================================================================

def collect_data():
    """Collect data from all scenarios"""
    ckpt = f'{DATA_DIR}/dataset_checkpoint.pkl'
    
    if os.path.exists(ckpt):
        with open(ckpt, 'rb') as f:
            dataset = pickle.load(f)
        done = set(d['track_id'] for d in dataset)
        print(f"✓ Resume: {len(dataset)} points, scenarios {sorted(done)}")
    else:
        dataset = []
        done = set()
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(30.0)
    
    selector = MapRouteSelector()
    
    print("="*60)
    print("COLLECTING DATA: 20 SCENARIOS FROM 7 MAPS")
    print("="*60)
    
    for tid in range(selector.get_total_scenarios()):
        if tid in done:
            print(f"\nScenario {tid+1}/{selector.get_total_scenarios()} SKIP")
            continue
        
        scenario = selector.get_scenario(tid)
        
        print(f"\nScenario {tid+1}/{selector.get_total_scenarios()}: {scenario['map']} - {scenario['type']} ({scenario['complexity']})")
        
        # Load map
        print(f"  Loading {scenario['map']}...", end=" ", flush=True)
        world = client.load_world(scenario['map'])
        print("done")
        
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        # Get route using seed
        print("  Planning route...", end=" ", flush=True)
        wps = get_route_waypoints(world, seed=scenario['seed'], route_length=500)
        print(f"{len(wps)} waypoints")
        
        if len(wps) < 50:
            print("  ERROR: route too short")
            continue
        
        # Extract features
        print("  Extracting features...", end=" ", flush=True)
        ext = TrackFeatureExtractor()
        feats = ext.extract_features(wps)
        print("done")
        
        if not feats:
            print("  ERROR: no features")
            continue
        
        print(f"  {feats['total_length']:.0f}m, {feats['tight_corners_pct']*100:.0f}% curves")
        
        # Test all configs
        results = []
        for gear in GEAR_RATIOS:
            for friction in TIRE_FRICTIONS:
                res = test_one_config(world, wps, feats, scenario, gear, friction)
                if res:
                    results.append(res)
        
        dataset.extend(results)
        
        with open(ckpt, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"  ✓ {len(results)} results ({len(dataset)} total)")
    
    # Final save
    with open(f'{DATA_DIR}/dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    with open(f'{DATA_DIR}/dataset.json', 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n✓ {len(dataset)} points collected")
    return dataset


# ============================================================================
# TRAINING
# ============================================================================

def train_model(dataset):
    """Train model"""
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    exclude = {'track_id', 'map_name', 'route_type', 'complexity', 'seed', 'travel_time'}
    feat_names = [k for k in dataset[0].keys() if k not in exclude]
    
    X = np.array([[d[k] for k in feat_names] for d in dataset])
    y = np.array([d['travel_time'] for d in dataset])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X_train, y_train)
    
    train_r2 = r2_score(y_train, model.predict(X_train))
    test_r2 = r2_score(y_test, model.predict(X_test))
    
    print(f"Train R²: {train_r2:.3f}")
    print(f"Test R²: {test_r2:.3f}")
    
    importances = model.feature_importances_
    top = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)[:5]
    print("\nTop Features:")
    for name, imp in top:
        print(f"  {name}: {imp:.3f}")
    
    with open(f'{DATA_DIR}/model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'features': feat_names}, f)
    
    return model, scaler, feat_names


def main():
    print("="*60)
    print("TRACK PARAMETER OPTIMIZER")
    print("20 Scenarios from 7 Available Maps")
    print("="*60)
    
    dataset = collect_data()
    
    if len(dataset) < 10:
        print("ERROR: not enough data")
        return
    
    train_model(dataset)
    
    print("\n✓ DONE")


if __name__ == "__main__":
    main()
