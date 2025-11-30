"""
Track Parameter Optimizer - WITH WORKING AUTOPILOT
Pre-initialize Traffic Manager to prevent world regeneration
"""

import carla
import numpy as np
import math
import time
import pickle
import json
import os
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

DATA_DIR = "track_optimization_data"
os.makedirs(DATA_DIR, exist_ok=True)

GEAR_RATIOS = [2.5, 3.0, 3.5, 4.0]
TIRE_FRICTIONS = [0.6, 0.8, 1.0, 1.2]
NUM_RANDOM_TRACKS = 20
TRACK_SEGMENTS = 20

from track_param_optimizer_robust import RandomTrackGenerator, TrackFeatureExtractor

def get_waypoints_safe(world):
    """Get waypoints"""
    map_obj = world.get_map()
    
    wp = None
    for lane_id in (-1, 1):
        for s0 in (1.0, 5.0):
            try:
                wp = map_obj.get_waypoint_xodr(0, lane_id, s0)
                if wp:
                    break
            except:
                pass
        if wp:
            break
    
    if not wp:
        return []
    
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


def test_one_config(world, traffic_manager, waypoints, features, track_id, gear, friction):
    """Test ONE config - WITH AUTOPILOT"""
    print(f"    g={gear:.1f} f={friction:.1f}", end=" ", flush=True)
    
    vehicle = None
    
    try:
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        
        # Spawn
        sp = waypoints[5].transform
        sp.location.z += 2.0
        vehicle = world.spawn_actor(vehicle_bp, sp)
        
        # Settle
        vehicle.set_simulate_physics(False)
        world.tick()
        sp.location.z = waypoints[5].transform.location.z + 0.5
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
        
        # Enable autopilot with pre-initialized Traffic Manager
        tm_port = traffic_manager.get_port()
        vehicle.set_autopilot(True, tm_port)
        
        # Configure TM for this vehicle
        traffic_manager.ignore_lights_percentage(vehicle, 100)
        traffic_manager.ignore_signs_percentage(vehicle, 100)
        traffic_manager.ignore_vehicles_percentage(vehicle, 100)
        traffic_manager.distance_to_leading_vehicle(vehicle, 0)
        traffic_manager.vehicle_percentage_speed_difference(vehicle, -50)  # Go faster
        
        for _ in range(10):
            world.tick()
        
        # Drive
        t0 = time.time()
        start_loc = vehicle.get_location()
        stuck = 0
        done = False
        
        for step in range(2400):
            world.tick()
            
            loc = vehicle.get_location()
            vel = vehicle.get_velocity()
            dist = math.sqrt((loc.x - start_loc.x)**2 + (loc.y - start_loc.y)**2)
            speed = math.sqrt(vel.x**2 + vel.y**2)
            
            if speed < 0.5:
                stuck += 1
                if stuck > 100:
                    vehicle.set_autopilot(False, tm_port)
                    world.tick()
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
        
        vehicle.set_autopilot(False, tm_port)
        world.tick()
        vehicle.destroy()
        world.tick()
        
        if not done:
            print("TIMEOUT")
            return None
        
        print(f"{elapsed:.1f}s ✓")
        
        return {
            'track_id': track_id,
            'gear_ratio': gear,
            'tire_friction': friction,
            'travel_time': elapsed,
            **features
        }
        
    except Exception as e:
        print(f"ERR: {e}")
        if vehicle:
            try:
                vehicle.set_autopilot(False)
                vehicle.destroy()
                world.tick()
            except:
                pass
        return None


def collect_data():
    """Collect all data"""
    ckpt = f'{DATA_DIR}/dataset_checkpoint.pkl'
    
    if os.path.exists(ckpt):
        with open(ckpt, 'rb') as f:
            dataset = pickle.load(f)
        done = set(d['track_id'] for d in dataset)
        print(f"✓ Resume: {len(dataset)} points, tracks {sorted(done)}")
    else:
        dataset = []
        done = set()
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(30.0)
    
    # INITIALIZE TRAFFIC MANAGER ONCE AT START
    print("Initializing Traffic Manager...", end=" ", flush=True)
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(0.0)
    print("done")
    
    gen = RandomTrackGenerator()
    
    print("="*60)
    print("COLLECTING DATA")
    print("="*60)
    
    for tid in range(NUM_RANDOM_TRACKS):
        if tid in done:
            print(f"\nTrack {tid+1}/{NUM_RANDOM_TRACKS} SKIP")
            continue
        
        print(f"\nTrack {tid+1}/{NUM_RANDOM_TRACKS}")
        
        xodr = gen.generate_random_track(tid)
        
        print("  Generating world...", end=" ", flush=True)
        params = carla.OpendriveGenerationParameters(2.0, 50.0, 1.0, 0.6, True, True, True)
        world = client.generate_opendrive_world(xodr, params)
        print("done")
        
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        print("  Getting waypoints...", end=" ", flush=True)
        wps = get_waypoints_safe(world)
        print(f"{len(wps)} waypoints")
        
        if len(wps) < 10:
            print("  ERROR: too few waypoints")
            continue
        
        print("  Extracting features...", end=" ", flush=True)
        ext = TrackFeatureExtractor()
        feats = ext.extract_features(wps)
        print("done")
        
        if not feats:
            print("  ERROR: no features")
            continue
        
        print(f"  {feats['total_length']:.0f}m, {feats['tight_corners_pct']*100:.0f}% curves")
        
        results = []
        for gear in GEAR_RATIOS:
            for friction in TIRE_FRICTIONS:
                res = test_one_config(world, traffic_manager, wps, feats, tid, gear, friction)
                if res:
                    results.append(res)
        
        dataset.extend(results)
        
        with open(ckpt, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"  ✓ {len(results)} results ({len(dataset)} total)")
    
    with open(f'{DATA_DIR}/dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    with open(f'{DATA_DIR}/dataset.json', 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n✓ {len(dataset)} points collected")
    return dataset


def train_model(dataset):
    """Train model"""
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    exclude = {'track_id', 'travel_time'}
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
    print("WITH AUTOPILOT (Pre-initialized TM)")
    print("="*60)
    
    dataset = collect_data()
    
    if len(dataset) < 10:
        print("ERROR: not enough data")
        return
    
    train_model(dataset)
    
    print("\n✓ DONE")


if __name__ == "__main__":
    main()
