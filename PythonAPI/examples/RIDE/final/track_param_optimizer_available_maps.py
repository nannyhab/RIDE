"""
Track Parameter Optimizer - CARLA AUTOPILOT WITH FIXED ROUTES
Uses autopilot with set_destination() for consistent routes
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


class MapRouteSelector:
    def __init__(self):
        # Each scenario has a START and END spawn point index
        # This creates a consistent route for autopilot
        self.scenarios = [
            # Town01 - 3 different routes
            {'id': 0, 'map': 'Town01', 'start_idx': 0, 'end_idx': 50, 'type': 'route_1', 'complexity': 'easy'},
            {'id': 1, 'map': 'Town01', 'start_idx': 10, 'end_idx': 60, 'type': 'route_2', 'complexity': 'medium'},
            {'id': 2, 'map': 'Town01', 'start_idx': 20, 'end_idx': 70, 'type': 'route_3', 'complexity': 'medium'},
            
            # Town02 - 3 different routes
            {'id': 3, 'map': 'Town02', 'start_idx': 0, 'end_idx': 40, 'type': 'route_1', 'complexity': 'easy'},
            {'id': 4, 'map': 'Town02', 'start_idx': 15, 'end_idx': 55, 'type': 'route_2', 'complexity': 'medium'},
            {'id': 5, 'map': 'Town02', 'start_idx': 30, 'end_idx': 70, 'type': 'route_3', 'complexity': 'medium'},
            
            # Town03 - 3 different routes
            {'id': 6, 'map': 'Town03', 'start_idx': 0, 'end_idx': 80, 'type': 'route_1', 'complexity': 'easy'},
            {'id': 7, 'map': 'Town03', 'start_idx': 20, 'end_idx': 100, 'type': 'route_2', 'complexity': 'medium'},
            {'id': 8, 'map': 'Town03', 'start_idx': 40, 'end_idx': 120, 'type': 'route_3', 'complexity': 'hard'},
            
            # Town04 - 3 different routes
            {'id': 9, 'map': 'Town04', 'start_idx': 0, 'end_idx': 60, 'type': 'route_1', 'complexity': 'easy'},
            {'id': 10, 'map': 'Town04', 'start_idx': 25, 'end_idx': 85, 'type': 'route_2', 'complexity': 'medium'},
            {'id': 11, 'map': 'Town04', 'start_idx': 50, 'end_idx': 110, 'type': 'route_3', 'complexity': 'medium'},
            
            # Town05 - 3 different routes
            {'id': 12, 'map': 'Town05', 'start_idx': 0, 'end_idx': 100, 'type': 'route_1', 'complexity': 'easy'},
            {'id': 13, 'map': 'Town05', 'start_idx': 30, 'end_idx': 130, 'type': 'route_2', 'complexity': 'medium'},
            {'id': 14, 'map': 'Town05', 'start_idx': 60, 'end_idx': 160, 'type': 'route_3', 'complexity': 'hard'},
            
            # Town10HD - 5 different routes
            {'id': 15, 'map': 'Town10HD', 'start_idx': 0, 'end_idx': 80, 'type': 'route_1', 'complexity': 'medium'},
            {'id': 16, 'map': 'Town10HD', 'start_idx': 20, 'end_idx': 100, 'type': 'route_2', 'complexity': 'hard'},
            {'id': 17, 'map': 'Town10HD', 'start_idx': 40, 'end_idx': 120, 'type': 'route_3', 'complexity': 'hard'},
            {'id': 18, 'map': 'Town10HD', 'start_idx': 60, 'end_idx': 140, 'type': 'route_4', 'complexity': 'hard'},
            {'id': 19, 'map': 'Town10HD', 'start_idx': 80, 'end_idx': 160, 'type': 'route_5', 'complexity': 'hard'},
        ]
    
    def get_scenario(self, track_id):
        if track_id < len(self.scenarios):
            return self.scenarios[track_id]
        return None
    
    def get_total_scenarios(self):
        return len(self.scenarios)


class TrackFeatureExtractor:
    def __init__(self, num_segments=TRACK_SEGMENTS):
        self.num_segments = num_segments
    
    def extract_features(self, waypoints):
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


def get_planned_route(world, start_spawn_idx, end_spawn_idx):
    """
    Get the route autopilot will take from start to end
    Returns waypoints along the route for feature extraction
    """
    map_obj = world.get_map()
    spawn_points = map_obj.get_spawn_points()
    
    if len(spawn_points) == 0:
        return None, None, []
    
    # Get start and end points
    start_idx = start_spawn_idx % len(spawn_points)
    end_idx = end_spawn_idx % len(spawn_points)
    
    start_location = spawn_points[start_idx].location
    end_location = spawn_points[end_idx].location
    
    # Get waypoints along the route
    start_wp = map_obj.get_waypoint(start_location)
    end_wp = map_obj.get_waypoint(end_location)
    
    if not start_wp or not end_wp:
        return None, None, []
    
    # Sample waypoints for feature extraction
    # (Autopilot will navigate between start/end on its own)
    waypoints = [start_wp]
    current_wp = start_wp
    
    for _ in range(300):
        next_wps = current_wp.next(3.0)
        if not next_wps:
            break
        current_wp = next_wps[0]
        waypoints.append(current_wp)
    
    return start_location, end_location, waypoints


class SmartCollisionSensor:
    def __init__(self):
        self.hard_collision = False
        self.collision_count = 0
        self.min_impulse = 800.0  # Lower threshold - catch more collisions
    
    def on_collision(self, event):
        impulse = math.sqrt(
            event.normal_impulse.x**2 +
            event.normal_impulse.y**2 +
            event.normal_impulse.z**2
        )
        
        if impulse > self.min_impulse:
            self.hard_collision = True
            self.collision_count += 1
    
    def reset(self):
        self.hard_collision = False
        self.collision_count = 0


class PerformanceMetrics:
    def __init__(self):
        self.speeds = []
    
    def update(self, speed):
        self.speeds.append(speed)
    
    def get_summary(self):
        if len(self.speeds) == 0:
            return {}
        
        return {
            'avg_speed_ms': np.mean(self.speeds),
            'max_speed_ms': np.max(self.speeds),
            'avg_speed_kmh': np.mean(self.speeds) * 3.6,
            'max_speed_kmh': np.max(self.speeds) * 3.6,
        }


def test_one_config(world, traffic_manager, start_loc, end_loc, route_length, scenario, gear, friction):
    """Test with autopilot using fixed start/end destination"""
    print(f"    g={gear:.1f} f={friction:.1f}", end=" ", flush=True)
    
    vehicle = None
    collision_sensor = None
    
    try:
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        
        # Spawn at start location
        spawn_transform = carla.Transform(
            start_loc + carla.Location(z=2.0),
            carla.Rotation()
        )
        
        vehicle = world.spawn_actor(vehicle_bp, spawn_transform)
        
        # Collision sensor
        collision_bp = bp_lib.find('sensor.other.collision')
        collision_sensor_obj = SmartCollisionSensor()
        collision_sensor = world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=vehicle
        )
        collision_sensor.listen(collision_sensor_obj.on_collision)
        
        # Get spectator
        spectator = world.get_spectator()
        
        # Settle vehicle
        vehicle.set_simulate_physics(False)
        for _ in range(5):
            world.tick()
        
        final_spawn = carla.Transform(
            start_loc + carla.Location(z=0.3),
            carla.Rotation()
        )
        vehicle.set_transform(final_spawn)
        
        vehicle.set_simulate_physics(True)
        for _ in range(20):
            world.tick()
        
        # Apply physics parameters
        phys = vehicle.get_physics_control()
        phys.max_rpm = 5000.0 * gear / 3.0
        for w in phys.wheels:
            w.tire_friction = friction
        vehicle.apply_physics_control(phys)
        
        for _ in range(10):
            world.tick()
        
        # Enable autopilot with DESTINATION
        tm_port = traffic_manager.get_port()
        vehicle.set_autopilot(True, tm_port)
        
        # SET DESTINATION - This makes autopilot follow consistent route
        traffic_manager.set_path(vehicle, [end_loc])
        
        # Configure Traffic Manager
        traffic_manager.ignore_lights_percentage(vehicle, 100)
        traffic_manager.ignore_signs_percentage(vehicle, 100)
        traffic_manager.ignore_vehicles_percentage(vehicle, 100)
        traffic_manager.ignore_walkers_percentage(vehicle, 100)
        traffic_manager.distance_to_leading_vehicle(vehicle, 0.0)
        traffic_manager.vehicle_percentage_speed_difference(vehicle, -20)
        traffic_manager.auto_lane_change(vehicle, False)
        
        for _ in range(10):
            world.tick()
        
        # Performance tracking
        metrics = PerformanceMetrics()
        
        # Drive with autopilot
        start_time = time.time()
        start_location = vehicle.get_location()
        stuck_counter = 0
        completed = False
        
        for step in range(4000):  # Longer timeout
            world.tick()
            
            # Update camera every 5 frames
            if step % 5 == 0:
                vt = vehicle.get_transform()
                yaw_rad = math.radians(vt.rotation.yaw)
                camera_transform = carla.Transform(
                    carla.Location(
                        x=vt.location.x - 15.0 * math.cos(yaw_rad),
                        y=vt.location.y - 15.0 * math.sin(yaw_rad),
                        z=vt.location.z + 7.0
                    ),
                    carla.Rotation(pitch=-25.0, yaw=vt.rotation.yaw, roll=0.0)
                )
                spectator.set_transform(camera_transform)
            
            # Get vehicle state
            current_location = vehicle.get_location()
            velocity = vehicle.get_velocity()
            speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            # Update metrics
            metrics.update(speed)
            
            # Check collision
            if collision_sensor_obj.hard_collision:
                vehicle.set_autopilot(False, tm_port)
                world.tick()
                collision_sensor.stop()
                collision_sensor.destroy()
                vehicle.destroy()
                world.tick()
                print("COLLISION")
                return None
            
            # Calculate distance to destination
            dist_to_end = current_location.distance(end_loc)
            
            # Calculate distance traveled
            distance_traveled = math.sqrt(
                (current_location.x - start_location.x)**2 +
                (current_location.y - start_location.y)**2
            )
            
            # Stuck detection
            if speed < 0.5:
                stuck_counter += 1
                if stuck_counter > 250:
                    vehicle.set_autopilot(False, tm_port)
                    world.tick()
                    collision_sensor.stop()
                    collision_sensor.destroy()
                    vehicle.destroy()
                    world.tick()
                    print("STUCK")
                    return None
            else:
                stuck_counter = 0
            
            # Completion: reached destination OR traveled expected distance
            if dist_to_end < 15.0 or distance_traveled > route_length * 0.8:
                completed = True
                break
        
        elapsed_time = time.time() - start_time
        
        # Disable autopilot and cleanup
        vehicle.set_autopilot(False, tm_port)
        world.tick()
        collision_sensor.stop()
        collision_sensor.destroy()
        vehicle.destroy()
        world.tick()
        
        if not completed:
            print("TIMEOUT")
            return None
        
        perf = metrics.get_summary()
        
        print(f"{elapsed_time:.1f}s | {perf['avg_speed_kmh']:.1f}km/h ✓")
        
        return {
            'track_id': scenario['id'],
            'map_name': scenario['map'],
            'route_type': scenario['type'],
            'complexity': scenario['complexity'],
            'gear_ratio': gear,
            'tire_friction': friction,
            'travel_time': elapsed_time,
            **perf
        }
        
    except Exception as e:
        print(f"ERR: {e}")
        if vehicle:
            try:
                vehicle.set_autopilot(False)
                world.tick()
            except:
                pass
        if collision_sensor:
            try:
                collision_sensor.stop()
                collision_sensor.destroy()
            except:
                pass
        if vehicle:
            try:
                vehicle.destroy()
                world.tick()
            except:
                pass
        return None


def collect_data():
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
    
    print("\n" + "="*60)
    print("INITIALIZING TRAFFIC MANAGER")
    print("="*60)
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(0.0)
    print("✓ Traffic Manager ready\n")
    
    selector = MapRouteSelector()
    
    print("="*60)
    print("COLLECTING DATA - FIXED ROUTES")
    print("="*60)
    
    for tid in range(selector.get_total_scenarios()):
        if tid in done:
            print(f"\nScenario {tid+1}/{selector.get_total_scenarios()} SKIP")
            continue
        
        scenario = selector.get_scenario(tid)
        
        print(f"\nScenario {tid+1}/{selector.get_total_scenarios()}: {scenario['map']} - {scenario['type']}")
        
        print(f"  Loading {scenario['map']}...", end=" ", flush=True)
        world = client.load_world(scenario['map'])
        print("done")
        
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        print("  Planning route...", end=" ", flush=True)
        start_loc, end_loc, waypoints = get_planned_route(
            world,
            scenario['start_idx'],
            scenario['end_idx']
        )
        
        if not start_loc or not end_loc or len(waypoints) < 10:
            print("ERROR: invalid route")
            continue
        
        print(f"{len(waypoints)} waypoints")
        
        print("  Extracting features...", end=" ", flush=True)
        ext = TrackFeatureExtractor()
        feats = ext.extract_features(waypoints)
        print("done")
        
        if not feats:
            print("  ERROR: no features")
            continue
        
        print(f"  {feats['total_length']:.0f}m, {feats['tight_corners_pct']*100:.0f}% curves")
        
        results = []
        for gear in GEAR_RATIOS:
            for friction in TIRE_FRICTIONS:
                res = test_one_config(
                    world, traffic_manager,
                    start_loc, end_loc,
                    feats['total_length'],
                    scenario, gear, friction
                )
                if res:
                    # Add track features to result
                    res.update(feats)
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
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    exclude = {'track_id', 'map_name', 'route_type', 'complexity', 'travel_time'}
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
    top = sorted(zip(feat_names, importances), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 Features:")
    for name, imp in top:
        print(f"  {name}: {imp:.3f}")
    
    with open(f'{DATA_DIR}/model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler, 'features': feat_names}, f)
    
    return model, scaler, feat_names


def main():
    print("="*60)
    print("TRACK PARAMETER OPTIMIZER")
    print("CARLA AUTOPILOT - FIXED ROUTES")
    print("="*60)
    
    dataset = collect_data()
    
    if len(dataset) < 10:
        print("ERROR: not enough data")
        return
    
    train_model(dataset)
    
    print("\n✓ DONE")


if __name__ == "__main__":
    main()
