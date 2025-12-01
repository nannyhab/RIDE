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

GEAR_RATIOS = [1.5, 2.5, 4.0, 6.0]
TIRE_FRICTIONS = [0.5, 1.0, 2.0, 3.0]
TRACK_SEGMENTS = 20

# ========================================
# CONFIGURATION FLAGS
# ========================================
OBEY_TRAFFIC_RULES = False        # Overridden per scenario
NO_RENDERING_MODE = False          # True = no graphics (faster), False = show graphics
AUTOPILOT_SPEED_BOOST = -150      # Percentage faster than speed limit
PREVIEW_MODE = True               # True = only test 1 config per scenario
                                   # False = test all 16 configs
# ========================================


class MapRouteSelector:
    def __init__(self):
        self.scenarios = [
            # TOWN01 - NO stop signs (only traffic lights)
            {'id': 0, 'map': 'Town01', 'start_idx': 0, 'end_idx': 50, 
             'type': 'straight_cruise', 'complexity': 'easy',
             'obey_traffic': False},
            # Simulates: Highway cruising, no stops
            
            {'id': 1, 'map': 'Town01', 'start_idx': 15, 'end_idx': 75, 
             'type': 'traffic_light_stops', 'complexity': 'medium',
             'obey_traffic': True, 'obey_lights': True},
            # Simulates: Stop-and-go at TRAFFIC LIGHTS (Town01 has no stop signs)
            
            # TOWN02 - NO stop signs (only traffic lights)
            {'id': 2, 'map': 'Town02', 'start_idx': 5, 'end_idx': 65, 
             'type': 'gentle_curves', 'complexity': 'easy',
             'obey_traffic': False},
            # Simulates: Suburban curves
            
            {'id': 3, 'map': 'Town02', 'start_idx': 40, 'end_idx': 10, 
             'type': 'tight_residential', 'complexity': 'medium',
             'obey_traffic': False},
            # Simulates: Tight turns
            
            # TOWN03 - 22 STOP SIGNS! 🛑
            {'id': 4, 'map': 'Town03', 'start_idx': 0, 'end_idx': 80, 
             'type': 'highway_straight', 'complexity': 'easy',
             'obey_traffic': False},
            # Simulates: Highway, high speed
            
            {'id': 5, 'map': 'Town03', 'start_idx': 50, 'end_idx': 130, 
             'type': 'STOP_SIGNS_navigation', 'complexity': 'medium',
             'obey_traffic': True, 'obey_signs': True},
            # Simulates: ACTUAL STOP SIGN stops! Town03 has 22 stop signs
            
            {'id': 6, 'map': 'Town03', 'start_idx': 123, 'end_idx': 55, 
             'type': 'roundabout_actual', 'complexity': 'hard',
             'obey_traffic': False},
            # Simulates: Roundabout navigation
            
            # TOWN04 - 10 STOP SIGNS + Highway ramps 🛑
            {'id': 7, 'map': 'Town04', 'start_idx': 0, 'end_idx': 60, 
             'type': 'highway_cruise', 'complexity': 'easy',
             'obey_traffic': False},
            # Simulates: Multi-lane highway
            
            {'id': 8, 'map': 'Town04', 'start_idx': 103, 'end_idx': 263, 
             'type': 'highway_on_ramp', 'complexity': 'medium',
             'obey_traffic': False},
            # Simulates: Highway on-ramp merge
            
            {'id': 9, 'map': 'Town04', 'start_idx': 272, 'end_idx': 100, 
             'type': 'highway_off_ramp', 'complexity': 'medium',
             'obey_traffic': False},
            # Simulates: Highway off-ramp exit
            
            {'id': 10, 'map': 'Town04', 'start_idx': 150, 'end_idx': 50, 
             'type': 'STOP_SIGNS_highway_exits', 'complexity': 'medium',
             'obey_traffic': True, 'obey_signs': True},
            # Simulates: STOP SIGNS near highway exits (Town04 has 10 stop signs)
            
            # TOWN05 - 34 STOP SIGNS (MOST!) 🛑🛑🛑
            {'id': 11, 'map': 'Town05', 'start_idx': 50, 'end_idx': 150, 
             'type': 'dense_city_grid', 'complexity': 'easy',
             'obey_traffic': False},
            # Simulates: Urban grid, fast
            
            {'id': 12, 'map': 'Town05', 'start_idx': 0, 'end_idx': 100, 
             'type': 'STOP_SIGNS_heavy', 'complexity': 'hard',
             'obey_traffic': True, 'obey_signs': True},
            # Simulates: MANY STOP SIGNS! Town05 has 34 stop signs - best for stop-and-go
            
            {'id': 13, 'map': 'Town05', 'start_idx': 145, 'end_idx': 25, 
             'type': 'STOP_SIGNS_downtown', 'complexity': 'hard',
             'obey_traffic': True, 'obey_signs': True},
            # Simulates: Downtown with MULTIPLE stop signs
            
            {'id': 14, 'map': 'Town05', 'start_idx': 100, 'end_idx': 200, 
             'type': 'boulevard_fast', 'complexity': 'medium',
             'obey_traffic': False},
            # Simulates: Fast boulevard, ignore signs
            
            # TOWN10HD - 9 STOP SIGNS + 1 YIELD 🛑
            {'id': 15, 'map': 'Town10HD', 'start_idx': 20, 'end_idx': 100, 
             'type': 'main_avenue', 'complexity': 'medium',
             'obey_traffic': False},
            # Simulates: Main city avenue
            
            {'id': 16, 'map': 'Town10HD', 'start_idx': 50, 'end_idx': 130, 
             'type': 'STOP_SIGNS_urban', 'complexity': 'hard',
             'obey_traffic': True, 'obey_signs': True},
            # Simulates: Urban STOP SIGNS (Town10HD has 9 stop signs)
            
            {'id': 17, 'map': 'Town10HD', 'start_idx': 75, 'end_idx': 155, 
             'type': 'narrow_streets', 'complexity': 'hard',
             'obey_traffic': False},
            # Simulates: Narrow streets, fast
            
            {'id': 18, 'map': 'Town10HD', 'start_idx': 90, 'end_idx': 10, 
             'type': 'STOP_SIGNS_intersections', 'complexity': 'hard',
             'obey_traffic': True, 'obey_signs': True},
            # Simulates: Complex intersections with STOP SIGNS
            
            {'id': 19, 'map': 'Town10HD', 'start_idx': 110, 'end_idx': 190, 
             'type': 'mixed_conditions', 'complexity': 'hard',
             'obey_traffic': False},
            # Simulates: Mixed: straights, turns, fast navigation
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
    map_obj = world.get_map()
    spawn_points = map_obj.get_spawn_points()
    
    if len(spawn_points) == 0:
        return None, None, []
    
    start_idx = start_spawn_idx % len(spawn_points)
    end_idx = end_spawn_idx % len(spawn_points)
    
    start_location = spawn_points[start_idx].location
    end_location = spawn_points[end_idx].location
    
    start_wp = map_obj.get_waypoint(start_location)
    end_wp = map_obj.get_waypoint(end_location)
    
    if not start_wp or not end_wp:
        return None, None, []
    
    waypoints = [start_wp]
    current_wp = start_wp
    
    for _ in range(300):
        next_wps = current_wp.next(3.0)
        if not next_wps:
            break
        current_wp = next_wps[0]
        waypoints.append(current_wp)
    
    return start_location, end_location, waypoints


class CollisionTracker:
    def __init__(self):
        self.collision_count = 0
        self.severe_collision = False
        self.last_collision_time = 0
        self.min_impulse_severe = 2000.0
    
    def on_collision(self, event):
        impulse = math.sqrt(
            event.normal_impulse.x**2 +
            event.normal_impulse.y**2 +
            event.normal_impulse.z**2
        )
        
        current_time = time.time()
        
        if current_time - self.last_collision_time > 0.5:
            self.collision_count += 1
            self.last_collision_time = current_time
        
        if impulse > self.min_impulse_severe:
            self.severe_collision = True
    
    def reset(self):
        self.collision_count = 0
        self.severe_collision = False
        self.last_collision_time = 0


class ComprehensiveMetrics:
    """Tracks ALL performance metrics including stops"""
    def __init__(self):
        self.speeds = []
        self.lateral_accels = []
        self.longitudinal_accels = []
        self.lateral_offsets = []
        self.steering_angles = []
        self.total_distance = 0.0
        self.prev_location = None
        self.prev_velocity = None
        self.stop_count = 0
        self.stopped_frames = 0
    
    def update(self, vehicle, waypoints):
        location = vehicle.get_location()
        velocity = vehicle.get_velocity()
        control = vehicle.get_control()
        
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        self.speeds.append(speed)
        
        # Count stops
        if speed < 0.5:
            self.stopped_frames += 1
            if self.stopped_frames == 10:
                self.stop_count += 1
        else:
            self.stopped_frames = 0
        
        if self.prev_location is not None:
            dist = math.sqrt(
                (location.x - self.prev_location.x)**2 +
                (location.y - self.prev_location.y)**2
            )
            self.total_distance += dist
        
        if len(waypoints) > 0:
            min_dist = float('inf')
            for wp in waypoints:
                dist = location.distance(wp.transform.location)
                if dist < min_dist:
                    min_dist = dist
            self.lateral_offsets.append(min_dist)
        
        self.steering_angles.append(abs(control.steer))
        
        if self.prev_velocity is not None:
            dv_long = speed - math.sqrt(
                self.prev_velocity.x**2 + 
                self.prev_velocity.y**2 + 
                self.prev_velocity.z**2
            )
            longitudinal_accel = dv_long * 20.0
            self.longitudinal_accels.append(longitudinal_accel)
            
            angular_velocity = vehicle.get_angular_velocity()
            lateral_accel = abs(angular_velocity.z * speed)
            self.lateral_accels.append(lateral_accel)
        
        self.prev_location = location
        self.prev_velocity = velocity
    
    def get_summary(self):
        if len(self.speeds) == 0:
            return {}
        
        summary = {
            'avg_speed_ms': np.mean(self.speeds),
            'max_speed_ms': np.max(self.speeds),
            'min_speed_ms': np.min(self.speeds),
            'speed_std': np.std(self.speeds),
            'avg_speed_kmh': np.mean(self.speeds) * 3.6,
            'max_speed_kmh': np.max(self.speeds) * 3.6,
            'actual_distance_m': self.total_distance,
            'stop_count': self.stop_count,
        }
        
        if len(self.lateral_accels) > 0:
            summary['avg_lateral_accel'] = np.mean(np.abs(self.lateral_accels))
            summary['max_lateral_accel'] = np.max(np.abs(self.lateral_accels))
        else:
            summary['avg_lateral_accel'] = 0.0
            summary['max_lateral_accel'] = 0.0
        
        if len(self.longitudinal_accels) > 0:
            summary['avg_longitudinal_accel'] = np.mean(np.abs(self.longitudinal_accels))
            summary['max_longitudinal_accel'] = np.max(np.abs(self.longitudinal_accels))
            summary['smoothness'] = 1.0 / (1.0 + np.std(self.longitudinal_accels))
        else:
            summary['avg_longitudinal_accel'] = 0.0
            summary['max_longitudinal_accel'] = 0.0
            summary['smoothness'] = 1.0
        
        if len(self.lateral_offsets) > 0:
            summary['avg_lateral_offset'] = np.mean(self.lateral_offsets)
            summary['max_lateral_offset'] = np.max(self.lateral_offsets)
        else:
            summary['avg_lateral_offset'] = 0.0
            summary['max_lateral_offset'] = 0.0
        
        if len(self.steering_angles) > 0:
            summary['avg_steering'] = np.mean(self.steering_angles)
            summary['max_steering'] = np.max(self.steering_angles)
        else:
            summary['avg_steering'] = 0.0
            summary['max_steering'] = 0.0
        
        return summary


def test_one_config(world, traffic_manager, start_loc, end_loc, waypoints, route_length, scenario, gear, friction, rendering_enabled):
    """Test with scenario-specific traffic rules"""
    print(f"    g={gear:.1f} f={friction:.1f}", end=" ", flush=True)
    
    vehicle = None
    collision_sensor = None
    spectator = None
    
    try:
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        
        if len(waypoints) < 2:
            print("ERR: not enough waypoints")
            return None
        
        start_waypoint = waypoints[0]
        road_yaw = start_waypoint.transform.rotation.yaw
        
        spawn_transform = carla.Transform(
            start_loc + carla.Location(z=2.0),
            carla.Rotation(pitch=0.0, yaw=road_yaw, roll=0.0)
        )
        
        vehicle = world.spawn_actor(vehicle_bp, spawn_transform)
        
        if rendering_enabled:
            spectator = world.get_spectator()
        
        collision_bp = bp_lib.find('sensor.other.collision')
        collision_tracker = CollisionTracker()
        collision_sensor = world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=vehicle
        )
        collision_sensor.listen(collision_tracker.on_collision)
        
        vehicle.set_simulate_physics(False)
        for _ in range(5):
            world.tick()
        
        final_spawn = carla.Transform(
            start_loc + carla.Location(z=0.3),
            carla.Rotation(pitch=0.0, yaw=road_yaw, roll=0.0)
        )
        vehicle.set_transform(final_spawn)
        
        if rendering_enabled and spectator:
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
        
        vehicle.set_simulate_physics(True)
        for _ in range(20):
            world.tick()
        
        phys = vehicle.get_physics_control()
        phys.max_rpm = 5000.0 * gear / 3.0
        
        if gear < 2.0:
            phys.mass = 2000.0
        elif gear > 5.0:
            phys.mass = 1200.0
        else:
            phys.mass = 1500.0
        
        for w in phys.wheels:
            w.tire_friction = friction
        
        vehicle.apply_physics_control(phys)
        
        for _ in range(10):
            world.tick()
        
        tm_port = traffic_manager.get_port()
        vehicle.set_autopilot(True, tm_port)
        traffic_manager.set_path(vehicle, [end_loc])
        
        # ========================================
        # SCENARIO-SPECIFIC TRAFFIC RULES
        # ========================================
        obey_traffic = scenario.get('obey_traffic', False)
        obey_signs = scenario.get('obey_signs', False)
        obey_lights = scenario.get('obey_lights', False)
        
        if obey_traffic:
            # Obey STOP SIGNS (not lights - too variable)
            traffic_manager.ignore_signs_percentage(vehicle, 0 if obey_signs else 100)
            traffic_manager.ignore_lights_percentage(vehicle, 0 if obey_lights else 100)
            traffic_manager.ignore_vehicles_percentage(vehicle, 100)
            traffic_manager.ignore_walkers_percentage(vehicle, 100)
            traffic_manager.vehicle_percentage_speed_difference(vehicle, 0)
        else:
            # Ignore everything
            traffic_manager.ignore_lights_percentage(vehicle, 100)
            traffic_manager.ignore_signs_percentage(vehicle, 100)
            traffic_manager.ignore_vehicles_percentage(vehicle, 100)
            traffic_manager.ignore_walkers_percentage(vehicle, 100)
            traffic_manager.vehicle_percentage_speed_difference(vehicle, AUTOPILOT_SPEED_BOOST)
        
        traffic_manager.distance_to_leading_vehicle(vehicle, 0.0)
        traffic_manager.auto_lane_change(vehicle, False)
        
        for _ in range(10):
            world.tick()
        
        metrics = ComprehensiveMetrics()
        
        sim_start_time = world.get_snapshot().timestamp.elapsed_seconds
        stuck_counter = 0
        completed = False
        
        for step in range(10000):
            world.tick()
            
            metrics.update(vehicle, waypoints)
            
            current_location = vehicle.get_location()
            velocity = vehicle.get_velocity()
            speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
            
            if rendering_enabled and spectator and step % 2 == 0:
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
            
            if collision_tracker.severe_collision:
                vehicle.set_autopilot(False, tm_port)
                world.tick()
                collision_sensor.stop()
                collision_sensor.destroy()
                vehicle.destroy()
                world.tick()
                print(f"SEVERE COLLISION ({collision_tracker.collision_count} total)")
                return None
            
            dist_to_end = current_location.distance(end_loc)
            
            if speed < 0.5:
                stuck_counter += 1
                timeout_threshold = 1000 if obey_traffic else 400
                if stuck_counter > timeout_threshold:
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
            
            if (dist_to_end < 25.0 and metrics.total_distance > route_length * 0.6) or \
               (metrics.total_distance > route_length * 0.85):
                completed = True
                break
        
        sim_end_time = world.get_snapshot().timestamp.elapsed_seconds
        sim_elapsed_time = sim_end_time - sim_start_time
        
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
        
        stop_info = f" stops={perf['stop_count']}" if obey_traffic else ""
        print(f"{sim_elapsed_time:.1f}s | {perf['actual_distance_m']:.0f}m | {perf['avg_speed_kmh']:.1f}km/h{stop_info} ✓")
        
        return {
            'track_id': scenario['id'],
            'map_name': scenario['map'],
            'route_type': scenario['type'],
            'complexity': scenario['complexity'],
            'obey_traffic': obey_traffic,
            'obey_signs': obey_signs,
            'obey_lights': obey_lights,
            'gear_ratio': gear,
            'tire_friction': friction,
            'travel_time': sim_elapsed_time,
            'collision_count': collision_tracker.collision_count,
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
    print("CONFIGURATION")
    print("="*60)
    print(f"Gear Ratios: {GEAR_RATIOS}")
    print(f"Tire Frictions: {TIRE_FRICTIONS}")
    print(f"NO_RENDERING_MODE: {NO_RENDERING_MODE}")
    print(f"AUTOPILOT_SPEED_BOOST: {AUTOPILOT_SPEED_BOOST}%")
    print(f"PREVIEW_MODE: {PREVIEW_MODE} {'(1 config per scenario)' if PREVIEW_MODE else '(all 16 configs)'}")
    print("Traffic Rules: Per-scenario")
    print("  - Town03: 22 stop signs")
    print("  - Town04: 10 stop signs")
    print("  - Town05: 34 stop signs (BEST!)")
    print("  - Town10HD: 9 stop signs + 1 yield")
    if not NO_RENDERING_MODE:
        print("CAMERA: Following vehicle")
    print("="*60)
    
    print("\nInitializing Traffic Manager...", end=" ", flush=True)
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(0.0)
    print("done\n")
    
    selector = MapRouteSelector()
    
    print("="*60)
    print("COLLECTING DATA")
    print("="*60)
    
    for tid in range(selector.get_total_scenarios()):
        if tid in done:
            print(f"\nScenario {tid+1}/{selector.get_total_scenarios()} SKIP")
            continue
        
        scenario = selector.get_scenario(tid)
        
        obey_str = ""
        if scenario.get('obey_signs', False):
            obey_str = " [OBEYS STOP SIGNS 🛑]"
        elif scenario.get('obey_lights', False):
            obey_str = " [OBEYS TRAFFIC LIGHTS 🚦]"
        
        print(f"\nScenario {tid+1}/{selector.get_total_scenarios()}: {scenario['map']} - {scenario['type']}{obey_str}")
        
        print(f"  Loading {scenario['map']}...", end=" ", flush=True)
        world = client.load_world(scenario['map'])
        print("done")
        
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        
        rendering_enabled = not NO_RENDERING_MODE
        
        if NO_RENDERING_MODE:
            settings.no_rendering_mode = True
        else:
            settings.no_rendering_mode = False
        
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
        
        if PREVIEW_MODE:
            print("  [PREVIEW MODE: Testing 1 config only]")
            gear = 2.5
            friction = 1.0
            
            res = test_one_config(
                world, traffic_manager,
                start_loc, end_loc, waypoints,
                feats['total_length'],
                scenario, gear, friction,
                rendering_enabled
            )
            
            if res:
                res.update(feats)
                dataset.append(res)
                print(f"  ✓ Preview complete ({len(dataset)} total)")
            else:
                print(f"  ✗ Preview failed - fix this scenario!")
            
        else:
            results = []
            for gear in GEAR_RATIOS:
                for friction in TIRE_FRICTIONS:
                    res = test_one_config(
                        world, traffic_manager,
                        start_loc, end_loc, waypoints,
                        feats['total_length'],
                        scenario, gear, friction,
                        rendering_enabled
                    )
                    if res:
                        res.update(feats)
                        results.append(res)
            
            dataset.extend(results)
            print(f"  ✓ {len(results)} results ({len(dataset)} total)")
        
        with open(ckpt, 'wb') as f:
            pickle.dump(dataset, f)
    
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
    
    if PREVIEW_MODE:
        print("PREVIEW MODE: Skipping model training (need full dataset)")
        return None, None, None
    
    exclude = {'track_id', 'map_name', 'route_type', 'complexity', 'travel_time', 'obey_traffic', 'obey_signs', 'obey_lights'}
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
    print("REAL STOP SIGNS IN CARLA!")
    if PREVIEW_MODE:
        print("MODE: Preview (1 config per scenario for debugging)")
    else:
        print("MODE: Full Collection (16 configs per scenario)")
    print("="*60)
    
    dataset = collect_data()
    
    if len(dataset) < 10:
        print("ERROR: not enough data")
        return
    
    train_model(dataset)
    
    print("\n✓ DONE")
    if PREVIEW_MODE:
        print("\nTo collect full dataset, set PREVIEW_MODE = False")


if __name__ == "__main__":
    main()
