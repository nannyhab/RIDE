import carla
import numpy as np
import math
import time
import pickle
import json
import os
import sys
import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

DATA_DIR = "track_optimization_data_group1"
os.makedirs(DATA_DIR, exist_ok=True)

GEAR_RATIOS = [1.5, 2.5, 4.0, 6.0]
TIRE_FRICTIONS = [0.5, 1.0, 2.0, 3.0]
TRACK_SEGMENTS = 20

OBEY_TRAFFIC_RULES = False
NO_RENDERING_MODE = True
AUTOPILOT_SPEED_BOOST = -150
PREVIEW_MODE = True

TRAFFIC_LIGHT_GREEN_TIME = 3.0
TRAFFIC_LIGHT_YELLOW_TIME = 1.0
TRAFFIC_LIGHT_RED_TIME = 2.0


def configure_traffic_lights(world, green=None, yellow=None, red=None):
    if green is None:
        green = TRAFFIC_LIGHT_GREEN_TIME
    if yellow is None:
        yellow = TRAFFIC_LIGHT_YELLOW_TIME
    if red is None:
        red = TRAFFIC_LIGHT_RED_TIME
    
    lights = world.get_actors().filter('traffic.traffic_light')
    
    for light in lights:
        light.set_green_time(green)
        light.set_yellow_time(yellow)
        light.set_red_time(red)
        light.set_state(carla.TrafficLightState.Green)
    
    world.tick()
    
    if len(lights) > 0:
        print(f"    Configured {len(lights)} lights - FAST CYCLING ✓")


class MapRouteSelector:
    def __init__(self):
        self.scenarios = [
            # ========================================
            # TOWN01 SCENARIOS (4 scenarios)
            # ========================================
            {'id': 0, 'map': 'Town01', 'start_idx': 0, 'end_idx': 50, 
             'type': 'urban_straight_road', 'complexity': 'easy',
             'obey_traffic': False},
            
            {'id': 1, 'map': 'Town01', 'start_idx': 15, 'end_idx': 75, 
             'type': 'traffic_light_stop_go', 'complexity': 'medium',
             'obey_traffic': True, 'obey_lights': True},
            
            {'id': 2, 'map': 'Town01', 'start_idx': 30, 'end_idx': 90, 
             'type': 'urban_mixed_1', 'complexity': 'medium',
             'obey_traffic': False},
            
            {'id': 3, 'map': 'Town01', 'start_idx': 50, 'end_idx': 110, 
             'type': 'urban_mixed_2', 'complexity': 'medium',
             'obey_traffic': False},
            
            # ========================================
            # TOWN02 SCENARIOS (4 scenarios - S-CURVES!)
            # ========================================
            {'id': 4, 'map': 'Town02', 'start_idx': 30, 'end_idx': 70, 
             'type': 'residential_S_curves_1', 'complexity': 'medium',
             'obey_traffic': False},
            
            {'id': 5, 'map': 'Town02', 'start_idx': 5, 'end_idx': 65, 
             'type': 'residential_winding_1', 'complexity': 'medium',
             'obey_traffic': False},
            
            {'id': 6, 'map': 'Town02', 'start_idx': 50, 'end_idx': 10, 
             'type': 'residential_S_curves_2', 'complexity': 'medium',
             'obey_traffic': False},
            
            {'id': 7, 'map': 'Town02', 'start_idx': 20, 'end_idx': 80, 
             'type': 'residential_winding_2', 'complexity': 'medium',
             'obey_traffic': False},
            
            # ========================================
            # TOWN04 SCENARIOS (3 scenarios - ELEVATION!)
            # ========================================
            {'id': 8, 'map': 'Town04', 'start_idx': 0, 'end_idx': 200, 
             'type': 'highway_ELEVATION_route1', 'complexity': 'medium',
             'obey_traffic': False},
            
            {'id': 9, 'map': 'Town04', 'start_idx': 50, 'end_idx': 250, 
             'type': 'highway_ELEVATION_route2', 'complexity': 'medium',
             'obey_traffic': False},
            
            {'id': 10, 'map': 'Town04', 'start_idx': 100, 'end_idx': 300, 
             'type': 'highway_ELEVATION_route3', 'complexity': 'hard',
             'obey_traffic': False},
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
                # Calculate slope as elevation change / horizontal distance
                elevation_change = abs(seg_z[-1] - seg_z[0])
                horizontal_distance = end_s - start_s
                slope = elevation_change / max(horizontal_distance, 1e-6)
            else:
                slope = 0.0
            slopes.append(slope)
        
        curvature_array = np.array(curvatures)
        
        # Calculate total elevation change
        elevation_range = np.max(z) - np.min(z)
        max_elevation = np.max(z)
        min_elevation = np.min(z)
        
        features = {
            'total_length': total_length,
            'elevation_range': elevation_range,
            'max_elevation': max_elevation,
            'min_elevation': min_elevation,
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
        
        obey_traffic = scenario.get('obey_traffic', False)
        obey_signs = scenario.get('obey_signs', False)
        obey_lights = scenario.get('obey_lights', False)
        
        if obey_traffic:
            traffic_manager.ignore_signs_percentage(vehicle, 0 if obey_signs else 100)
            traffic_manager.ignore_lights_percentage(vehicle, 0 if obey_lights else 100)
            traffic_manager.ignore_vehicles_percentage(vehicle, 100)
            traffic_manager.ignore_walkers_percentage(vehicle, 100)
            traffic_manager.vehicle_percentage_speed_difference(vehicle, 0)
        else:
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


def collect_data(scenarios_to_run=None, preview_mode=True):
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
    total_scenarios = selector.get_total_scenarios()
    
    if scenarios_to_run is None:
        scenarios_to_run = list(range(total_scenarios))
    
    print("\n" + "="*60)
    print("CONFIGURATION - FINAL STABLE VERSION")
    print("="*60)
    print(f"Maps: Town01 (urban+lights), Town02 (S-curves), Town04 (elevation)")
    print(f"Scenarios to run: {scenarios_to_run}")
    print(f"Total scenarios: {total_scenarios}")
    print(f"Gear Ratios: {GEAR_RATIOS}")
    print(f"Tire Frictions: {TIRE_FRICTIONS}")
    print(f"Total data points: {len(scenarios_to_run)} × {'1' if preview_mode else '16'} = {len(scenarios_to_run) * (1 if preview_mode else 16)}")
    print(f"Traffic Lights: FAST CYCLING (G:{TRAFFIC_LIGHT_GREEN_TIME}s Y:{TRAFFIC_LIGHT_YELLOW_TIME}s R:{TRAFFIC_LIGHT_RED_TIME}s)")
    print("="*60)
    
    print("\nInitializing Traffic Manager...", end=" ", flush=True)
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(0.0)
    print("done\n")
    
    print("="*60)
    print("COLLECTING DATA")
    print("="*60)
    
    for tid in scenarios_to_run:
        if tid >= total_scenarios:
            print(f"\n⚠️  Scenario {tid} doesn't exist (max is {total_scenarios-1})")
            continue
            
        if tid in done:
            print(f"\nScenario {tid+1}/{total_scenarios} SKIP (already done)")
            continue
        
        scenario = selector.get_scenario(tid)
        
        obey_str = ""
        if scenario.get('obey_lights'):
            obey_str = " [STOP-AND-GO LIGHTS 🚦]"
        
        print(f"\nScenario {tid+1}/{total_scenarios}: {scenario['map']} - {scenario['type']}{obey_str}")
        
        print(f"  Loading {scenario['map']}...", end=" ", flush=True)
        world = client.load_world(scenario['map'])
        time.sleep(1)
        print("done")
        
        configure_traffic_lights(world)
        
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
        
        elev_str = f"elev={feats['elevation_range']:.1f}m" if feats.get('elevation_range', 0) > 1 else "flat"
        print(f"  {feats['total_length']:.0f}m, curves={feats['tight_corners_pct']*100:.0f}%, {elev_str}, max_slope={feats['max_slope']:.4f}")
        
        if preview_mode:
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
                print(f"  ✗ Preview failed")
            
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
    
    if len(dataset) < 10:
        print("Not enough data for training (need at least 10 points)")
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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Track Parameter Optimizer - FINAL VERSION',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--scenario', type=int, help='Run only scenario N (0-10)')
    parser.add_argument('--scenarios', type=str, help='Run multiple scenarios (comma-separated)')
    parser.add_argument('--full', action='store_true', help='Full mode: 16 configs per scenario')
    parser.add_argument('--render', action='store_true', help='Enable rendering')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    scenarios_to_run = None
    
    if args.scenario is not None:
        scenarios_to_run = [args.scenario]
    elif args.scenarios is not None:
        try:
            scenarios_to_run = [int(s.strip()) for s in args.scenarios.split(',')]
        except ValueError:
            print("ERROR: --scenarios must be comma-separated numbers")
            return
    
    preview_mode = not args.full
    
    global NO_RENDERING_MODE
    if args.render:
        NO_RENDERING_MODE = False
    
    print("="*60)
    print("TRACK PARAMETER OPTIMIZER - FINAL")
    print("11 scenarios using ONLY stable maps")
    print("Town01 (urban+lights) + Town02 (S-curves) + Town04 (elevation)")
    print(f"MODE: {'Preview (1 config/scenario)' if preview_mode else 'Full (16 configs/scenario)'}")
    print("="*60)
    
    dataset = collect_data(scenarios_to_run, preview_mode)
    
    if len(dataset) >= 10 and (not preview_mode or len(dataset) >= 50):
        train_model(dataset)
    
    print("\n✓ DONE")


if __name__ == "__main__":
    main()
