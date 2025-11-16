"""
CARLA Reinforcement Learning Environment - Improved Version
Integrates XODR track generation and better spawn/completion logic
"""

import carla
import numpy as np
import math
import time
import datetime
from collections import deque

class CarlaRLEnvironment:
    """
    Enhanced CARLA RL Environment with XODR track generation
    """
    
    def __init__(self, 
                 host='localhost', 
                 port=2000,
                 difficulty=0,  # 0=straight, 1=circle, 2=S, 3=tight S, 4=double S
                 lane_width=3.5,
                 render_display=True):
        """
        Initialize CARLA RL Environment
        
        Args:
            host: CARLA server host
            port: CARLA server port
            difficulty: Track difficulty (0-4)
            lane_width: Track width in meters
            render_display: Whether to show visualization
        """
        # Connect to CARLA
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        
        # Generate custom track world
        self.difficulty = difficulty
        self.lane_width = lane_width
        self.world = self._generate_track_world()
        self.map = self.world.get_map()
        
        # Settings
        self.render_display = render_display
        
        # Vehicle and sensors
        self.vehicle = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        
        # Track waypoints
        self.waypoints = self._generate_dense_centerline()
        self._draw_track_visualization()
        
        # Convert to numpy arrays for fast computation
        self.track_x = np.array([wp.transform.location.x for wp in self.waypoints])
        self.track_y = np.array([wp.transform.location.y for wp in self.waypoints])
        self.track_z = np.array([wp.transform.location.z for wp in self.waypoints])
        
        # Episode tracking
        self.episode_start_time = 0
        self.episode_step = 0
        self.max_steps = 2000
        
        # Collision tracking
        self.collision_history = []
        self.lane_invasion_history = []
        
        # Reward tracking
        self.total_reward = 0
        self.reward_history = deque(maxlen=100)
        
        # Performance tracking
        self.best_lap_time = float('inf')
        self.lap_times = []
        self.speeds = deque(maxlen=50)
        
        # Previous state for delta calculations
        self.prev_checkpoint_idx = 0
        self.prev_distance_along_track = 0.0
        
        # Gate for lap completion
        self.start_gate_idx = min(10, len(self.waypoints) // 10)  # Start 10 waypoints in
        self.gate_A, self.gate_B = self._create_start_gate()
        self.prev_gate_side = None
        self.lap_started = False
        
    def _generate_track_world(self):
        """Generate XODR world based on difficulty"""
        xodr = self._create_xodr_for_difficulty(self.difficulty, self.lane_width)
        
        params = carla.OpendriveGenerationParameters(
            vertex_distance=2.0,
            max_road_length=50.0,
            wall_height=1.0,
            additional_width=0.6,
            smooth_junctions=True,
            enable_mesh_visibility=True,
            enable_pedestrian_navigation=True
        )
        
        world = self.client.generate_opendrive_world(xodr, params)
        
        # Set good weather
        weather = carla.WeatherParameters(
            cloudiness=35.0,
            precipitation=0.0,
            sun_azimuth_angle=220.0,
            sun_altitude_angle=35.0,
            fog_density=0.0,
            wetness=0.0
        )
        world.set_weather(weather)
        
        # Enable sync mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        return world
    
    def _create_xodr_for_difficulty(self, diff, lane_width):
        """Create XODR string for different difficulty levels"""
        header = f'''<?xml version="1.0" standalone="yes"?>
<OpenDRIVE revMajor="1" revMinor="4" name="track" version="1.5">
  <header revMajor="1" revMinor="4" name="track" version="1.5"
          date="{datetime.datetime.utcnow().isoformat()}" 
          north="0" south="0" east="0" west="0" vendor="generated"/>'''
        
        # Define track segments based on difficulty
        if diff == 0:  # Straight
            segments = [{"type": "line", "L": 200.0}]
            x0, y0, hdg0 = 0.0, 0.0, 0.0
        elif diff == 1:  # Circle
            R = 90.0
            L = 2.0 * math.pi * R
            segments = [{"type": "arc", "L": L, "k": 1.0/R}]
            x0, y0, hdg0 = 0.0, R, -math.pi/2
        elif diff == 2:  # Gentle S
            R = 80.0
            segments = [
                {"type": "arc", "L": 60.0, "k": 1.0/R},
                {"type": "line", "L": 30.0},
                {"type": "arc", "L": 60.0, "k": -1.0/R},
            ]
            x0, y0, hdg0 = 0.0, 0.0, 0.0
        elif diff == 3:  # Tight S
            R = 35.0
            segments = [
                {"type": "arc", "L": 55.0, "k": 1.0/R},
                {"type": "line", "L": 20.0},
                {"type": "arc", "L": 55.0, "k": -1.0/R},
            ]
            x0, y0, hdg0 = 0.0, 0.0, 0.0
        elif diff == 4:  # Double S
            R = 40.0
            segments = [
                {"type": "arc", "L": 55.0, "k": 1.0/R},
                {"type": "line", "L": 15.0},
                {"type": "arc", "L": 55.0, "k": -1.0/R},
                {"type": "line", "L": 15.0},
                {"type": "arc", "L": 55.0, "k": 1.0/R},
            ]
            x0, y0, hdg0 = 0.0, 0.0, 0.0
        else:
            raise ValueError("Difficulty must be 0-4")
        
        # Generate plan view
        planview, total_length = self._planview_from_segments(segments, x0, y0, hdg0)
        
        # Lane definition
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
            <width sOffset="0.0" a="{lane_width:.6f}" b="0" c="0" d="0"/>
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
        """Generate planView XML from segments"""
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
    
    def _generate_dense_centerline(self):
        """Generate dense waypoints along track centerline"""
        spacing = 0.5
        
        # Try to get waypoint at road 0, lane -1
        wp = None
        for lane_id in (-1, 1):
            for s0 in (0.5, 1.0, 2.0, 5.0, 10.0):
                try:
                    wp = self.map.get_waypoint_xodr(0, lane_id, s0)
                    if wp:
                        break
                except:
                    pass
            if wp:
                break
        
        if wp is None:
            # Fallback: spatial projection
            wp = self.map.get_waypoint(
                carla.Location(0, 0, 0),
                project_to_road=True,
                lane_type=carla.LaneType.Driving
            )
        
        if wp is None:
            raise RuntimeError("Could not find starting waypoint for track")
        
        # Walk forward collecting waypoints
        waypoints = []
        seen = set()
        max_steps = 20000
        
        for _ in range(max_steps):
            key = (int(wp.transform.location.x * 1000),
                   int(wp.transform.location.y * 1000),
                   wp.road_id, wp.lane_id)
            
            if key in seen:
                break
            
            seen.add(key)
            waypoints.append(wp)
            
            next_wps = wp.next(spacing)
            if not next_wps:
                break
            
            wp = next_wps[0]
        
        print(f"✓ Generated {len(waypoints)} waypoints for track")
        return waypoints
    
    def _draw_track_visualization(self):
        """Draw track with centerline and edges"""
        debug = self.world.debug
        life = 0.0  # Persistent
        
        GREEN = carla.Color(0, 255, 0)
        YELLOW = carla.Color(255, 255, 0)
        
        half_width = self.lane_width / 2.0
        
        for i in range(len(self.waypoints) - 1):
            wp = self.waypoints[i]
            wp_next = self.waypoints[i + 1]
            
            loc = wp.transform.location
            loc_next = wp_next.transform.location
            
            # Draw centerline (green)
            debug.draw_line(
                loc, loc_next,
                thickness=0.1, color=GREEN,
                life_time=life, persistent_lines=True
            )
            
            # Draw lane edges (yellow)
            fwd = wp.transform.get_forward_vector()
            right = carla.Vector3D(fwd.y, -fwd.x, 0)
            
            left_edge = loc + right * half_width
            right_edge = loc - right * half_width
            
            left_edge_next = loc_next + right * half_width
            right_edge_next = loc_next - right * half_width
            
            debug.draw_line(left_edge, left_edge_next,
                          thickness=0.07, color=YELLOW,
                          life_time=life, persistent_lines=True)
            debug.draw_line(right_edge, right_edge_next,
                          thickness=0.07, color=YELLOW,
                          life_time=life, persistent_lines=True)
    
    def _create_start_gate(self):
        """Create start/finish gate for lap detection"""
        idx = self.start_gate_idx
        wp = self.waypoints[idx]
        
        loc = wp.transform.location
        fwd = wp.transform.get_forward_vector()
        right = carla.Vector3D(fwd.y, -fwd.x, 0)
        
        gate_width = self.lane_width * 0.8
        gate_A = loc + right * (gate_width / 2.0)
        gate_B = loc - right * (gate_width / 2.0)
        
        # Draw gate
        RED = carla.Color(255, 0, 0)
        self.world.debug.draw_line(
            gate_A, gate_B,
            thickness=0.15, color=RED,
            life_time=0.0, persistent_lines=True
        )
        self.world.debug.draw_string(
            loc + carla.Location(z=2.0),
            "START/FINISH",
            draw_shadow=False,
            color=RED,
            life_time=0.0,
            persistent_lines=True
        )
        
        return gate_A, gate_B
    
    def reset(self):
        """Reset environment for new episode"""
        # Clean up
        self._cleanup()
        
        # Spawn vehicle
        self._spawn_vehicle()
        
        # Attach sensors
        self._attach_sensors()
        
        # Reset tracking
        self.episode_start_time = time.time()
        self.episode_step = 0
        self.total_reward = 0
        self.collision_history.clear()
        self.lane_invasion_history.clear()
        self.prev_checkpoint_idx = self.start_gate_idx
        self.prev_distance_along_track = 0.0
        self.lap_started = False
        
        # Initialize gate side
        loc = self.vehicle.get_location()
        ax, ay = self.gate_A.x, self.gate_A.y
        bx, by = self.gate_B.x, self.gate_B.y
        self.prev_gate_side = (bx - ax) * (loc.y - ay) - (by - ay) * (loc.x - ax)
        
        return self._get_state()
    
    def _spawn_vehicle(self):
        """Spawn vehicle with improved logic"""
        bp_lib = self.world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        
        # Spawn at start gate with correct orientation
        idx = self.start_gate_idx
        spawn_loc = self.waypoints[idx].transform.location + carla.Location(z=2.0)
        
        # Calculate heading from waypoint direction
        if idx < len(self.waypoints) - 1:
            dx = self.track_x[idx + 1] - self.track_x[idx]
            dy = self.track_y[idx + 1] - self.track_y[idx]
            yaw = math.degrees(math.atan2(dy, dx))
        else:
            yaw = self.waypoints[idx].transform.rotation.yaw
        
        spawn_transform = carla.Transform(
            spawn_loc,
            carla.Rotation(pitch=0, yaw=yaw, roll=0)
        )
        
        # Try spawning with increasing height
        for z_offset in [2.0, 5.0, 10.0]:
            try:
                spawn_transform.location.z = self.track_z[idx] + z_offset
                self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_transform)
                print(f"✓ Vehicle spawned (z_offset={z_offset}m, yaw={yaw:.1f}°)")
                break
            except RuntimeError:
                continue
        
        if self.vehicle is None:
            raise RuntimeError("Failed to spawn vehicle")
        
        # Let vehicle settle
        for _ in range(10):
            self.world.tick()
    
    def step(self, action):
        """Execute action and return state, reward, done, info"""
        self.episode_step += 1
        
        # Apply action
        throttle, steer = self._process_action(action)
        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=0.0
        )
        self.vehicle.apply_control(control)
        
        # Tick simulation
        self.world.tick()
        
        # Get state
        state = self._get_state()
        
        # Calculate reward
        reward = self._calculate_reward()
        self.total_reward += reward
        self.reward_history.append(reward)
        
        # Check done
        done, info = self._check_done()
        
        return state, reward, done, info
    
    def _process_action(self, action):
        """Process actions"""
        if len(action) == 2:
            throttle = np.clip((action[0] + 1) / 2, 0.0, 1.0)  # Convert [-1,1] to [0,1]
            steer = np.clip(action[1], -1.0, 1.0)
        else:
            throttle = np.clip(action[0], 0.0, 1.0)
            steer = np.clip(action[1], -1.0, 1.0)
        
        return throttle, steer
    
    def _get_state(self):
        """Get current state observation"""
        # Vehicle physics
        velocity = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        self.speeds.append(speed)
        
        # Position
        loc = self.vehicle.get_location()
        rot = self.vehicle.get_transform().rotation
        
        # Nearest waypoint
        dx = self.track_x - loc.x
        dy = self.track_y - loc.y
        distances = dx**2 + dy**2
        nearest_idx = int(np.argmin(distances))
        
        # Distance and angle to next waypoint
        next_idx = min(nearest_idx + 10, len(self.waypoints) - 1)
        target_x = self.track_x[next_idx]
        target_y = self.track_y[next_idx]
        
        distance_to_next = math.sqrt((target_x - loc.x)**2 + (target_y - loc.y)**2)
        angle_to_next = math.degrees(math.atan2(target_y - loc.y, target_x - loc.x)) - rot.yaw
        
        # Normalize angle to [-180, 180]
        while angle_to_next > 180:
            angle_to_next -= 360
        while angle_to_next < -180:
            angle_to_next += 360
        
        # Lateral offset from centerline
        lateral_offset = self._lateral_offset(loc, nearest_idx)
        
        # Progress
        progress = nearest_idx / len(self.waypoints)
        
        # State vector
        state = np.array([
            speed / 100.0,
            distance_to_next / 50.0,
            angle_to_next / 180.0,
            lateral_offset / 5.0,
            rot.yaw / 180.0,
            velocity.x / 50.0,
            velocity.y / 50.0,
            progress
        ], dtype=np.float32)
        
        return state
    
    def _lateral_offset(self, loc, idx):
        """Calculate lateral offset from track centerline"""
        if idx >= len(self.waypoints) - 1:
            idx = len(self.waypoints) - 2
        
        # Vector along track
        dx = self.track_x[idx + 1] - self.track_x[idx]
        dy = self.track_y[idx + 1] - self.track_y[idx]
        L = math.hypot(dx, dy) or 1e-6
        tx, ty = dx / L, dy / L
        
        # Vector from track to car
        cx = loc.x - self.track_x[idx]
        cy = loc.y - self.track_y[idx]
        
        # Cross product gives signed distance (left positive)
        return tx * cy - ty * cx
    
    def _calculate_reward(self):
        """Calculate reward"""
        reward = 0.0
        
        # Speed reward
        current_speed = self.speeds[-1] if self.speeds else 0.0
        speed_reward = current_speed / 30.0
        reward += speed_reward * 0.3
        
        # Progress reward
        loc = self.vehicle.get_location()
        dx = self.track_x - loc.x
        dy = self.track_y - loc.y
        nearest_idx = int(np.argmin(dx**2 + dy**2))
        
        if nearest_idx > self.prev_checkpoint_idx:
            progress_reward = (nearest_idx - self.prev_checkpoint_idx) * 2.0
            reward += progress_reward
        self.prev_checkpoint_idx = nearest_idx
        
        # Centerline reward
        lateral_offset = abs(self._lateral_offset(loc, nearest_idx))
        if lateral_offset < 1.0:
            reward += 1.0
        else:
            reward -= lateral_offset * 0.5
        
        # Collision penalty
        if self.collision_history:
            reward -= 50.0
            self.collision_history.clear()
        
        # Lane invasion penalty
        if self.lane_invasion_history:
            reward -= 10.0
            self.lane_invasion_history.clear()
        
        # Time penalty
        reward -= 0.1
        
        # Check gate crossing for lap completion
        if self._check_gate_crossing():
            reward += 100.0
            self.lap_times.append(time.time() - self.episode_start_time)
            print(f"  🏁 LAP COMPLETE! Time: {self.lap_times[-1]:.2f}s")
        
        return reward
    
    def _check_gate_crossing(self):
        """Check if vehicle crossed start/finish gate"""
        loc = self.vehicle.get_location()
        ax, ay = self.gate_A.x, self.gate_A.y
        bx, by = self.gate_B.x, self.gate_B.y
        
        # Calculate side of gate
        side = (bx - ax) * (loc.y - ay) - (by - ay) * (loc.x - ax)
        
        # Check if crossed (sign change)
        crossed = False
        if self.prev_gate_side is not None:
            if (side > 0) != (self.prev_gate_side > 0):
                # Also check we're close to the gate
                tx, ty = (bx - ax), (by - ay)
                tlen = math.sqrt(tx**2 + ty**2) or 1e-9
                dist_to_line = abs(side) / tlen
                
                if dist_to_line < self.lane_width * 0.5:
                    if self.lap_started:  # Only count after moving away first time
                        crossed = True
                    else:
                        self.lap_started = True
        
        self.prev_gate_side = side
        return crossed
    
    def _check_done(self):
        """Check if episode should end"""
        done = False
        info = {
            'episode_step': self.episode_step,
            'total_reward': self.total_reward,
            'lap_complete': len(self.lap_times) > 0,
            'collisions': len(self.collision_history),
        }
        
        # Max steps
        if self.episode_step >= self.max_steps:
            done = True
            info['termination_reason'] = 'max_steps'
        
        # Lap complete
        if len(self.lap_times) > 0:
            done = True
            info['termination_reason'] = 'lap_complete'
            info['lap_time'] = self.lap_times[-1]
        
        # Stuck
        if len(self.speeds) > 50 and np.mean(list(self.speeds)[-50:]) < 0.5:
            done = True
            info['termination_reason'] = 'stuck'
        
        # Off track
        loc = self.vehicle.get_location()
        dx = self.track_x - loc.x
        dy = self.track_y - loc.y
        nearest_idx = int(np.argmin(dx**2 + dy**2))
        lateral_offset = abs(self._lateral_offset(loc, nearest_idx))
        
        if lateral_offset > self.lane_width * 0.7:
            done = True
            info['termination_reason'] = 'off_track'
        
        return done, info
    
    def _attach_sensors(self):
        """Attach collision sensor"""
        bp_lib = self.world.get_blueprint_library()
        
        collision_bp = bp_lib.find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(
            collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )
        self.collision_sensor.listen(
            lambda event: self.collision_history.append(event)
        )
    
    def _cleanup(self):
        """Clean up actors"""
        if self.collision_sensor is not None:
            if self.collision_sensor.is_alive:
                self.collision_sensor.destroy()
            self.collision_sensor = None
        
        if self.vehicle is not None:
            if self.vehicle.is_alive:
                self.vehicle.destroy()
            self.vehicle = None
        
        self.world.tick()
    
    def close(self):
        """Close environment"""
        self._cleanup()


if __name__ == "__main__":
    print("Testing improved CARLA RL environment...")
    env = CarlaRLEnvironment(difficulty=0)  # Straight track
    state = env.reset()
    print(f"✓ Environment ready, state shape: {state.shape}")
    env.close()

    def set_physics_config(self, k_s, c_s):
        """Set suspension physics parameters"""
        if self.vehicle is not None:
            physics = self.vehicle.get_physics_control()
            
            for wheel in physics.wheels:
                wheel.suspension_stiffness = k_s
                wheel.damping_rate = c_s
            
            self.vehicle.apply_physics_control(physics)
            print(f"  Set physics: k_s={k_s}, c_s={c_s}")
