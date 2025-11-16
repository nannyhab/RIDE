
# carla_env.py
# Plug‑and‑play CARLA environment for PPO co‑design (gear ratio + tire friction).
# Matches signatures used in your PPO loop:
#   reset(gear_norm, mu_norm)  -> obs
#   step(action, gear_norm, mu_norm) -> obs, reward, done, info
#
# Action: np.array([steer, throttle, brake]) in ranges [-1,1], [0,1], [0,1]
# Observation (default):
#   [speed_mps, dist_to_goal, lateral_error, heading_error, route_progress]
#
# Reward (time-optimal with safety penalties):
#   r = -dt - 50*collision - 5*offroad - 0.1*lat_err^2 - 0.05*heading_err^2
#   + 5 on arrival
#
# Dependencies: CARLA >= 0.9.13 recommended

import math
import time
from typing import List, Optional, Tuple

import numpy as np

try:
    import carla  # type: ignore
except Exception as e:
    carla = None

class CarlaPointToPointEnv:
    def __init__(
        self,
        client: "carla.Client",
        town: str = "Town03",
        dt: float = 0.05,
        seed: int = 42,
        start: Optional["carla.Transform"] = None,
        goal: Optional["carla.Location"] = None,
        route: Optional[List["carla.Location"]] = None,
        gear_min: float = 2.5,
        gear_max: float = 5.0,
        mu_min: float = 0.6,
        mu_max: float = 1.2,
        lateral_offroad_thresh: float = 3.0,
        arrival_thresh: float = 5.0,
        sync_mode: bool = True,
        vehicle_filter: str = "vehicle.tesla.model3",
        camera_visual: bool = False,
    ) -> None:
        assert carla is not None, "CARLA Python API not found. Please run inside a CARLA-enabled environment."
        self.client = client
        self.dt = dt
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.sync_mode = sync_mode
        self.gear_min, self.gear_max = gear_min, gear_max
        self.mu_min, self.mu_max = mu_min, mu_max
        self.lateral_offroad_thresh = lateral_offroad_thresh
        self.arrival_thresh = arrival_thresh
        self.vehicle_filter = vehicle_filter
        self.camera_visual = camera_visual

        self.world = self.client.load_world(town)
        self.map = self.world.get_map()

        # Set synchronous mode if requested
        settings = self.world.get_settings()
        settings.synchronous_mode = sync_mode
        settings.fixed_delta_seconds = dt if sync_mode else None
        self.world.apply_settings(settings)

        self.blueprint_lib = self.world.get_blueprint_library()

        # Route definition
        self.start_tf = start
        self.goal_loc = goal
        self.route = route  # list of Locations forming the centerline route
        self._build_default_route_if_needed()

        # Runtime actors
        self.vehicle: Optional["carla.Vehicle"] = None
        self.collision_sensor: Optional["carla.Actor"] = None
        self.cam_sensor: Optional["carla.Actor"] = None
        self.collision_flag = False

        # Episode/bookkeeping
        self.max_steps = int(60.0 / dt)  # 60 seconds budget by default
        self.current_step = 0
        self.design = (0.5, 0.5)  # normalized (gear, mu)

        # Spaces (Gym-like; keep simple numeric vectors)
        # obs: [speed, dist_to_goal, lateral_err, heading_err, route_progress]
        self.obs_dim = 5
        self.act_dim = 3

    # ------------------------ Public API (PPO expects these) ------------------------

    def reset(self, gear_norm: float, mu_norm: float) -> np.ndarray:
        """Spawn ego, apply design params, set route, return initial observation."""
        self._cleanup_actors()

        spawn_tf = self._pick_start()
        self.vehicle = self._spawn_vehicle(spawn_tf)

        self._attach_collision_sensor()

        if self.camera_visual:
            self._attach_camera()

        # Apply co-design parameters
        self.design = (float(gear_norm), float(mu_norm))
        gear_ratio = self._denorm(gear_norm, self.gear_min, self.gear_max)
        mu = self._denorm(mu_norm, self.mu_min, self.mu_max)
        self._apply_vehicle_physics(gear_ratio, mu)

        self.current_step = 0
        self.collision_flag = False

        # Initial observation
        obs = self._get_observation()
        return obs

    def step(self, action: np.ndarray, gear_norm: float, mu_norm: float) -> Tuple[np.ndarray, float, bool, dict]:
        """Apply action for one tick, compute reward & done, return new obs."""
        assert self.vehicle is not None, "Call reset() before step()"

        # Re-apply design if changed (no-op if same norms)
        if (gear_norm, mu_norm) != self.design:
            self.design = (float(gear_norm), float(mu_norm))
            gear_ratio = self._denorm(gear_norm, self.gear_min, self.gear_max)
            mu = self._denorm(mu_norm, self.mu_min, self.mu_max)
            self._apply_vehicle_physics(gear_ratio, mu)

        # Action mapping
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle = float(np.clip(action[1], 0.0, 1.0))
        brake = float(np.clip(action[2], 0.0, 1.0))
        control = carla.VehicleControl(steer=steer, throttle=throttle, brake=brake)
        self.vehicle.apply_control(control)

        # Tick the world
        if self.sync_mode:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        obs = self._get_observation()
        reward, done, info = self._compute_reward_done_info(obs)

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        return obs, float(reward), bool(done), info

    def render(self, mode="human"):
        return  # Visualization handled by UE window

    def close(self):
        self._cleanup_actors()

    # ------------------------ Core mechanics ------------------------

    def _denorm(self, x: float, lo: float, hi: float) -> float:
        return float(lo + np.clip(x, 0.0, 1.0) * (hi - lo))

    def _apply_vehicle_physics(self, gear_ratio: float, mu: float):
        """Apply gearbox ratio and tire friction to the ego vehicle."""
        assert self.vehicle is not None

        physics: carla.VehiclePhysicsControl = self.vehicle.get_physics_control()

        # Transmission: set a single forward gear with desired ratio (or scale existing gears)
        # Keep auto box on so CARLA shifts as needed (with single gear it stays in that ratio).
        physics.use_gear_autobox = True
        if len(physics.forward_gears) == 0:
            # Construct single gear
            gear = carla.GearPhysicsControl()
            gear.ratio = gear_ratio
            gear.down_ratio = 0.5
            gear.up_ratio = 0.8
            physics.forward_gears = [gear]
        else:
            # Scale or overwrite first gear ratio
            physics.forward_gears[0].ratio = gear_ratio

        # Tire friction: assign to all wheels
        wheels = physics.wheels
        for w in wheels:
            w.tire_friction = float(mu)
        physics.wheels = wheels

        self.vehicle.apply_physics_control(physics)

    def _pick_start(self) -> "carla.Transform":
        if self.start_tf is not None:
            return self.start_tf
        # Default: use the first spawn point
        spawns = self.map.get_spawn_points()
        assert len(spawns) > 0, "No spawn points available in map"
        return spawns[0]

    def _build_default_route_if_needed(self):
        if self.route is not None and len(self.route) > 1:
            return
        # Build a short straight route from start to a goal 200m ahead along the lane
        start_tf = self.start_tf or self._pick_start()
        start_wp = self.map.get_waypoint(start_tf.location, project_to_road=True)
        route = [start_wp.transform.location]
        dist = 0.0
        wp = start_wp
        while dist < 200.0:
            nxt = wp.next(5.0)  # 5m increments
            if not nxt:
                break
            wp = nxt[0]
            route.append(wp.transform.location)
            dist += 5.0
        if self.goal_loc is None:
            self.goal_loc = route[-1]
        self.route = route

    def _spawn_vehicle(self, spawn_tf: "carla.Transform") -> "carla.Vehicle":
        bp = self.blueprint_lib.find(self.vehicle_filter)
        if bp is None:
            # fallback to any vehicle
            bp = self.blueprint_lib.filter("vehicle.*")[0]
        # Randomize color (optional)
        if bp.has_attribute("color"):
            color = np.random.choice(bp.get_attribute("color").recommended_values)
            bp.set_attribute("color", color)
        veh = self.world.spawn_actor(bp, spawn_tf)
        return veh

    def _attach_collision_sensor(self):
        assert self.vehicle is not None
        col_bp = self.blueprint_lib.find("sensor.other.collision")
        col_tf = carla.Transform(carla.Location(x=0, y=0, z=0))
        self.collision_sensor = self.world.spawn_actor(col_bp, col_tf, attach_to=self.vehicle)
        self.collision_flag = False

        def _on_collision(event):
            self.collision_flag = True

        self.collision_sensor.listen(_on_collision)

    def _attach_camera(self):
        assert self.vehicle is not None
        cam_bp = self.blueprint_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "640")
        cam_bp.set_attribute("image_size_y", "360")
        cam_bp.set_attribute("fov", "90")
        cam_tf = carla.Transform(carla.Location(x=1.5, z=2.0))
        self.cam_sensor = self.world.spawn_actor(cam_bp, cam_tf, attach_to=self.vehicle)
        self.cam_sensor.listen(lambda image: None)  # no-op; user can override

    def _cleanup_actors(self):
        actors = [self.collision_sensor, self.cam_sensor, self.vehicle]
        for a in actors:
            if a is not None:
                try:
                    a.destroy()
                except Exception:
                    pass
        self.collision_sensor = None
        self.cam_sensor = None
        self.vehicle = None

    # ------------------------ Observation & reward ------------------------

    def _get_vehicle_state(self) -> Tuple[float, float, float, float, "carla.Transform"]:
        """Return speed (m/s), lateral error (m), heading error (rad), progress [0,1], and transform."""
        veh = self.vehicle
        assert veh is not None
        tf: carla.Transform = veh.get_transform()
        vel: carla.Vector3D = veh.get_velocity()
        speed = math.sqrt(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z)

        # Compute route progress & errors against nearest segment
        lat_err, head_err, progress = self._route_errors(tf)

        # Distance to goal
        dist_to_goal = tf.location.distance(self.goal_loc) if self.goal_loc is not None else 0.0

        return speed, lat_err, head_err, progress, tf

    def _get_observation(self) -> np.ndarray:
        speed, lat_err, head_err, progress, tf = self._get_vehicle_state()
        dist_to_goal = tf.location.distance(self.goal_loc) if self.goal_loc is not None else 0.0
        obs = np.array([
            speed,
            dist_to_goal,
            lat_err,
            head_err,
            progress,
        ], dtype=np.float32)
        return obs

    def _compute_reward_done_info(self, obs: np.ndarray) -> Tuple[float, bool, dict]:
        speed, dist_to_goal, lat_err, head_err, progress = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3]), float(obs[4])

        # Base: minimize time
        reward = -self.dt

        # Penalties
        if self.collision_flag:
            reward -= 50.0
        reward -= 0.1 * (lat_err ** 2)
        reward -= 0.05 * (head_err ** 2)

        # Offroad heuristic: large lateral error
        offroad = abs(lat_err) > self.lateral_offroad_thresh
        if offroad:
            reward -= 5.0

        arrived = dist_to_goal < self.arrival_thresh
        if arrived:
            reward += 5.0

        done = bool(self.collision_flag or arrived)

        info = {
            "arrived": arrived,
            "collision": bool(self.collision_flag),
            "offroad": bool(offroad),
            "progress": progress,
            "dist_to_goal": dist_to_goal,
        }
        return reward, done, info

    def _route_errors(self, tf: "carla.Transform") -> Tuple[float, float, float]:
        """Compute lateral error, heading error, and normalized progress along the route polyline."""
        assert self.route is not None and len(self.route) >= 2

        # Find closest segment
        px = np.array([p.x for p in self.route], dtype=np.float32)
        py = np.array([p.y for p in self.route], dtype=np.float32)
        pz = np.array([p.z for p in self.route], dtype=np.float32)

        loc = tf.location
        pos = np.array([loc.x, loc.y, loc.z], dtype=np.float32)

        # Project onto each segment and keep best
        best_s = 0.0
        best_lat = 0.0
        best_i = 0
        best_d2 = 1e12

        seg_cum = [0.0]
        for i in range(1, len(px)):
            seg_cum.append(seg_cum[-1] + math.sqrt((px[i]-px[i-1])**2 + (py[i]-py[i-1])**2 + (pz[i]-pz[i-1])**2))
        total_len = seg_cum[-1]

        for i in range(len(px)-1):
            a = np.array([px[i], py[i], pz[i]], dtype=np.float32)
            b = np.array([px[i+1], py[i+1], pz[i+1]], dtype=np.float32)
            ab = b - a
            t = float(np.clip(np.dot(pos - a, ab) / (np.dot(ab, ab) + 1e-6), 0.0, 1.0))
            proj = a + t * ab
            d2 = float(np.dot(pos - proj, pos - proj))
            if d2 < best_d2:
                best_d2 = d2
                best_i = i
                # lateral signed error in horizontal plane using left normal
                heading = math.atan2(ab[1], ab[0])
                # left normal
                nx, ny = -math.sin(heading), math.cos(heading)
                lat = (pos[0] - proj[0]) * nx + (pos[1] - proj[1]) * ny
                best_lat = float(lat)
                best_s = seg_cum[i] + t * float(np.linalg.norm(ab))

        # Heading error (difference between vehicle yaw and route tangent)
        yaw = math.radians(tf.rotation.yaw)
        seg_a = np.array([px[best_i], py[best_i]], dtype=np.float32)
        seg_b = np.array([px[best_i+1], py[best_i+1]], dtype=np.float32)
        tangent = math.atan2(seg_b[1]-seg_a[1], seg_b[0]-seg_a[0])
        head_err = float(_wrap_angle(yaw - tangent))

        progress = float(np.clip(best_s / max(total_len, 1e-3), 0.0, 1.0))
        return best_lat, head_err, progress


def _wrap_angle(a: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (a + math.pi) % (2*math.pi) - math.pi
