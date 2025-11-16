"""
CARLA Reinforcement Learning Environment
This environment wraps CARLA simulator for RL training of autonomous racing agents.
"""

import carla
import numpy as np
import math
import time
from collections import deque

class CarlaRLEnvironment:
    """
    Custom CARLA environment for racing RL agent.
    Provides gym-like interface with state observations, actions, and rewards.
    """
    
    def __init__(self, 
                 host='localhost', 
                 port=2000, 
                 checkpoint_file='checkpoints.txt',
                 render_display=True):
        """
        Initialize CARLA RL Environment
        
        Args:
            host: CARLA server host
            port: CARLA server port
            checkpoint_file: Path to checkpoint waypoints file
            render_display: Whether to show visualization
        """
        # Connect to CARLA
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        
        # Environment settings
        self.render_display = render_display
        self.checkpoint_file = checkpoint_file
        
        # Vehicle and sensors
        self.vehicle = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.camera_sensor = None
        
        # Checkpoints and track info
        self.checkpoints = self._load_checkpoints()
        self.current_checkpoint = 0
        self.lap_complete = False
        
        # Episode tracking
        self.episode_start_time = 0
        self.episode_step = 0
        self.max_steps = 2000  # Maximum steps per episode
        
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
        self.prev_location = None
        self.prev_distance_to_checkpoint = None
        
    def _load_checkpoints(self):
        """Load track checkpoints from file or generate default ones"""
        try:
            with open(self.checkpoint_file, 'r') as f:
                checkpoints = []
                for line in f:
                    x, y, z = map(float, line.strip().split(','))
                    checkpoints.append(carla.Location(x=x, y=y, z=z))
                return checkpoints
        except FileNotFoundError:
            # Return default checkpoints (example oval track)
            print("Checkpoint file not found. Using default checkpoints.")
            return self._generate_default_checkpoints()
    
    def _generate_default_checkpoints(self):
        """Generate default oval track checkpoints"""
        checkpoints = []
        num_points = 20
        radius = 50.0
        center_x, center_y = 0.0, 0.0
        
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            checkpoints.append(carla.Location(x=x, y=y, z=0.5))
        
        return checkpoints
    
    def reset(self):
        """
        Reset environment for new episode
        
        Returns:
            Initial state observation
        """
        # Clean up existing actors
        self._cleanup()
        
        # Spawn vehicle at starting position
        self._spawn_vehicle()
        
        # Attach sensors
        self._attach_sensors()
        
        # Reset tracking variables
        self.current_checkpoint = 0
        self.lap_complete = False
        self.episode_start_time = time.time()
        self.episode_step = 0
        self.total_reward = 0
        self.collision_history.clear()
        self.lane_invasion_history.clear()
        self.prev_location = self.vehicle.get_location()
        self.prev_distance_to_checkpoint = self._get_distance_to_checkpoint()
        
        # Get initial state
        state = self._get_state()
        
        return state
    
    def step(self, action):
        """
        Execute action and return new state, reward, done flag, and info
        
        Args:
            action: [throttle, steer, brake] where values are in [-1, 1]
        
        Returns:
            state: Current observation
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information dictionary
        """
        self.episode_step += 1
        
        # Apply action to vehicle
        throttle, steer, brake = self._process_action(action)
        control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake)
        )
        self.vehicle.apply_control(control)
        
        # Wait for simulation step
        self.world.tick()
        
        # Get new state
        state = self._get_state()
        
        # Calculate reward
        reward = self._calculate_reward()
        self.total_reward += reward
        self.reward_history.append(reward)
        
        # Check if episode is done
        done, info = self._check_done()
        
        # Store current location for next step
        self.prev_location = self.vehicle.get_location()
        self.prev_distance_to_checkpoint = self._get_distance_to_checkpoint()
        
        return state, reward, done, info
    
    def _process_action(self, action):
        """
        Process and clip action values
        
        Args:
            action: Raw action from agent
        
        Returns:
            throttle, steer, brake (clipped to valid ranges)
        """
        if len(action) == 2:
            # [throttle, steer] - no brake
            throttle = np.clip(action[0], 0.0, 1.0)
            steer = np.clip(action[1], -1.0, 1.0)
            brake = 0.0
        else:
            # [throttle, steer, brake]
            throttle = np.clip(action[0], 0.0, 1.0)
            steer = np.clip(action[1], -1.0, 1.0)
            brake = np.clip(action[2], 0.0, 1.0)
        
        return throttle, steer, brake
    
    def _get_state(self):
        """
        Get current state observation
        
        Returns:
            State vector containing:
            - Speed (normalized)
            - Distance to next checkpoint
            - Angle to next checkpoint
            - Distance from track center
            - Previous actions/velocities
        """
        # Get vehicle physics
        velocity = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # km/h
        self.speeds.append(speed)
        
        # Get location and rotation
        transform = self.vehicle.get_transform()
        location = transform.location
        rotation = transform.rotation
        
        # Distance to next checkpoint
        distance_to_checkpoint = self._get_distance_to_checkpoint()
        
        # Angle to next checkpoint
        angle_to_checkpoint = self._get_angle_to_checkpoint(location, rotation)
        
        # Distance from track center (approximate)
        distance_from_center = self._get_distance_from_center(location)
        
        # Waypoint info
        waypoint = self.map.get_waypoint(location)
        
        # Construct state vector
        state = np.array([
            speed / 100.0,  # Normalize speed
            distance_to_checkpoint / 100.0,  # Normalize distance
            angle_to_checkpoint / 180.0,  # Normalize angle
            distance_from_center / 10.0,  # Normalize distance from center
            rotation.yaw / 180.0,  # Normalize yaw
            velocity.x / 50.0,  # Normalize velocities
            velocity.y / 50.0,
            self.current_checkpoint / len(self.checkpoints)  # Progress
        ], dtype=np.float32)
        
        return state
    
    def _calculate_reward(self):
        """
        Calculate reward for current step
        
        Reward components:
        + Speed reward (faster is better)
        + Progress toward checkpoint
        + Staying on track
        - Collisions
        - Going off track
        - Excessive steering
        """
        reward = 0.0
        
        # Speed reward
        if len(self.speeds) > 0:
            current_speed = self.speeds[-1]
            speed_reward = current_speed / 50.0  # Normalize by target speed
            reward += speed_reward * 0.5
        
        # Progress toward checkpoint
        current_distance = self._get_distance_to_checkpoint()
        if self.prev_distance_to_checkpoint is not None:
            distance_delta = self.prev_distance_to_checkpoint - current_distance
            progress_reward = distance_delta * 0.5
            reward += progress_reward
        
        # Check if checkpoint reached
        if current_distance < 5.0:  # Within 5 meters of checkpoint
            reward += 10.0  # Bonus for reaching checkpoint
            self.current_checkpoint = (self.current_checkpoint + 1) % len(self.checkpoints)
            
            # Check if lap completed
            if self.current_checkpoint == 0:
                lap_time = time.time() - self.episode_start_time
                self.lap_times.append(lap_time)
                if lap_time < self.best_lap_time:
                    self.best_lap_time = lap_time
                    reward += 50.0  # Big bonus for new best lap
                else:
                    reward += 20.0  # Bonus for completing lap
                self.lap_complete = True
        
        # Penalty for being far from track center
        location = self.vehicle.get_location()
        distance_from_center = self._get_distance_from_center(location)
        if distance_from_center > 3.0:
            reward -= (distance_from_center - 3.0) * 0.5
        
        # Collision penalty
        if len(self.collision_history) > 0:
            reward -= 20.0 * len(self.collision_history)
            self.collision_history.clear()
        
        # Lane invasion penalty
        if len(self.lane_invasion_history) > 0:
            reward -= 5.0 * len(self.lane_invasion_history)
            self.lane_invasion_history.clear()
        
        # Small time penalty to encourage speed
        reward -= 0.05
        
        return reward
    
    def _check_done(self):
        """
        Check if episode should end
        
        Returns:
            done: Boolean
            info: Dictionary with episode info
        """
        done = False
        info = {
            'episode_step': self.episode_step,
            'total_reward': self.total_reward,
            'lap_complete': self.lap_complete,
            'checkpoints_reached': self.current_checkpoint,
            'collisions': len(self.collision_history),
        }
        
        # Check if max steps reached
        if self.episode_step >= self.max_steps:
            done = True
            info['termination_reason'] = 'max_steps'
        
        # Check if lap completed
        if self.lap_complete:
            done = True
            info['termination_reason'] = 'lap_complete'
            info['lap_time'] = self.lap_times[-1] if self.lap_times else None
        
        # Check if vehicle is stuck or crashed
        if len(self.speeds) > 10 and np.mean(list(self.speeds)[-10:]) < 1.0:
            done = True
            info['termination_reason'] = 'stuck'
        
        return done, info
    
    def _get_distance_to_checkpoint(self):
        """Calculate distance to next checkpoint"""
        if self.vehicle is None:
            return 0.0
        
        location = self.vehicle.get_location()
        checkpoint = self.checkpoints[self.current_checkpoint]
        
        distance = math.sqrt(
            (location.x - checkpoint.x)**2 +
            (location.y - checkpoint.y)**2
        )
        
        return distance
    
    def _get_angle_to_checkpoint(self, location, rotation):
        """Calculate angle to next checkpoint"""
        checkpoint = self.checkpoints[self.current_checkpoint]
        
        # Vector to checkpoint
        dx = checkpoint.x - location.x
        dy = checkpoint.y - location.y
        
        # Target angle
        target_angle = math.degrees(math.atan2(dy, dx))
        
        # Current heading
        current_angle = rotation.yaw
        
        # Angle difference
        angle_diff = target_angle - current_angle
        
        # Normalize to [-180, 180]
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        
        return angle_diff
    
    def _get_distance_from_center(self, location):
        """Calculate approximate distance from track center"""
        waypoint = self.map.get_waypoint(location)
        if waypoint is None:
            return 10.0  # Large penalty if off road
        
        center = waypoint.transform.location
        distance = math.sqrt(
            (location.x - center.x)**2 +
            (location.y - center.y)**2
        )
        
        return distance
    
    def _spawn_vehicle(self):
        """Spawn vehicle at starting position"""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]

        # Try multiple spawn locations
        spawn_attempts = [
            # Original position at first checkpoint
            carla.Transform(
                location=self.checkpoints[0] + carla.Location(z=2.0),  # 2m higher
                rotation=carla.Rotation(pitch=0, yaw=0, roll=0)
            ),
            # Backup: slightly offset
            carla.Transform(
                location=self.checkpoints[0] + carla.Location(x=5.0, z=2.0),
                rotation=carla.Rotation(pitch=0, yaw=0, roll=0)
            ),
            # Backup: use CARLA's recommended spawn points
            None  # Will use get_spawn_points()
        ]

        for i, spawn_point in enumerate(spawn_attempts):
            try:
                if spawn_point is None:
                    # Use CARLA's built-in spawn points
                    spawn_points = self.map.get_spawn_points()
                    if spawn_points:
                        spawn_point = spawn_points[0]
                    else:
                        continue
                    
                self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                print(f"✓ Vehicle spawned successfully (attempt {i+1})")
                break

            except RuntimeError as e:
                if i < len(spawn_attempts) - 1:
                    print(f"  Spawn attempt {i+1} failed, trying next location...")
                    continue
                else:
                    raise RuntimeError(f"Failed to spawn vehicle after {len(spawn_attempts)} attempts: {e}")
    
        # Let vehicle settle
        for _ in range(10):
            self.world.tick()
        
    def _attach_sensors(self):
        """Attach collision and lane invasion sensors"""
        blueprint_library = self.world.get_blueprint_library()
        
        # Collision sensor
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_transform = carla.Transform(carla.Location(x=0, y=0, z=0))
        self.collision_sensor = self.world.spawn_actor(
            collision_bp, collision_transform, attach_to=self.vehicle
        )
        self.collision_sensor.listen(
            lambda event: self.collision_history.append(event)
        )
        
        # Lane invasion sensor
        lane_invasion_bp = blueprint_library.find('sensor.other.lane_invasion')
        lane_invasion_transform = carla.Transform(carla.Location(x=0, y=0, z=0))
        self.lane_invasion_sensor = self.world.spawn_actor(
            lane_invasion_bp, lane_invasion_transform, attach_to=self.vehicle
        )
        self.lane_invasion_sensor.listen(
            lambda event: self.lane_invasion_history.append(event)
        )
    
    def _cleanup(self):
        """Clean up actors"""
        actors_to_destroy = []
        
        if self.collision_sensor is not None:
            actors_to_destroy.append(self.collision_sensor)
            self.collision_sensor = None
        
        if self.lane_invasion_sensor is not None:
            actors_to_destroy.append(self.lane_invasion_sensor)
            self.lane_invasion_sensor = None
        
        if self.vehicle is not None:
            actors_to_destroy.append(self.vehicle)
            self.vehicle = None
        
        if actors_to_destroy:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in actors_to_destroy])
    
    def get_feedback_data(self):
        """
        Get current feedback data for visualization
        
        Returns:
            Dictionary with all feedback metrics
        """
        feedback = {
            'episode_step': self.episode_step,
            'total_reward': self.total_reward,
            'recent_avg_reward': np.mean(list(self.reward_history)) if self.reward_history else 0,
            'current_speed': self.speeds[-1] if self.speeds else 0,
            'avg_speed': np.mean(list(self.speeds)) if self.speeds else 0,
            'distance_to_checkpoint': self._get_distance_to_checkpoint(),
            'current_checkpoint': self.current_checkpoint,
            'total_checkpoints': len(self.checkpoints),
            'progress': self.current_checkpoint / len(self.checkpoints) * 100,
            'lap_complete': self.lap_complete,
            'best_lap_time': self.best_lap_time if self.best_lap_time != float('inf') else None,
            'current_lap_time': time.time() - self.episode_start_time,
            'collisions': len(self.collision_history),
            'lane_invasions': len(self.lane_invasion_history),
        }
        
        if self.vehicle:
            location = self.vehicle.get_location()
            feedback['distance_from_center'] = self._get_distance_from_center(location)
            
            transform = self.vehicle.get_transform()
            feedback['angle_to_checkpoint'] = self._get_angle_to_checkpoint(
                location, transform.rotation
            )
        
        return feedback
    
    def close(self):
        """Clean up and close environment"""
        self._cleanup()
        print("Environment closed.")


if __name__ == "__main__":
    # Test the environment
    print("Testing CARLA RL Environment...")
    
    env = CarlaRLEnvironment()
    state = env.reset()
    
    print(f"Initial state shape: {state.shape}")
    print(f"Initial state: {state}")
    
    # Run a few random steps
    for i in range(10):
        action = np.random.uniform(-1, 1, size=3)
        state, reward, done, info = env.step(action)
        
        feedback = env.get_feedback_data()
        print(f"\nStep {i+1}:")
        print(f"  Reward: {reward:.2f}")
        print(f"  Speed: {feedback['current_speed']:.1f} km/h")
        print(f"  Distance to checkpoint: {feedback['distance_to_checkpoint']:.1f}m")
        
        if done:
            print(f"  Episode ended: {info['termination_reason']}")
            break
    
    env.close()
