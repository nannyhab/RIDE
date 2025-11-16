"""
Car Configuration Optimizer using Naive Bayes and Deep Learning
Finds optimal suspension parameters (k_s, c_s) for different tracks
"""

import carla
import numpy as np
import math
import time
import csv
import os
import pickle
from collections import deque, defaultdict
import json

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ============================================================================
# CONFIGURATION PARAMETERS
# ============================================================================

# Suspension configurations to test
CONFIGURATIONS = [
    {'k_s': 3000, 'c_s': 600},
    {'k_s': 3000, 'c_s': 1000},
    {'k_s': 3000, 'c_s': 1400},
    {'k_s': 5000, 'c_s': 600},
    {'k_s': 5000, 'c_s': 1000},
    {'k_s': 5000, 'c_s': 1400},
    {'k_s': 7000, 'c_s': 600},
    {'k_s': 7000, 'c_s': 1000},
    {'k_s': 7000, 'c_s': 1400},
    {'k_s': 9000, 'c_s': 600},
    {'k_s': 9000, 'c_s': 1000},
    {'k_s': 9000, 'c_s': 1400},
]

# Track difficulties (0=straight, 1=circle, 2=S, 3=tight S, 4=double S)
DIFFICULTIES = [0, 1, 2, 3, 4]

# Controller parameters
LOOKAHEAD = 10.0
TARGET_SPEED = 30.0  # km/h
MAX_LAP_TIME = 120.0  # seconds

# Data collection
DATA_DIR = "config_data"
os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================================
# PURE PURSUIT CONTROLLER (Fixed - not learning)
# ============================================================================

class PurePursuitController:
    """
    Simple pure pursuit controller
    This does NOT learn - it's a fixed algorithm
    """
    
    def __init__(self, lookahead=LOOKAHEAD, target_speed=TARGET_SPEED):
        self.lookahead = lookahead
        self.target_speed = target_speed / 3.6  # Convert to m/s
        self.prev_error = 0.0
        
    def get_control(self, vehicle, track_x, track_y, nearest_idx):
        """
        Pure pursuit steering + simple speed control
        Returns: (throttle, steer)
        """
        tf = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        current_speed = math.sqrt(velocity.x**2 + velocity.y**2)
        
        # Find lookahead point
        lookahead_idx = min(nearest_idx + int(self.lookahead), len(track_x) - 1)
        target_x = track_x[lookahead_idx]
        target_y = track_y[lookahead_idx]
        
        # Calculate steering (pure pursuit)
        dx = target_x - tf.location.x
        dy = target_y - tf.location.y
        
        # Transform to vehicle frame
        yaw = math.radians(tf.rotation.yaw)
        local_x = math.cos(yaw) * dx + math.sin(yaw) * dy
        local_y = -math.sin(yaw) * dx + math.cos(yaw) * dy
        
        # Pure pursuit formula
        L = math.sqrt(local_x**2 + local_y**2) or 1e-6
        steer = math.atan2(2.0 * 2.88 * local_y, L * L)  # 2.88 = wheelbase
        steer = np.clip(steer / 0.6, -1.0, 1.0)
        
        # Simple speed control (PID)
        speed_error = self.target_speed - current_speed
        throttle = 0.5 + 0.1 * speed_error
        throttle = np.clip(throttle, 0.0, 1.0)
        
        return throttle, steer


# ============================================================================
# DATA COLLECTION
# ============================================================================

def collect_data_for_config(env, config, difficulty, num_laps=3):
    """
    Run laps with a fixed configuration and collect data
    
    Returns:
        data: list of state dictionaries
        metrics: performance metrics (lap_time, comfort, stability)
    """
    print(f"  Testing config k_s={config['k_s']}, c_s={config['c_s']} on difficulty {difficulty}")
    
    controller = PurePursuitController()
    
    # Reset environment with this configuration
    env.set_physics_config(config['k_s'], config['c_s'])
    state = env.reset()
    
    # Data storage
    data_points = []
    lap_start_time = time.time()
    
    # Metrics
    comfort_scores = []
    stability_scores = []
    speeds = []
    lateral_offsets = []
    
    done = False
    step = 0
    max_steps = 5000
    
    while not done and step < max_steps:
        # Get vehicle state
        vehicle = env.vehicle
        tf = vehicle.get_transform()
        velocity = vehicle.get_velocity()
        accel = vehicle.get_acceleration()
        angular_vel = vehicle.get_angular_velocity()
        
        # Get nearest point on track
        loc = tf.location
        dx = env.track_x - loc.x
        dy = env.track_y - loc.y
        nearest_idx = int(np.argmin(dx**2 + dy**2))
        
        # Get control action from FIXED controller
        throttle, steer = controller.get_control(
            vehicle, env.track_x, env.track_y, nearest_idx
        )
        
        # Execute action
        action = np.array([throttle, steer])
        next_state, reward, done, info = env.step(action)
        
        # Calculate metrics
        speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2)
        speeds.append(speed)
        
        # Comfort (lower acceleration = more comfortable)
        comfort = 1.0 / (1.0 + abs(accel.x) + abs(accel.y))
        comfort_scores.append(comfort)
        
        # Stability (lower angular velocity = more stable)
        stability = 1.0 / (1.0 + abs(angular_vel.z))
        stability_scores.append(stability)
        
        # Lateral offset
        lateral_offset = env._lateral_offset(loc, nearest_idx)
        lateral_offsets.append(abs(lateral_offset))
        
        # Record data point
        data_point = {
            # Configuration
            'k_s': config['k_s'],
            'c_s': config['c_s'],
            'difficulty': difficulty,
            
            # State (same as RL state)
            'speed': state[0],
            'distance_to_next': state[1],
            'angle_to_next': state[2],
            'lateral_offset': state[3],
            'heading': state[4],
            'velocity_x': state[5],
            'velocity_y': state[6],
            'progress': state[7],
            
            # Additional metrics
            'comfort': comfort,
            'stability': stability,
        }
        data_points.append(data_point)
        
        state = next_state
        step += 1
    
    # Calculate final metrics
    lap_time = time.time() - lap_start_time
    
    metrics = {
        'lap_time': lap_time,
        'completed': info.get('lap_complete', False),
        'avg_speed': np.mean(speeds) if speeds else 0.0,
        'avg_comfort': np.mean(comfort_scores) if comfort_scores else 0.0,
        'avg_stability': np.mean(stability_scores) if stability_scores else 0.0,
        'avg_lateral_offset': np.mean(lateral_offsets) if lateral_offsets else 0.0,
        'termination_reason': info.get('termination_reason', 'unknown')
    }
    
    return data_points, metrics


def collect_all_data():
    """
    Collect data for all configurations on all tracks
    """
    from carla_rl_environment_improved import CarlaRLEnvironment
    
    all_data = []
    all_metrics = []
    
    for difficulty in DIFFICULTIES:
        print(f"\n{'='*60}")
        print(f"Testing on Track Difficulty {difficulty}")
        print(f"{'='*60}")
        
        # Create environment for this difficulty
        env = CarlaRLEnvironment(difficulty=difficulty)
        
        for config in CONFIGURATIONS:
            try:
                data_points, metrics = collect_data_for_config(env, config, difficulty)
                
                # Store
                all_data.extend(data_points)
                all_metrics.append({
                    **config,
                    'difficulty': difficulty,
                    **metrics
                })
                
                print(f"    Lap time: {metrics['lap_time']:.2f}s, "
                      f"Comfort: {metrics['avg_comfort']:.3f}, "
                      f"Stability: {metrics['avg_stability']:.3f}")
                
            except Exception as e:
                print(f"    ERROR: {e}")
                continue
        
        env.close()
    
    # Save data
    with open(f'{DATA_DIR}/training_data.pkl', 'wb') as f:
        pickle.dump(all_data, f)
    
    with open(f'{DATA_DIR}/metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    print(f"\n✓ Collected {len(all_data)} data points")
    print(f"✓ Saved to {DATA_DIR}/")
    
    return all_data, all_metrics


# ============================================================================
# NAIVE BAYES CLASSIFIER
# ============================================================================

class ConfigurationClassifier:
    """
    Naive Bayes classifier to predict if a configuration is good/bad
    """
    
    def __init__(self):
        self.model = GaussianNB()
        self.scaler = StandardScaler()
        
    def train(self, data, metrics):
        """
        Train classifier
        
        data: list of data points with states
        metrics: list of performance metrics
        """
        # Prepare features
        features = []
        labels = []
        
        # Group by config + difficulty
        config_performance = defaultdict(list)
        for metric in metrics:
            key = (metric['k_s'], metric['c_s'], metric['difficulty'])
            config_performance[key].append(metric)
        
        # Calculate average performance for each config
        config_scores = {}
        for key, perfs in config_performance.items():
            avg_lap_time = np.mean([p['lap_time'] for p in perfs])
            avg_comfort = np.mean([p['avg_comfort'] for p in perfs])
            avg_stability = np.mean([p['avg_stability'] for p in perfs])
            
            # Combined score (lower is better for lap time)
            score = -avg_lap_time + 50 * avg_comfort + 50 * avg_stability
            config_scores[key] = score
        
        # Classify as good (1) or bad (0) based on median
        median_score = np.median(list(config_scores.values()))
        
        for point in data:
            key = (point['k_s'], point['c_s'], point['difficulty'])
            
            # Features: track difficulty + current states
            feature = [
                point['difficulty'],
                point['speed'],
                point['distance_to_next'],
                point['angle_to_next'],
                point['lateral_offset'],
                point['heading'],
                point['velocity_x'],
                point['velocity_y'],
                point['progress'],
            ]
            features.append(feature)
            
            # Label: good (1) or bad (0) config
            label = 1 if config_scores.get(key, 0) >= median_score else 0
            labels.append(label)
        
        # Train
        X = np.array(features)
        y = np.array(labels)
        
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        # Evaluate
        accuracy = self.model.score(X_scaled, y)
        print(f"✓ Naive Bayes accuracy: {accuracy:.3f}")
        
    def predict(self, difficulty, state):
        """
        Predict if current state with config is good
        """
        feature = np.array([[
            difficulty,
            state[0], state[1], state[2], state[3],
            state[4], state[5], state[6], state[7]
        ]])
        
        feature_scaled = self.scaler.transform(feature)
        proba = self.model.predict_proba(feature_scaled)[0]
        
        return proba[1]  # Probability of good config
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
    
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.scaler = data['scaler']


# ============================================================================
# DEEP LEARNING PREDICTOR
# ============================================================================

class ConfigurationPredictor(nn.Module):
    """
    Deep neural network to predict optimal k_s and c_s
    """
    
    def __init__(self, input_dim=9):
        super(ConfigurationPredictor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            
            nn.Linear(64, 2)  # Output: [k_s, c_s]
        )
        
    def forward(self, x):
        return self.network(x)


class DeepConfigOptimizer:
    """
    Deep learning model to predict optimal configuration
    """
    
    def __init__(self):
        self.model = ConfigurationPredictor()
        self.scaler = StandardScaler()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        
    def train(self, data, metrics, epochs=100):
        """
        Train deep learning model
        """
        # Prepare dataset
        features = []
        targets = []
        
        # Map configs to performance scores
        config_scores = defaultdict(list)
        for metric in metrics:
            key = (metric['k_s'], metric['c_s'], metric['difficulty'])
            score = -metric['lap_time'] + 50 * metric['avg_comfort'] + 50 * metric['avg_stability']
            config_scores[key].append(score)
        
        # Average scores per config
        best_configs = {}
        for difficulty in DIFFICULTIES:
            diff_scores = {
                (k, c): np.mean(config_scores[(k, c, difficulty)])
                for k, c, d in config_scores.keys() if d == difficulty
            }
            if diff_scores:
                best_configs[difficulty] = max(diff_scores, key=diff_scores.get)
        
        # Create training data
        for point in data:
            difficulty = point['difficulty']
            
            # Feature: difficulty + states
            feature = [
                difficulty,
                point['speed'],
                point['distance_to_next'],
                point['angle_to_next'],
                point['lateral_offset'],
                point['heading'],
                point['velocity_x'],
                point['velocity_y'],
                point['progress'],
            ]
            features.append(feature)
            
            # Target: best k_s and c_s for this difficulty
            if difficulty in best_configs:
                best_k, best_c = best_configs[difficulty]
                target = [best_k / 10000.0, best_c / 2000.0]  # Normalize
            else:
                target = [0.5, 0.5]  # Default
            
            targets.append(target)
        
        # Convert to tensors
        X = torch.FloatTensor(features)
        y = torch.FloatTensor(targets)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X.numpy())
        X = torch.FloatTensor(X_scaled)
        
        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Train
            self.model.train()
            self.optimizer.zero_grad()
            
            outputs = self.model(X_train)
            loss = self.criterion(outputs, y_train)
            
            loss.backward()
            self.optimizer.step()
            
            # Validate
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val)
                val_loss = self.criterion(val_outputs, y_val)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, "
                      f"Train Loss: {loss.item():.4f}, "
                      f"Val Loss: {val_loss.item():.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
        
        print(f"✓ Deep learning training complete. Best val loss: {best_val_loss:.4f}")
    
    def predict(self, difficulty, state):
        """
        Predict optimal k_s and c_s
        """
        self.model.eval()
        
        feature = np.array([[
            difficulty,
            state[0], state[1], state[2], state[3],
            state[4], state[5], state[6], state[7]
        ]])
        
        feature_scaled = self.scaler.transform(feature)
        feature_tensor = torch.FloatTensor(feature_scaled)
        
        with torch.no_grad():
            prediction = self.model(feature_tensor).numpy()[0]
        
        # Denormalize
        k_s = prediction[0] * 10000.0
        c_s = prediction[1] * 2000.0
        
        # Clip to valid ranges
        k_s = np.clip(k_s, 1000, 10000)
        c_s = np.clip(c_s, 100, 3000)
        
        return k_s, c_s
    
    def save(self, path):
        torch.save({
            'model': self.model.state_dict(),
            'scaler': self.scaler
        }, path)
    
    def load(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.scaler = checkpoint['scaler']


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    print("="*60)
    print("CAR CONFIGURATION OPTIMIZER")
    print("Using Naive Bayes + Deep Learning")
    print("="*60)
    
    # Step 1: Collect data
    print("\n[1/3] Collecting data across all configurations and tracks...")
    data, metrics = collect_all_data()
    
    # Step 2: Train Naive Bayes
    print("\n[2/3] Training Naive Bayes classifier...")
    nb_classifier = ConfigurationClassifier()
    nb_classifier.train(data, metrics)
    nb_classifier.save(f'{DATA_DIR}/naive_bayes.pkl')
    
    # Step 3: Train Deep Learning
    print("\n[3/3] Training Deep Learning predictor...")
    dl_predictor = DeepConfigOptimizer()
    dl_predictor.train(data, metrics, epochs=100)
    dl_predictor.save(f'{DATA_DIR}/deep_learning.pth')
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    # Test predictions
    print("\nTesting predictions on each track:")
    test_state = np.array([0.3, 0.5, 0.0, 0.1, 0.0, 0.2, 0.1, 0.5])
    
    for diff in DIFFICULTIES:
        nb_prob = nb_classifier.predict(diff, test_state)
        k_s, c_s = dl_predictor.predict(diff, test_state)
        
        print(f"\nDifficulty {diff}:")
        print(f"  Naive Bayes: {nb_prob:.3f} probability of good config")
        print(f"  Deep Learning: k_s={k_s:.0f}, c_s={c_s:.0f}")
    
    # Analyze best configurations
    print("\n" + "="*60)
    print("BEST CONFIGURATIONS PER TRACK")
    print("="*60)
    
    for diff in DIFFICULTIES:
        diff_metrics = [m for m in metrics if m['difficulty'] == diff and m['completed']]
        
        if diff_metrics:
            best = min(diff_metrics, key=lambda x: x['lap_time'])
            print(f"\nDifficulty {diff}:")
            print(f"  Best: k_s={best['k_s']}, c_s={best['c_s']}")
            print(f"  Lap time: {best['lap_time']:.2f}s")
            print(f"  Comfort: {best['avg_comfort']:.3f}")
            print(f"  Stability: {best['avg_stability']:.3f}")


if __name__ == "__main__":
    main()
