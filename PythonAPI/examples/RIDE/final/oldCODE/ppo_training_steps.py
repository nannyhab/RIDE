"""
Complete Guide: RL-PPO Training Steps for Autonomous Racing
Covers all theoretical and practical steps from dynamics to deployment
"""

# ============================================================================
# STEP 1: DYNAMIC EQUATION (System Dynamics)
# ============================================================================

"""
The dynamic equation models how the vehicle responds to control inputs.

For autonomous racing, we need to model:
1. Vehicle kinematics (position, velocity, orientation)
2. Vehicle dynamics (acceleration, steering response)
3. Environmental interactions (friction, collisions)

VEHICLE DYNAMICS MODEL:
"""

import numpy as np

class VehicleDynamics:
    """
    Simplified vehicle dynamics model for racing
    Models the physical behavior of the car
    """
    
    def __init__(self, 
                 mass=1500.0,          # kg
                 length=4.5,           # m (wheelbase)
                 max_steer=0.7,        # radians
                 max_throttle=1.0,
                 k_s=5000.0,           # Spring constant (N/m)
                 c_s=1000.0):          # Damping coefficient (N·s/m)
        """
        Initialize physical parameters
        
        Args:
            mass: Vehicle mass
            length: Wheelbase length
            max_steer: Maximum steering angle
            max_throttle: Maximum throttle
            k_s: Spring constant (suspension stiffness)
            c_s: Damping ratio (suspension damping)
        """
        self.m = mass
        self.L = length
        self.max_steer = max_steer
        self.max_throttle = max_throttle
        
        # Suspension parameters
        self.k_s = k_s  # Spring stiffness
        self.c_s = c_s  # Damping coefficient
        
        # State variables
        self.x = 0.0      # x position
        self.y = 0.0      # y position
        self.theta = 0.0  # heading angle
        self.v = 0.0      # velocity
        self.omega = 0.0  # angular velocity
        
    def kinematic_bicycle_model(self, throttle, steer, dt):
        """
        Kinematic bicycle model - simplified vehicle dynamics
        
        State equations:
        x_dot = v * cos(theta)
        y_dot = v * sin(theta)
        theta_dot = (v / L) * tan(steer)
        v_dot = throttle * a_max - friction
        
        Args:
            throttle: Throttle input [0, 1]
            steer: Steering angle [-max_steer, max_steer]
            dt: Time step
        
        Returns:
            Updated state
        """
        # Maximum acceleration
        a_max = 5.0  # m/s^2
        
        # Friction/drag (simplified)
        friction = 0.1 * self.v
        
        # State derivatives
        x_dot = self.v * np.cos(self.theta)
        y_dot = self.v * np.sin(self.theta)
        theta_dot = (self.v / self.L) * np.tan(steer)
        v_dot = throttle * a_max - friction
        
        # Update state (Euler integration)
        self.x += x_dot * dt
        self.y += y_dot * dt
        self.theta += theta_dot * dt
        self.v = max(0, self.v + v_dot * dt)  # Velocity can't be negative
        
        return self.get_state()
    
    def dynamic_model_with_suspension(self, throttle, steer, dt):
        """
        More detailed dynamic model including suspension
        
        Suspension force: F_s = -k_s * x - c_s * x_dot
        
        This affects:
        - Vertical dynamics (comfort)
        - Tire contact (traction)
        - Vehicle stability
        """
        # Basic kinematic update
        state = self.kinematic_bicycle_model(throttle, steer, dt)
        
        # Suspension dynamics (simplified vertical motion)
        # z_dot_dot = -(k_s/m) * z - (c_s/m) * z_dot + road_profile
        
        # This affects comfort metric in reward function
        suspension_quality = 1.0 / (1.0 + self.k_s/10000.0)  # Softer = more comfortable
        
        return state, suspension_quality
    
    def get_state(self):
        """Return current state vector"""
        return np.array([self.x, self.y, self.theta, self.v, self.omega])


# ============================================================================
# STEP 2: DEFINE SYSTEM ENVIRONMENT
# ============================================================================

"""
The environment encapsulates:
1. System dynamics
2. State space
3. Action space
4. Reward function
5. Episode termination conditions
"""

class RacingEnvironment:
    """
    Complete racing environment with vehicle dynamics
    """
    
    def __init__(self, use_carla=True):
        """
        Initialize environment
        
        Args:
            use_carla: Use CARLA simulator (True) or simple dynamics (False)
        """
        self.use_carla = use_carla
        
        if use_carla:
            from carla_rl_environment import CarlaRLEnvironment
            self.env = CarlaRLEnvironment()
            self.state_dim = 8
            self.action_dim = 2
        else:
            # Use simplified dynamics
            self.dynamics = VehicleDynamics()
            self.state_dim = 5
            self.action_dim = 2
        
        # Physical parameters
        self.k_s = 5000.0  # Spring constant
        self.c_s = 1000.0  # Damping ratio
        
    def reset(self):
        """Reset environment to initial state"""
        if self.use_carla:
            return self.env.reset()
        else:
            self.dynamics = VehicleDynamics(k_s=self.k_s, c_s=self.c_s)
            return self.dynamics.get_state()
    
    def step(self, action):
        """
        Execute action and return next state, reward, done, info
        
        Args:
            action: [throttle, steer]
        
        Returns:
            state, reward, done, info
        """
        if self.use_carla:
            return self.env.step(action)
        else:
            # Use simplified dynamics
            throttle, steer = action
            dt = 0.05  # 50ms time step
            
            state, suspension_quality = self.dynamics.dynamic_model_with_suspension(
                throttle, steer, dt
            )
            
            reward = self._compute_reward(state, suspension_quality)
            done = self._check_done(state)
            info = {}
            
            return state, reward, done, info
    
    def _compute_reward(self, state, suspension_quality):
        """
        Reward function design
        
        Multi-objective optimization:
        - Speed (fast lap times)
        - Comfort (smooth ride via suspension)
        - Safety (avoid collisions)
        """
        x, y, theta, v, omega = state
        
        # Speed reward
        reward_speed = v / 20.0  # Normalize by target speed
        
        # Comfort reward (from suspension)
        reward_comfort = suspension_quality
        
        # Stability reward (penalize excessive angular velocity)
        reward_stability = -abs(omega) * 0.1
        
        # Total reward with weights
        reward = 0.5 * reward_speed + 0.3 * reward_comfort + 0.2 * reward_stability
        
        return reward
    
    def _check_done(self, state):
        """Check if episode should terminate"""
        x, y, theta, v, omega = state
        
        # Terminate if vehicle is too far off track
        if abs(x) > 100 or abs(y) > 100:
            return True
        
        return False


# ============================================================================
# STEP 3: HYPERPARAMETERS FOR RL-PPO
# ============================================================================

"""
Critical hyperparameters that control learning behavior
"""

class PPOHyperparameters:
    """
    PPO Algorithm Hyperparameters
    These control the learning process
    """
    
    def __init__(self):
        # ---- Policy Network Architecture ----
        self.state_dim = 8           # Dimension of state space
        self.action_dim = 2          # Dimension of action space
        self.hidden_dim = 256        # Hidden layer size
        
        # ---- Learning Rates ----
        self.lr_actor = 3e-4         # Learning rate for policy (actor)
        self.lr_critic = 3e-4        # Learning rate for value function (critic)
        
        # ---- PPO Specific ----
        self.gamma = 0.99            # Discount factor (future reward weight)
        self.gae_lambda = 0.95       # GAE parameter (advantage estimation)
        self.clip_epsilon = 0.2      # PPO clipping parameter
        self.c1 = 0.5                # Value loss coefficient
        self.c2 = 0.01               # Entropy coefficient (exploration)
        
        # ---- Training ----
        self.epochs = 10             # PPO update epochs per batch
        self.batch_size = 64         # Mini-batch size
        self.buffer_size = 2048      # Experience buffer size
        self.max_grad_norm = 0.5     # Gradient clipping
        
        # ---- Episode ----
        self.max_episode_steps = 2000    # Max steps per episode
        self.num_episodes = 1000         # Total training episodes
        
        # ---- System Specific (Co-Design Parameters) ----
        self.k_s = 5000.0            # Spring constant (can be optimized)
        self.c_s = 1000.0            # Damping ratio (can be optimized)
        
    def print_hyperparameters(self):
        """Print all hyperparameters"""
        print("\n" + "=" * 60)
        print("PPO HYPERPARAMETERS")
        print("=" * 60)
        
        print("\n📐 Network Architecture:")
        print(f"  State dimension: {self.state_dim}")
        print(f"  Action dimension: {self.action_dim}")
        print(f"  Hidden layer size: {self.hidden_dim}")
        
        print("\n📊 Learning Parameters:")
        print(f"  Actor learning rate: {self.lr_actor}")
        print(f"  Critic learning rate: {self.lr_critic}")
        print(f"  Discount factor (γ): {self.gamma}")
        print(f"  GAE lambda (λ): {self.gae_lambda}")
        
        print("\n🔧 PPO Parameters:")
        print(f"  Clip epsilon (ε): {self.clip_epsilon}")
        print(f"  Value loss coeff (c1): {self.c1}")
        print(f"  Entropy coeff (c2): {self.c2}")
        
        print("\n🎯 Training Parameters:")
        print(f"  Epochs per update: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Buffer size: {self.buffer_size}")
        print(f"  Max episode steps: {self.max_episode_steps}")
        print(f"  Total episodes: {self.num_episodes}")
        
        print("\n⚙️  Physical Parameters:")
        print(f"  Spring constant (k_s): {self.k_s} N/m")
        print(f"  Damping coefficient (c_s): {self.c_s} N·s/m")
        
        print("\n" + "=" * 60)


# ============================================================================
# STEP 4: TRAIN INITIAL POLICY (Actor Network)
# ============================================================================

"""
The policy network (actor) maps states to actions
This is what the agent learns during training
"""

import torch
import torch.nn as nn

class PolicyNetwork(nn.Module):
    """
    Actor network - learns the policy π(a|s)
    Maps states to action probabilities
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        
        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Action mean (μ)
        self.action_mean = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )
        
        # Action std deviation (σ) - learnable
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(self, state):
        """
        Forward pass
        
        Returns:
            action_mean, action_std
        """
        features = self.features(state)
        action_mean = self.action_mean(features)
        action_std = torch.exp(self.action_log_std)
        
        return action_mean, action_std
    
    def get_action(self, state, deterministic=False):
        """
        Sample action from policy
        
        Policy is Gaussian: π(a|s) = N(μ(s), σ(s))
        
        Args:
            state: Current state
            deterministic: If True, return mean action (no exploration)
        
        Returns:
            action, log_prob
        """
        action_mean, action_std = self.forward(state)
        
        if deterministic:
            return action_mean, None
        
        # Sample from Gaussian distribution
        dist = torch.distributions.Normal(action_mean, action_std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob


def initialize_policy():
    """
    Initialize policy network with proper weight initialization
    
    Important for stable training!
    """
    policy = PolicyNetwork(state_dim=8, action_dim=2, hidden_dim=256)
    
    # Xavier/Glorot initialization
    for layer in policy.features:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
    
    for layer in policy.action_mean:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
    
    print("✓ Policy network initialized")
    return policy


# ============================================================================
# STEP 5: TRAIN INITIAL VALUE FUNCTION (Critic Network)
# ============================================================================

"""
The value network (critic) estimates V(s) - expected return from state s
Used for advantage estimation in PPO
"""

class ValueNetwork(nn.Module):
    """
    Critic network - learns the value function V(s)
    Estimates expected cumulative reward from state s
    """
    
    def __init__(self, state_dim, hidden_dim=256):
        super(ValueNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output: V(s)
        )
    
    def forward(self, state):
        """
        Forward pass
        
        Returns:
            State value V(s)
        """
        return self.network(state).squeeze()


def initialize_value_function():
    """
    Initialize value network
    """
    value_net = ValueNetwork(state_dim=8, hidden_dim=256)
    
    # Xavier initialization
    for layer in value_net.network:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
    
    print("✓ Value network initialized")
    return value_net


# ============================================================================
# STEP 6: CO-DESIGN (Optimize Both Policy and Physical Parameters)
# ============================================================================

"""
Co-design optimizes BOTH:
1. Control policy (neural network weights)
2. Physical parameters (k_s, c_s, etc.)

This allows the system to adapt both the controller and the plant
"""

class CoDesignOptimizer:
    """
    Simultaneous optimization of policy and physical parameters
    
    Optimizes:
    - θ: Policy parameters (neural network weights)
    - φ: Physical parameters (k_s, c_s)
    
    Objective: Maximize J(θ, φ) = E[Σ r_t]
    """
    
    def __init__(self, initial_k_s=5000.0, initial_c_s=1000.0):
        """
        Initialize co-design optimizer
        
        Args:
            initial_k_s: Initial spring constant
            initial_c_s: Initial damping coefficient
        """
        # Physical parameters (learnable)
        self.k_s = torch.tensor(initial_k_s, requires_grad=True)
        self.c_s = torch.tensor(initial_c_s, requires_grad=True)
        
        # Policy and value networks
        self.policy = initialize_policy()
        self.value_net = initialize_value_function()
        
        # Optimizers
        # Separate optimizer for physical parameters
        self.physical_optimizer = torch.optim.Adam([self.k_s, self.c_s], lr=1e-3)
        
        # Optimizer for neural network parameters
        self.policy_optimizer = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value_net.parameters()),
            lr=3e-4
        )
        
    def update_physical_parameters(self, reward_gradient):
        """
        Update physical parameters based on reward gradient
        
        ∂J/∂φ = E[∂r/∂φ]
        """
        self.physical_optimizer.zero_grad()
        
        # Compute gradient of reward w.r.t. physical parameters
        loss = -reward_gradient  # Negative because we maximize reward
        loss.backward()
        
        self.physical_optimizer.step()
        
        # Apply constraints (physical parameters must be positive)
        self.k_s.data = torch.clamp(self.k_s.data, min=1000.0, max=10000.0)
        self.c_s.data = torch.clamp(self.c_s.data, min=100.0, max=3000.0)
    
    def get_current_parameters(self):
        """Get current physical parameters"""
        return {
            'k_s': self.k_s.item(),
            'c_s': self.c_s.item()
        }


# ============================================================================
# STEP 7: COMPLETE PPO TRAINING ALGORITHM
# ============================================================================

"""
Complete PPO training loop integrating all components
"""

def train_ppo_with_codesign(num_episodes=1000):
    """
    Complete PPO training with co-design
    
    Algorithm:
    1. Initialize policy π_θ, value function V_φ, physical parameters
    2. For each episode:
        a. Collect trajectories using current policy
        b. Compute advantages using GAE
        c. Update policy using PPO objective
        d. Update value function
        e. Update physical parameters (co-design)
    3. Repeat until convergence
    """
    
    # Initialize
    hyperparams = PPOHyperparameters()
    hyperparams.print_hyperparameters()
    
    env = RacingEnvironment(use_carla=True)
    co_design = CoDesignOptimizer(
        initial_k_s=hyperparams.k_s,
        initial_c_s=hyperparams.c_s
    )
    
    print("\n" + "=" * 60)
    print("STARTING PPO TRAINING WITH CO-DESIGN")
    print("=" * 60)
    
    # Training statistics
    episode_rewards = []
    physical_params_history = []
    
    for episode in range(num_episodes):
        # ---- Step 1: Collect trajectories ----
        state = env.reset()
        trajectory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'log_probs': [],
            'values': [],
            'dones': []
        }
        
        episode_reward = 0
        
        for step in range(hyperparams.max_episode_steps):
            # Get action from policy
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, log_prob = co_design.policy.get_action(state_tensor)
            value = co_design.value_net(state_tensor)
            
            # Execute action
            next_state, reward, done, info = env.step(action.detach().numpy()[0])
            
            # Store transition
            trajectory['states'].append(state)
            trajectory['actions'].append(action.detach().numpy()[0])
            trajectory['rewards'].append(reward)
            trajectory['log_probs'].append(log_prob.item())
            trajectory['values'].append(value.item())
            trajectory['dones'].append(done)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # ---- Step 2: Compute advantages using GAE ----
        advantages = compute_gae(
            trajectory['rewards'],
            trajectory['values'],
            trajectory['dones'],
            gamma=hyperparams.gamma,
            gae_lambda=hyperparams.gae_lambda
        )
        
        # ---- Step 3: PPO Update ----
        ppo_update(
            co_design,
            trajectory,
            advantages,
            hyperparams
        )
        
        # ---- Step 4: Co-design Update (physical parameters) ----
        if episode % 10 == 0:  # Update less frequently
            reward_gradient = torch.tensor(episode_reward)
            co_design.update_physical_parameters(reward_gradient)
            
            current_params = co_design.get_current_parameters()
            physical_params_history.append(current_params)
            
            print(f"\nPhysical Parameters Updated:")
            print(f"  k_s = {current_params['k_s']:.1f} N/m")
            print(f"  c_s = {current_params['c_s']:.1f} N·s/m")
        
        # ---- Logging ----
        episode_rewards.append(episode_reward)
        
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"\nEpisode {episode}/{num_episodes}")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Avg Reward (last 100): {avg_reward:.2f}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    
    return co_design, episode_rewards, physical_params_history


def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """
    Compute Generalized Advantage Estimation (GAE)
    
    GAE formula:
    A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
    
    where δ_t = r_t + γV(s_{t+1}) - V(s_t)
    """
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
        advantages.insert(0, gae)
    
    return advantages


def ppo_update(co_design, trajectory, advantages, hyperparams):
    """
    PPO policy update
    
    PPO objective:
    L(θ) = E[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]
    
    where r_t(θ) = π_θ(a_t|s_t) / π_{θ_old}(a_t|s_t)
    """
    states = torch.FloatTensor(trajectory['states'])
    actions = torch.FloatTensor(trajectory['actions'])
    old_log_probs = torch.FloatTensor(trajectory['log_probs'])
    advantages = torch.FloatTensor(advantages)
    returns = advantages + torch.FloatTensor(trajectory['values'])
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # PPO update for multiple epochs
    for _ in range(hyperparams.epochs):
        # Get current log probs and values
        action_mean, action_std = co_design.policy(states)
        dist = torch.distributions.Normal(action_mean, action_std)
        new_log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()
        
        values = co_design.value_net(states)
        
        # Ratio for PPO
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # PPO clipped objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - hyperparams.clip_epsilon, 1 + hyperparams.clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        critic_loss = nn.MSELoss()(values, returns)
        
        # Total loss
        loss = actor_loss + hyperparams.c1 * critic_loss - hyperparams.c2 * entropy
        
        # Optimize
        co_design.policy_optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(co_design.policy.parameters()) + list(co_design.value_net.parameters()),
            hyperparams.max_grad_norm
        )
        co_design.policy_optimizer.step()


# ============================================================================
# SUMMARY OF ALL STEPS
# ============================================================================

def print_training_pipeline():
    """Print complete training pipeline summary"""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║              PPO TRAINING PIPELINE - COMPLETE STEPS                ║
╚════════════════════════════════════════════════════════════════════╝

STEP 1: DYNAMIC EQUATION
  ✓ Define vehicle dynamics model
  ✓ Kinematic bicycle model: x_dot = f(x, u)
  ✓ Include suspension: F_s = -k_s*x - c_s*x_dot
  ✓ Model parameters: mass, length, k_s, c_s

STEP 2: DEFINE SYSTEM ENVIRONMENT
  ✓ State space: [x, y, θ, v, ω, ...]
  ✓ Action space: [throttle, steer]
  ✓ Reward function: r = w1*speed + w2*comfort + w3*safety
  ✓ Termination conditions

STEP 3: INITIALIZE PHYSICAL PARAMETERS
  ✓ k_s: Spring constant (5000 N/m)
  ✓ c_s: Damping coefficient (1000 N·s/m)
  ✓ Vehicle parameters: mass, wheelbase, etc.

STEP 4: SET HYPERPARAMETERS FOR RL-PPO
  ✓ Learning rates: lr_actor, lr_critic
  ✓ PPO parameters: γ, λ, ε_clip
  ✓ Network architecture: hidden_dim
  ✓ Training parameters: batch_size, epochs

STEP 5: TRAIN INITIAL POLICY (Actor)
  ✓ Initialize policy network π_θ(a|s)
  ✓ Xavier/Glorot initialization
  ✓ Gaussian policy: N(μ(s), σ(s))

STEP 6: TRAIN INITIAL VALUE FUNCTION (Critic)
  ✓ Initialize value network V_φ(s)
  ✓ Estimates expected return from state
  ✓ Used for advantage computation

STEP 7: CO-DESIGN OPTIMIZATION
  ✓ Jointly optimize:
    - Policy parameters θ
    - Physical parameters φ = {k_s, c_s}
  ✓ Maximize J(θ, φ) = E[Σ rewards]

STEP 8: PPO TRAINING LOOP
  For each episode:
    1. Collect trajectories using π_θ
    2. Compute advantages using GAE
    3. Update policy with PPO objective:
       L(θ) = E[min(r_t*A_t, clip(r_t)*A_t)]
    4. Update value function
    5. Update physical parameters (co-design)
    6. Log performance metrics

STEP 9: CONVERGENCE & EVALUATION
  ✓ Monitor: rewards, lap times, collisions
  ✓ Early stopping if converged
  ✓ Save best policy
  ✓ Evaluate on test tracks

═══════════════════════════════════════════════════════════════════

KEY EQUATIONS:

State Dynamics:
  x_{t+1} = f(x_t, u_t, φ)  where φ = {k_s, c_s}

Policy:
  π_θ(a|s) = N(μ_θ(s), σ_θ(s))

Value Function:
  V_φ(s) = E[Σ_{t=0}^∞ γ^t r_t | s_0 = s]

Advantage (GAE):
  A_t = Σ_{l=0}^∞ (γλ)^l δ_{t+l}
  where δ_t = r_t + γV(s_{t+1}) - V(s_t)

PPO Objective:
  L(θ) = E_t[min(r_t(θ)A_t, clip(r_t(θ), 1-ε, 1+ε)A_t)]

Co-Design:
  max_{θ,φ} J(θ, φ) = E[Σ r(s_t, a_t, φ)]

═══════════════════════════════════════════════════════════════════
    """)


if __name__ == "__main__":
    print_training_pipeline()
    
    # Initialize hyperparameters
    hyperparams = PPOHyperparameters()
    hyperparams.print_hyperparameters()
    
    print("\n✓ All components ready for training!")
    print("\nTo start training:")
    print("  python train_agent.py --episodes 500")
