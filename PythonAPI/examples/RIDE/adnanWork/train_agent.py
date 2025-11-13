import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import time
import os
from collections import deque
import json

from carla_rl_environment import CarlaRLEnvironment
from feedback_overlay import FeedbackOverlay


class ActorCritic(nn.Module):
    """
    Actor-Critic neural network for PPO
    Actor: Policy network (chooses actions)
    Critic: Value network (estimates state values)
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        
        # Shared feature layers
        self.features = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )
        
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, state):
        features = self.features(state)
        return features
    
    def act(self, state):
        """Sample action from policy"""
        features = self.forward(state)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std)
        
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, action_log_prob
    
    def evaluate(self, state, action):
        """Evaluate action and state"""
        features = self.forward(state)
        
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std)
        
        dist = Normal(action_mean, action_std)
        action_log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        state_value = self.critic(features).squeeze()
        
        return action_log_prob, state_value, entropy


class PPOTrainer:
    """
    Proximal Policy Optimization (PPO) Trainer
    """
    
    def __init__(self, 
                 state_dim, 
                 action_dim,
                 lr=3e-4,
                 gamma=0.99,
                 clip_epsilon=0.2,
                 epochs=10,
                 batch_size=64):
        """
        Initialize PPO trainer
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            lr: Learning rate
            gamma: Discount factor
            clip_epsilon: PPO clipping parameter
            epochs: Number of optimization epochs per update
            batch_size: Mini-batch size
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.batch_size = batch_size
        
        # Replay buffer
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def select_action(self, state):
        """Select action using current policy"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob = self.policy.act(state_tensor)
            
            # Get value estimate
            features = self.policy.forward(state_tensor)
            value = self.policy.critic(features)
        
        return action.cpu().numpy()[0], log_prob.cpu().item(), value.cpu().item()
    
    def store_transition(self, state, action, reward, log_prob, value, done):
        """Store transition in replay buffer"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def update(self):
        """Update policy using PPO"""
        if len(self.states) == 0:
            return {}
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        
        # Calculate returns and advantages
        returns = self._compute_returns()
        returns = torch.FloatTensor(returns).to(self.device)
        
        values = torch.FloatTensor(np.array(self.values)).to(self.device)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(self.epochs):
            # Random mini-batches
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Evaluate actions
                log_probs, state_values, entropy = self.policy.evaluate(
                    batch_states, batch_actions
                )
                
                # Ratio for PPO
                ratios = torch.exp(log_probs - batch_old_log_probs)
                
                # Surrogate loss
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(state_values, batch_returns)
                
                # Entropy bonus (encourages exploration)
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
        
        # Clear buffer
        self.clear_buffer()
        
        # Return training stats
        num_updates = self.epochs * (len(states) // self.batch_size + 1)
        stats = {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates
        }
        
        return stats
    
    def _compute_returns(self):
        """Compute discounted returns"""
        returns = []
        R = 0
        
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                R = 0
            R = reward + self.gamma * R
            returns.insert(0, R)
        
        return returns
    
    def clear_buffer(self):
        """Clear replay buffer"""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def save(self, filepath):
        """Save model"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load model"""
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filepath}")


def train_agent(num_episodes=1000,
                max_steps_per_episode=2000,
                update_frequency=2048,
                checkpoint_dir='checkpoints',
                checkpoint_file='oval_track_checkpoints.txt',
                save_frequency=50,
                render=True):
    """
    Main training loop
    
    Args:
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        update_frequency: Update policy after this many steps
        checkpoint_dir: Directory to save model checkpoints
        checkpoint_file: Track checkpoint file
        save_frequency: Save model every N episodes
        render: Whether to show feedback overlay
    """
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize environment
    print("Initializing CARLA environment...")
    env = CarlaRLEnvironment(checkpoint_file=checkpoint_file)
    
    # Get state and action dimensions
    state = env.reset()
    state_dim = len(state)
    action_dim = 2  # [throttle, steer]
    
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")
    
    # Initialize trainer
    trainer = PPOTrainer(state_dim, action_dim)
    
    # Initialize feedback overlay
    overlay = None
    if render:
        overlay = FeedbackOverlay(width=900, height=700)
    
    # Training statistics
    episode_rewards = []
    episode_lengths = []
    best_reward = -float('inf')
    lap_times = []
    
    # Training metrics
    training_log = {
        'episodes': [],
        'rewards': [],
        'lap_times': [],
        'avg_speeds': [],
        'collisions': []
    }
    
    total_steps = 0
    
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    try:
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_length = 0
            episode_start = time.time()
            
            done = False
            
            while not done and episode_length < max_steps_per_episode:
                # Select action
                action, log_prob, value = trainer.select_action(state)
                
                # Convert action to [throttle, steer] format
                # Action[0]: throttle (0 to 1)
                # Action[1]: steer (-1 to 1)
                throttle = (action[0] + 1) / 2  # Convert from [-1,1] to [0,1]
                steer = action[1]
                formatted_action = [throttle, steer, 0]  # No brake
                
                # Take step
                next_state, reward, done, info = env.step(formatted_action)
                
                # Store transition
                trainer.store_transition(state, action, reward, log_prob, value, done)
                
                state = next_state
                episode_reward += reward
                episode_length += 1
                total_steps += 1
                
                # Update policy
                if total_steps % update_frequency == 0:
                    update_stats = trainer.update()
                    if update_stats:
                        print(f"\nPolicy Update at step {total_steps}:")
                        print(f"  Policy Loss: {update_stats['policy_loss']:.4f}")
                        print(f"  Value Loss: {update_stats['value_loss']:.4f}")
                        print(f"  Entropy: {update_stats['entropy']:.4f}")
                
                # Update visualization
                if overlay:
                    feedback_data = env.get_feedback_data()
                    if not overlay.update(feedback_data, episode_num=episode + 1):
                        raise KeyboardInterrupt
            
            episode_time = time.time() - episode_start
            
            # Episode statistics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            if info.get('lap_complete'):
                lap_time = info.get('lap_time', episode_time)
                lap_times.append(lap_time)
                if overlay:
                    overlay.add_lap_time(lap_time)
            
            # Log episode
            avg_reward_recent = np.mean(episode_rewards[-100:])
            
            print(f"\n{'=' * 60}")
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"{'=' * 60}")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Avg Reward (last 100): {avg_reward_recent:.2f}")
            print(f"  Episode Length: {episode_length} steps")
            print(f"  Episode Time: {episode_time:.2f}s")
            print(f"  Checkpoints: {info.get('checkpoints_reached', 0)}/{env.checkpoints.__len__()}")
            print(f"  Collisions: {info.get('collisions', 0)}")
            
            if info.get('lap_complete'):
                print(f"  LAP COMPLETE! Time: {info.get('lap_time', 0):.2f}s")
                if lap_times:
                    print(f"  Best Lap: {min(lap_times):.2f}s")
            
            print(f"  Termination: {info.get('termination_reason', 'unknown')}")
            
            # Save training log
            training_log['episodes'].append(episode + 1)
            training_log['rewards'].append(episode_reward)
            training_log['lap_times'].append(info.get('lap_time') if info.get('lap_complete') else None)
            training_log['collisions'].append(info.get('collisions', 0))
            
            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                trainer.save(os.path.join(checkpoint_dir, 'best_model.pth'))
                print(f"  *** New best reward! Saved model ***")
            
            # Periodic save
            if (episode + 1) % save_frequency == 0:
                trainer.save(os.path.join(checkpoint_dir, f'model_episode_{episode + 1}.pth'))
                
                # Save training log
                with open(os.path.join(checkpoint_dir, 'training_log.json'), 'w') as f:
                    json.dump(training_log, f, indent=2)
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    
    finally:
        # Final save
        trainer.save(os.path.join(checkpoint_dir, 'final_model.pth'))
        
        # Save final training log
        with open(os.path.join(checkpoint_dir, 'training_log.json'), 'w') as f:
            json.dump(training_log, f, indent=2)
        
        # Cleanup
        env.close()
        if overlay:
            overlay.close()
        
        # Print final statistics
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Total Episodes: {len(episode_rewards)}")
        print(f"Average Reward: {np.mean(episode_rewards):.2f}")
        print(f"Best Reward: {best_reward:.2f}")
        if lap_times:
            print(f"Best Lap Time: {min(lap_times):.2f}s")
            print(f"Total Laps Completed: {len(lap_times)}")
        print(f"Models saved to: {checkpoint_dir}/")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train CARLA RL Racing Agent')
    parser.add_argument('--episodes', type=int, default=500, help='Number of training episodes')
    parser.add_argument('--checkpoint-file', type=str, default='oval_track_checkpoints.txt',
                       help='Track checkpoint file')
    parser.add_argument('--no-render', action='store_true', help='Disable visualization')
    parser.add_argument('--save-freq', type=int, default=50, help='Save model every N episodes')
    
    args = parser.parse_args()
    
    train_agent(
        num_episodes=args.episodes,
        checkpoint_file=args.checkpoint_file,
        save_frequency=args.save_freq,
        render=not args.no_render
    )
