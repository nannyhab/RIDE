"""
Performance Analysis and Visualization
Analyze training results and show learning progression
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os


class PerformanceAnalyzer:
    """
    Analyze and visualize RL training performance
    Shows how agent learns from failures and improves over time
    """
    
    def __init__(self, training_log_path='checkpoints/training_log.json'):
        """
        Initialize analyzer
        
        Args:
            training_log_path: Path to training log JSON file
        """
        self.log_path = training_log_path
        self.data = self._load_data()
    
    def _load_data(self):
        """Load training log data"""
        if not os.path.exists(self.log_path):
            print(f"Warning: Training log not found at {self.log_path}")
            return None
        
        with open(self.log_path, 'r') as f:
            data = json.load(f)
        
        return data
    
    def generate_full_report(self, save_path='training_analysis.png'):
        """
        Generate comprehensive training analysis report
        
        Args:
            save_path: Path to save the figure
        """
        if self.data is None:
            print("No data to analyze")
            return
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Reward progression
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_reward_progression(ax1)
        
        # 2. Lap time improvement
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_lap_time_improvement(ax2)
        
        # 3. Collision rate
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_collision_rate(ax3)
        
        # 4. Learning rate analysis
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_learning_rate(ax4)
        
        # 5. Success rate
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_success_rate(ax5)
        
        plt.suptitle('CARLA RL Training Analysis - Learning from Failures', 
                    fontsize=16, fontweight='bold')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Analysis report saved to {save_path}")
        plt.show()
    
    def _plot_reward_progression(self, ax):
        """Plot reward progression over episodes"""
        episodes = self.data['episodes']
        rewards = self.data['rewards']
        
        # Plot raw rewards
        ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Episode Reward')
        
        # Plot moving average
        window = min(50, len(rewards) // 10)
        if window > 0:
            moving_avg = self._moving_average(rewards, window)
            ax.plot(episodes[:len(moving_avg)], moving_avg, 
                   color='red', linewidth=2, label=f'{window}-Episode Moving Avg')
        
        # Highlight improvement phases
        if len(rewards) > 100:
            improvement_phases = self._detect_improvement_phases(rewards)
            for start, end in improvement_phases:
                ax.axvspan(episodes[start], episodes[end], 
                          alpha=0.2, color='green', label='Improvement Phase' if start == improvement_phases[0][0] else '')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Total Reward', fontsize=12)
        ax.set_title('Reward Progression: Learning from Experience', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        # Add annotations for key milestones
        if rewards:
            best_idx = np.argmax(rewards)
            ax.annotate(f'Best: {rewards[best_idx]:.1f}',
                       xy=(episodes[best_idx], rewards[best_idx]),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    def _plot_lap_time_improvement(self, ax):
        """Plot lap time improvement"""
        lap_times = [lt for lt in self.data['lap_times'] if lt is not None]
        
        if not lap_times:
            ax.text(0.5, 0.5, 'No completed laps yet', 
                   ha='center', va='center', fontsize=12)
            ax.set_title('Lap Time Improvement', fontsize=14, fontweight='bold')
            return
        
        lap_numbers = list(range(1, len(lap_times) + 1))
        
        # Plot lap times
        ax.plot(lap_numbers, lap_times, marker='o', color='blue', 
               linewidth=2, markersize=6, label='Lap Time')
        
        # Plot trend line
        if len(lap_times) > 1:
            z = np.polyfit(lap_numbers, lap_times, 1)
            p = np.poly1d(z)
            ax.plot(lap_numbers, p(lap_numbers), "--", color='red', 
                   linewidth=2, label='Trend')
        
        # Highlight best lap
        best_lap_idx = np.argmin(lap_times)
        best_lap_time = lap_times[best_lap_idx]
        ax.plot(best_lap_idx + 1, best_lap_time, marker='*', 
               markersize=20, color='gold', label='Best Lap')
        
        ax.set_xlabel('Lap Number', fontsize=12)
        ax.set_ylabel('Lap Time (seconds)', fontsize=12)
        ax.set_title('Lap Time Improvement: Getting Faster!', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calculate improvement
        if len(lap_times) > 1:
            improvement = ((lap_times[0] - best_lap_time) / lap_times[0]) * 100
            ax.text(0.02, 0.98, f'Improvement: {improvement:.1f}%\nBest: {best_lap_time:.2f}s',
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def _plot_collision_rate(self, ax):
        """Plot collision rate over time"""
        episodes = self.data['episodes']
        collisions = self.data['collisions']
        
        # Calculate moving average
        window = min(20, len(collisions) // 5)
        if window > 0:
            collision_rate = self._moving_average(collisions, window)
            ax.plot(episodes[:len(collision_rate)], collision_rate, 
                   color='red', linewidth=2, label=f'{window}-Episode Avg')
        
        # Scatter plot
        ax.scatter(episodes, collisions, alpha=0.3, s=20, color='darkred')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Number of Collisions', fontsize=12)
        ax.set_title('Collision Rate: Learning to Avoid Crashes', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        if collisions:
            early_collisions = np.mean(collisions[:len(collisions)//4])
            late_collisions = np.mean(collisions[-len(collisions)//4:])
            reduction = ((early_collisions - late_collisions) / early_collisions) * 100 if early_collisions > 0 else 0
            
            stats_text = f'Early avg: {early_collisions:.2f}\nLate avg: {late_collisions:.2f}\nReduction: {reduction:.1f}%'
            ax.text(0.98, 0.98, stats_text,
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    def _plot_learning_rate(self, ax):
        """Plot learning rate (reward improvement per episode)"""
        rewards = self.data['rewards']
        episodes = self.data['episodes']
        
        # Calculate episode-to-episode improvement
        improvements = []
        for i in range(1, len(rewards)):
            improvements.append(rewards[i] - rewards[i-1])
        
        if improvements:
            # Plot improvement
            ax.bar(episodes[1:], improvements, alpha=0.6, color='green', 
                  label='Episode-to-Episode Improvement')
            
            # Add zero line
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
            
            # Moving average
            window = min(20, len(improvements) // 5)
            if window > 0:
                moving_avg = self._moving_average(improvements, window)
                ax.plot(episodes[1:len(moving_avg)+1], moving_avg, 
                       color='red', linewidth=2, label=f'{window}-Episode Avg')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward Improvement', fontsize=12)
        ax.set_title('Learning Rate: How Fast is the Agent Improving?', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_success_rate(self, ax):
        """Plot success rate (lap completion)"""
        lap_times = self.data['lap_times']
        episodes = self.data['episodes']
        
        # Calculate success rate over windows
        window = min(20, len(episodes) // 5)
        if window > 0:
            success_rates = []
            window_centers = []
            
            for i in range(0, len(lap_times) - window + 1, window // 2):
                window_laps = lap_times[i:i + window]
                successes = sum(1 for lt in window_laps if lt is not None)
                success_rate = (successes / window) * 100
                success_rates.append(success_rate)
                window_centers.append(episodes[i + window // 2])
            
            ax.plot(window_centers, success_rates, marker='o', 
                   linewidth=2, markersize=8, color='green', label='Success Rate')
            
            ax.fill_between(window_centers, 0, success_rates, alpha=0.3, color='green')
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.set_title('Success Rate: Lap Completion Over Time', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 105)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add target line
        ax.axhline(y=80, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Target: 80%')
    
    def _moving_average(self, data, window):
        """Calculate moving average"""
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    def _detect_improvement_phases(self, rewards, threshold=10):
        """Detect phases where agent is consistently improving"""
        phases = []
        window = 20
        
        for i in range(len(rewards) - window):
            window_data = rewards[i:i + window]
            # Check if trending upward
            trend = np.polyfit(range(window), window_data, 1)[0]
            if trend > threshold:
                phases.append((i, i + window))
        
        # Merge overlapping phases
        if phases:
            merged = [phases[0]]
            for start, end in phases[1:]:
                if start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            return merged
        
        return []
    
    def print_summary_statistics(self):
        """Print summary statistics of training"""
        if self.data is None:
            print("No data to analyze")
            return
        
        rewards = self.data['rewards']
        lap_times = [lt for lt in self.data['lap_times'] if lt is not None]
        collisions = self.data['collisions']
        
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY STATISTICS")
        print("=" * 60)
        
        print(f"\n📊 Overall Performance:")
        print(f"  Total Episodes: {len(rewards)}")
        print(f"  Average Reward: {np.mean(rewards):.2f}")
        print(f"  Best Reward: {max(rewards):.2f}")
        print(f"  Final Reward: {rewards[-1]:.2f}")
        
        if len(rewards) >= 100:
            print(f"\n📈 Learning Progress:")
            early_avg = np.mean(rewards[:50])
            late_avg = np.mean(rewards[-50:])
            improvement = ((late_avg - early_avg) / abs(early_avg)) * 100 if early_avg != 0 else 0
            print(f"  Early Average (first 50): {early_avg:.2f}")
            print(f"  Late Average (last 50): {late_avg:.2f}")
            print(f"  Improvement: {improvement:+.1f}%")
        
        if lap_times:
            print(f"\n🏁 Lap Performance:")
            print(f"  Completed Laps: {len(lap_times)}")
            print(f"  Best Lap Time: {min(lap_times):.2f}s")
            print(f"  Average Lap Time: {np.mean(lap_times):.2f}s")
            print(f"  First Lap Time: {lap_times[0]:.2f}s")
            print(f"  Latest Lap Time: {lap_times[-1]:.2f}s")
            
            if len(lap_times) > 1:
                improvement = ((lap_times[0] - min(lap_times)) / lap_times[0]) * 100
                print(f"  Time Improvement: {improvement:.1f}%")
        
        print(f"\n💥 Collision Analysis:")
        print(f"  Total Collisions: {sum(collisions)}")
        print(f"  Average per Episode: {np.mean(collisions):.2f}")
        
        if len(collisions) >= 100:
            early_col = np.mean(collisions[:50])
            late_col = np.mean(collisions[-50:])
            reduction = ((early_col - late_col) / early_col) * 100 if early_col > 0 else 0
            print(f"  Early Collision Rate: {early_col:.2f}")
            print(f"  Late Collision Rate: {late_col:.2f}")
            print(f"  Reduction: {reduction:.1f}%")
        
        print("\n" + "=" * 60)
    
    def compare_episodes(self, episode_numbers):
        """
        Compare specific episodes to show learning
        
        Args:
            episode_numbers: List of episode numbers to compare
        """
        print("\n" + "=" * 60)
        print("EPISODE COMPARISON")
        print("=" * 60)
        
        for ep_num in episode_numbers:
            if ep_num <= len(self.data['episodes']):
                idx = ep_num - 1
                print(f"\nEpisode {ep_num}:")
                print(f"  Reward: {self.data['rewards'][idx]:.2f}")
                print(f"  Collisions: {self.data['collisions'][idx]}")
                lap_time = self.data['lap_times'][idx]
                if lap_time:
                    print(f"  Lap Time: {lap_time:.2f}s ✓")
                else:
                    print(f"  Lap Time: Did not complete")
        
        print("\n" + "=" * 60)


def main():
    """Run performance analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze CARLA RL Training Results')
    parser.add_argument('--log-path', type=str, default='checkpoints/training_log.json',
                       help='Path to training log JSON file')
    parser.add_argument('--output', type=str, default='training_analysis.png',
                       help='Output path for analysis plot')
    parser.add_argument('--compare', type=int, nargs='+', 
                       help='Episode numbers to compare (e.g., --compare 1 50 100 200)')
    
    args = parser.parse_args()
    
    analyzer = PerformanceAnalyzer(args.log_path)
    
    # Print statistics
    analyzer.print_summary_statistics()
    
    # Generate full report
    analyzer.generate_full_report(save_path=args.output)
    
    # Compare specific episodes
    if args.compare:
        analyzer.compare_episodes(args.compare)


if __name__ == "__main__":
    main()
