"""
Real-time Feedback Overlay for CARLA RL Training
Displays training metrics and performance data on screen
"""

import pygame
import numpy as np
from collections import deque
import time

class FeedbackOverlay:
    """
    Display real-time feedback during CARLA RL training
    Shows reward, speed, progress, lap times, etc.
    """
    
    def __init__(self, width=800, height=600, title="CARLA RL Training Feedback"):
        """
        Initialize feedback overlay window
        
        Args:
            width: Window width
            height: Window height
            title: Window title
        """
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(title)
        
        # Fonts
        self.title_font = pygame.font.Font(None, 48)
        self.large_font = pygame.font.Font(None, 36)
        self.medium_font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 20)
        
        # Colors
        self.BLACK = (0, 0, 0)
        self.WHITE = (255, 255, 255)
        self.GREEN = (0, 255, 0)
        self.RED = (255, 0, 0)
        self.YELLOW = (255, 255, 0)
        self.BLUE = (0, 150, 255)
        self.GRAY = (128, 128, 128)
        self.DARK_GRAY = (50, 50, 50)
        self.ORANGE = (255, 165, 0)
        
        # History for graphs
        self.reward_history = deque(maxlen=200)
        self.speed_history = deque(maxlen=200)
        self.episode_rewards = deque(maxlen=50)
        self.lap_times = deque(maxlen=20)
        
        # Episode tracking
        self.current_episode = 0
        self.total_episodes = 0
        
        # Clock for FPS
        self.clock = pygame.time.Clock()
    
    def update(self, feedback_data, episode_num=None):
        """
        Update display with new feedback data
        
        Args:
            feedback_data: Dictionary containing feedback metrics
            episode_num: Current episode number
        """
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        
        # Update episode
        if episode_num is not None:
            self.current_episode = episode_num
        
        # Clear screen
        self.screen.fill(self.BLACK)
        
        # Draw all components
        self._draw_header(feedback_data)
        self._draw_main_metrics(feedback_data)
        self._draw_progress_bar(feedback_data)
        self._draw_reward_graph()
        self._draw_speed_graph()
        self._draw_status_indicators(feedback_data)
        self._draw_lap_times()
        
        # Update history
        self.reward_history.append(feedback_data.get('total_reward', 0))
        self.speed_history.append(feedback_data.get('current_speed', 0))
        
        # Update display
        pygame.display.flip()
        self.clock.tick(30)  # 30 FPS
        
        return True
    
    def _draw_header(self, feedback_data):
        """Draw header with episode info"""
        header_text = f"Episode {self.current_episode}"
        text_surface = self.title_font.render(header_text, True, self.BLUE)
        self.screen.blit(text_surface, (20, 10))
        
        # Step counter
        step_text = f"Step: {feedback_data.get('episode_step', 0)}"
        step_surface = self.medium_font.render(step_text, True, self.WHITE)
        self.screen.blit(step_surface, (20, 55))
    
    def _draw_main_metrics(self, feedback_data):
        """Draw main performance metrics"""
        y_offset = 100
        x_left = 20
        x_right = self.width // 2 + 20
        
        # Left column - Reward metrics
        self._draw_metric_box(
            "Current Reward", 
            f"{feedback_data.get('total_reward', 0):.1f}",
            x_left, y_offset, 
            self._get_reward_color(feedback_data.get('total_reward', 0))
        )
        
        self._draw_metric_box(
            "Avg Reward (Recent)", 
            f"{feedback_data.get('recent_avg_reward', 0):.2f}",
            x_left, y_offset + 80,
            self.YELLOW
        )
        
        # Right column - Speed metrics
        current_speed = feedback_data.get('current_speed', 0)
        self._draw_metric_box(
            "Speed", 
            f"{current_speed:.1f} km/h",
            x_right, y_offset,
            self._get_speed_color(current_speed)
        )
        
        self._draw_metric_box(
            "Avg Speed", 
            f"{feedback_data.get('avg_speed', 0):.1f} km/h",
            x_right, y_offset + 80,
            self.BLUE
        )
        
        # Distance metrics
        self._draw_metric_box(
            "Distance to Checkpoint", 
            f"{feedback_data.get('distance_to_checkpoint', 0):.1f}m",
            x_left, y_offset + 160,
            self.WHITE
        )
        
        self._draw_metric_box(
            "Distance from Center", 
            f"{feedback_data.get('distance_from_center', 0):.2f}m",
            x_right, y_offset + 160,
            self._get_distance_color(feedback_data.get('distance_from_center', 0))
        )
    
    def _draw_metric_box(self, label, value, x, y, color):
        """Draw a metric box with label and value"""
        # Draw background
        box_width = 350
        box_height = 60
        pygame.draw.rect(
            self.screen, 
            self.DARK_GRAY, 
            (x, y, box_width, box_height),
            border_radius=5
        )
        pygame.draw.rect(
            self.screen, 
            self.GRAY, 
            (x, y, box_width, box_height),
            2,
            border_radius=5
        )
        
        # Draw label
        label_surface = self.small_font.render(label, True, self.GRAY)
        self.screen.blit(label_surface, (x + 10, y + 8))
        
        # Draw value
        value_surface = self.large_font.render(value, True, color)
        self.screen.blit(value_surface, (x + 10, y + 28))
    
    def _draw_progress_bar(self, feedback_data):
        """Draw checkpoint progress bar"""
        y_pos = 340
        bar_width = self.width - 40
        bar_height = 40
        x_pos = 20
        
        # Background
        pygame.draw.rect(
            self.screen,
            self.DARK_GRAY,
            (x_pos, y_pos, bar_width, bar_height),
            border_radius=5
        )
        
        # Progress fill
        progress = feedback_data.get('progress', 0) / 100.0
        fill_width = int(bar_width * progress)
        if fill_width > 0:
            pygame.draw.rect(
                self.screen,
                self.GREEN,
                (x_pos, y_pos, fill_width, bar_height),
                border_radius=5
            )
        
        # Border
        pygame.draw.rect(
            self.screen,
            self.WHITE,
            (x_pos, y_pos, bar_width, bar_height),
            2,
            border_radius=5
        )
        
        # Text
        checkpoint_text = (f"Checkpoint: {feedback_data.get('current_checkpoint', 0)}/"
                          f"{feedback_data.get('total_checkpoints', 0)}")
        text_surface = self.medium_font.render(checkpoint_text, True, self.WHITE)
        text_rect = text_surface.get_rect(center=(x_pos + bar_width // 2, y_pos + bar_height // 2))
        self.screen.blit(text_surface, text_rect)
    
    def _draw_reward_graph(self):
        """Draw reward history graph"""
        if len(self.reward_history) < 2:
            return
        
        graph_x = 20
        graph_y = 400
        graph_width = (self.width - 60) // 2
        graph_height = 150
        
        # Title
        title = self.medium_font.render("Reward History", True, self.WHITE)
        self.screen.blit(title, (graph_x, graph_y - 30))
        
        # Draw background
        pygame.draw.rect(
            self.screen,
            self.DARK_GRAY,
            (graph_x, graph_y, graph_width, graph_height),
            border_radius=5
        )
        
        # Draw graph
        rewards = list(self.reward_history)
        if len(rewards) > 1:
            max_reward = max(rewards) if max(rewards) > 0 else 1
            min_reward = min(rewards) if min(rewards) < 0 else 0
            range_reward = max_reward - min_reward
            if range_reward == 0:
                range_reward = 1
            
            points = []
            for i, reward in enumerate(rewards):
                x = graph_x + int(i * graph_width / len(rewards))
                # Normalize to graph height
                normalized = (reward - min_reward) / range_reward
                y = graph_y + graph_height - int(normalized * graph_height)
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, self.GREEN, False, points, 2)
        
        # Draw border
        pygame.draw.rect(
            self.screen,
            self.WHITE,
            (graph_x, graph_y, graph_width, graph_height),
            2,
            border_radius=5
        )
        
        # Draw axis labels
        max_label = self.small_font.render(f"{max(rewards) if rewards else 0:.0f}", True, self.GRAY)
        self.screen.blit(max_label, (graph_x + 5, graph_y + 5))
        
        min_label = self.small_font.render(f"{min(rewards) if rewards else 0:.0f}", True, self.GRAY)
        self.screen.blit(min_label, (graph_x + 5, graph_y + graph_height - 20))
    
    def _draw_speed_graph(self):
        """Draw speed history graph"""
        if len(self.speed_history) < 2:
            return
        
        graph_x = self.width // 2 + 10
        graph_y = 400
        graph_width = (self.width - 60) // 2
        graph_height = 150
        
        # Title
        title = self.medium_font.render("Speed History", True, self.WHITE)
        self.screen.blit(title, (graph_x, graph_y - 30))
        
        # Draw background
        pygame.draw.rect(
            self.screen,
            self.DARK_GRAY,
            (graph_x, graph_y, graph_width, graph_height),
            border_radius=5
        )
        
        # Draw graph
        speeds = list(self.speed_history)
        if len(speeds) > 1:
            max_speed = max(speeds) if max(speeds) > 0 else 100
            
            points = []
            for i, speed in enumerate(speeds):
                x = graph_x + int(i * graph_width / len(speeds))
                y = graph_y + graph_height - int((speed / max_speed) * graph_height)
                points.append((x, y))
            
            if len(points) > 1:
                pygame.draw.lines(self.screen, self.BLUE, False, points, 2)
        
        # Draw border
        pygame.draw.rect(
            self.screen,
            self.WHITE,
            (graph_x, graph_y, graph_width, graph_height),
            2,
            border_radius=5
        )
        
        # Draw axis labels
        max_label = self.small_font.render(f"{max(speeds) if speeds else 0:.0f}", True, self.GRAY)
        self.screen.blit(max_label, (graph_x + 5, graph_y + 5))
    
    def _draw_status_indicators(self, feedback_data):
        """Draw status indicators for collisions, lap completion, etc."""
        y_pos = self.height - 40
        x_pos = 20
        
        # Collision indicator
        collision_count = feedback_data.get('collisions', 0)
        collision_color = self.RED if collision_count > 0 else self.GREEN
        collision_text = f"Collisions: {collision_count}"
        collision_surface = self.medium_font.render(collision_text, True, collision_color)
        self.screen.blit(collision_surface, (x_pos, y_pos))
        
        # Lap completion indicator
        if feedback_data.get('lap_complete', False):
            lap_text = "LAP COMPLETE!"
            lap_surface = self.large_font.render(lap_text, True, self.GREEN)
            lap_rect = lap_surface.get_rect(center=(self.width // 2, y_pos + 10))
            self.screen.blit(lap_surface, lap_rect)
        
        # Best lap time
        best_lap = feedback_data.get('best_lap_time')
        if best_lap:
            best_text = f"Best Lap: {best_lap:.2f}s"
            best_surface = self.medium_font.render(best_text, True, self.YELLOW)
            self.screen.blit(best_surface, (self.width - 250, y_pos))
    
    def _draw_lap_times(self):
        """Draw recent lap times"""
        if not self.lap_times:
            return
        
        x_pos = self.width - 200
        y_pos = 100
        
        title = self.medium_font.render("Recent Laps", True, self.WHITE)
        self.screen.blit(title, (x_pos, y_pos))
        
        y_offset = y_pos + 35
        for i, lap_time in enumerate(list(self.lap_times)[-5:]):
            lap_text = f"{i+1}. {lap_time:.2f}s"
            lap_surface = self.small_font.render(lap_text, True, self.GRAY)
            self.screen.blit(lap_surface, (x_pos, y_offset))
            y_offset += 25
    
    def _get_reward_color(self, reward):
        """Get color based on reward value"""
        if reward > 100:
            return self.GREEN
        elif reward > 0:
            return self.YELLOW
        else:
            return self.RED
    
    def _get_speed_color(self, speed):
        """Get color based on speed"""
        if speed > 60:
            return self.GREEN
        elif speed > 30:
            return self.YELLOW
        else:
            return self.RED
    
    def _get_distance_color(self, distance):
        """Get color based on distance from center"""
        if distance < 2:
            return self.GREEN
        elif distance < 4:
            return self.YELLOW
        else:
            return self.RED
    
    def add_lap_time(self, lap_time):
        """Add a lap time to history"""
        self.lap_times.append(lap_time)
    
    def close(self):
        """Close the display"""
        pygame.quit()


# Example usage
if __name__ == "__main__":
    overlay = FeedbackOverlay()
    
    # Simulate training data
    running = True
    step = 0
    total_reward = 0
    
    while running and step < 500:
        step += 1
        
        # Simulate feedback data
        reward_delta = np.random.uniform(-2, 5)
        total_reward += reward_delta
        
        feedback_data = {
            'episode_step': step,
            'total_reward': total_reward,
            'recent_avg_reward': reward_delta,
            'current_speed': 40 + np.random.uniform(-10, 20),
            'avg_speed': 45,
            'distance_to_checkpoint': 15 + np.random.uniform(-5, 5),
            'distance_from_center': np.random.uniform(0, 5),
            'current_checkpoint': (step // 50) % 10,
            'total_checkpoints': 10,
            'progress': ((step // 50) % 10) * 10,
            'lap_complete': (step % 100) == 0,
            'best_lap_time': 95.3,
            'current_lap_time': step * 0.1,
            'collisions': 0 if np.random.random() > 0.05 else 1,
            'lane_invasions': 0,
        }
        
        running = overlay.update(feedback_data, episode_num=1)
        
        if feedback_data['lap_complete']:
            overlay.add_lap_time(step * 0.1)
        
        time.sleep(0.033)  # ~30 FPS
    
    overlay.close()
