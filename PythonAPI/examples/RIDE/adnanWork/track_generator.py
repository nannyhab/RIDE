"""
CARLA Custom Track Generator
Creates custom racing tracks with checkpoints and configurable difficulty
"""

import carla
import math
import numpy as np
import json


class TrackGenerator:
    """
    Generate custom racing tracks in CARLA
    Supports multiple track types and difficulty levels
    """
    
    def __init__(self, client, world=None):
        """
        Initialize track generator
        
        Args:
            client: CARLA client
            world: CARLA world (optional)
        """
        self.client = client
        self.world = world or client.get_world()
        self.map = self.world.get_map()
        self.checkpoint_markers = []
    
    def generate_oval_track(self, 
                           center_x=0, 
                           center_y=0, 
                           major_radius=60, 
                           minor_radius=40, 
                           num_checkpoints=20,
                           save_path='oval_track_checkpoints.txt'):
        """
        Generate an oval racing track
        
        Args:
            center_x, center_y: Center of oval
            major_radius: Radius along x-axis
            minor_radius: Radius along y-axis
            num_checkpoints: Number of checkpoints
            save_path: File to save checkpoints
        
        Returns:
            List of checkpoint locations
        """
        checkpoints = []
        
        for i in range(num_checkpoints):
            # Parametric oval equation
            angle = 2 * math.pi * i / num_checkpoints
            x = center_x + major_radius * math.cos(angle)
            y = center_y + minor_radius * math.sin(angle)
            z = 0.5  # Slightly above ground
            
            checkpoints.append(carla.Location(x=x, y=y, z=z))
        
        # Save to file
        self._save_checkpoints(checkpoints, save_path)
        
        print(f"Generated oval track with {num_checkpoints} checkpoints")
        print(f"Track dimensions: {major_radius*2}m x {minor_radius*2}m")
        print(f"Saved to: {save_path}")
        
        return checkpoints
    
    def generate_figure_eight_track(self,
                                   center_x=0,
                                   center_y=0,
                                   radius=50,
                                   num_checkpoints=24,
                                   save_path='figure8_track_checkpoints.txt'):
        """
        Generate a figure-8 racing track
        
        Args:
            center_x, center_y: Center point
            radius: Radius of each loop
            num_checkpoints: Number of checkpoints
            save_path: File to save checkpoints
        
        Returns:
            List of checkpoint locations
        """
        checkpoints = []
        
        for i in range(num_checkpoints):
            t = 4 * math.pi * i / num_checkpoints
            
            # Figure-8 parametric equations (Lemniscate)
            scale = radius / (1 + math.sin(t)**2)
            x = center_x + scale * math.cos(t)
            y = center_y + scale * math.sin(t) * math.cos(t)
            z = 0.5
            
            checkpoints.append(carla.Location(x=x, y=y, z=z))
        
        self._save_checkpoints(checkpoints, save_path)
        
        print(f"Generated figure-8 track with {num_checkpoints} checkpoints")
        print(f"Saved to: {save_path}")
        
        return checkpoints
    
    def generate_technical_circuit(self,
                                   start_x=0,
                                   start_y=0,
                                   num_sections=8,
                                   section_length=30,
                                   save_path='technical_circuit_checkpoints.txt'):
        """
        Generate a technical circuit with varied corners
        
        Args:
            start_x, start_y: Starting position
            num_sections: Number of track sections
            section_length: Average length of each section
            save_path: File to save checkpoints
        
        Returns:
            List of checkpoint locations
        """
        checkpoints = []
        
        # Current position and heading
        current_x = start_x
        current_y = start_y
        current_heading = 0  # radians
        
        checkpoints.append(carla.Location(x=current_x, y=current_y, z=0.5))
        
        # Generate sections with varying difficulty
        for i in range(num_sections):
            # Vary turn angles (mix of easy and hard corners)
            if i % 3 == 0:
                # Hairpin turn
                turn_angle = np.random.choice([90, -90])
                segment_length = section_length * 0.7
            elif i % 3 == 1:
                # Medium corner
                turn_angle = np.random.choice([45, -45, 60, -60])
                segment_length = section_length
            else:
                # Sweeper/chicane
                turn_angle = np.random.choice([20, -20, 30, -30])
                segment_length = section_length * 1.2
            
            # Add checkpoints along the section
            num_checkpoints_section = int(segment_length / 10)  # ~10m between checkpoints
            
            for j in range(num_checkpoints_section):
                # Gradual turn
                current_heading += math.radians(turn_angle / num_checkpoints_section)
                
                # Move forward
                current_x += (segment_length / num_checkpoints_section) * math.cos(current_heading)
                current_y += (segment_length / num_checkpoints_section) * math.sin(current_heading)
                
                checkpoints.append(carla.Location(x=current_x, y=current_y, z=0.5))
        
        # Close the loop by heading back to start
        dx = start_x - current_x
        dy = start_y - current_y
        closing_distance = math.sqrt(dx**2 + dy**2)
        
        num_closing_checkpoints = max(3, int(closing_distance / 10))
        for j in range(1, num_closing_checkpoints + 1):
            t = j / num_closing_checkpoints
            x = current_x + t * dx
            y = current_y + t * dy
            checkpoints.append(carla.Location(x=x, y=y, z=0.5))
        
        self._save_checkpoints(checkpoints, save_path)
        
        print(f"Generated technical circuit with {len(checkpoints)} checkpoints")
        print(f"Number of sections: {num_sections}")
        print(f"Saved to: {save_path}")
        
        return checkpoints
    
    def generate_mountain_pass(self,
                              start_x=0,
                              start_y=0,
                              length=200,
                              elevation_change=20,
                              num_switchbacks=4,
                              save_path='mountain_pass_checkpoints.txt'):
        """
        Generate a mountain pass with elevation changes
        
        Args:
            start_x, start_y: Starting position
            length: Total track length
            elevation_change: Total elevation gain/loss
            num_switchbacks: Number of switchback turns
            save_path: File to save checkpoints
        
        Returns:
            List of checkpoint locations
        """
        checkpoints = []
        
        num_checkpoints = int(length / 10)  # ~10m spacing
        
        for i in range(num_checkpoints):
            t = i / num_checkpoints
            
            # Zigzag pattern for switchbacks
            if num_switchbacks > 0:
                zigzag_amplitude = 20
                x = start_x + length * t + zigzag_amplitude * math.sin(2 * math.pi * num_switchbacks * t)
                y = start_y + zigzag_amplitude * math.cos(2 * math.pi * num_switchbacks * t)
            else:
                x = start_x + length * t
                y = start_y
            
            # Elevation profile (smooth hill)
            z = 0.5 + elevation_change * math.sin(math.pi * t)
            
            checkpoints.append(carla.Location(x=x, y=y, z=z))
        
        self._save_checkpoints(checkpoints, save_path)
        
        print(f"Generated mountain pass with {len(checkpoints)} checkpoints")
        print(f"Length: {length}m, Elevation change: {elevation_change}m")
        print(f"Saved to: {save_path}")
        
        return checkpoints
    
    def generate_from_waypoints(self,
                               waypoint_list,
                               interpolation_distance=10,
                               save_path='custom_track_checkpoints.txt'):
        """
        Generate track from manually specified waypoints
        
        Args:
            waypoint_list: List of (x, y, z) tuples
            interpolation_distance: Distance between interpolated checkpoints
            save_path: File to save checkpoints
        
        Returns:
            List of checkpoint locations
        """
        checkpoints = []
        
        for i in range(len(waypoint_list)):
            current = waypoint_list[i]
            next_wp = waypoint_list[(i + 1) % len(waypoint_list)]
            
            # Calculate distance between waypoints
            dx = next_wp[0] - current[0]
            dy = next_wp[1] - current[1]
            dz = next_wp[2] - current[2]
            distance = math.sqrt(dx**2 + dy**2 + dz**2)
            
            # Interpolate checkpoints
            num_interp = max(1, int(distance / interpolation_distance))
            
            for j in range(num_interp):
                t = j / num_interp
                x = current[0] + t * dx
                y = current[1] + t * dy
                z = current[2] + t * dz
                
                checkpoints.append(carla.Location(x=x, y=y, z=z))
        
        self._save_checkpoints(checkpoints, save_path)
        
        print(f"Generated track from {len(waypoint_list)} waypoints")
        print(f"Total checkpoints: {len(checkpoints)}")
        print(f"Saved to: {save_path}")
        
        return checkpoints
    
    def _save_checkpoints(self, checkpoints, filepath):
        """Save checkpoints to file"""
        with open(filepath, 'w') as f:
            for cp in checkpoints:
                f.write(f"{cp.x},{cp.y},{cp.z}\n")
    
    def visualize_track(self, checkpoints, duration=60, draw_line=True):
        """
        Visualize track in CARLA by drawing markers and lines
        
        Args:
            checkpoints: List of checkpoint locations
            duration: How long to show markers (seconds)
            draw_line: Whether to draw lines between checkpoints
        """
        debug = self.world.debug
        
        # Draw checkpoint markers
        for i, checkpoint in enumerate(checkpoints):
            # Draw sphere at checkpoint
            debug.draw_point(
                checkpoint,
                size=0.5,
                color=carla.Color(r=0, g=255, b=0),
                life_time=duration
            )
            
            # Draw checkpoint number
            debug.draw_string(
                checkpoint + carla.Location(z=2),
                str(i),
                color=carla.Color(r=255, g=255, b=255),
                life_time=duration
            )
        
        # Draw lines between checkpoints
        if draw_line:
            for i in range(len(checkpoints)):
                start = checkpoints[i]
                end = checkpoints[(i + 1) % len(checkpoints)]
                
                debug.draw_line(
                    start,
                    end,
                    thickness=0.1,
                    color=carla.Color(r=255, g=0, b=0),
                    life_time=duration
                )
        
        print(f"Track visualized for {duration} seconds")
        print(f"Total checkpoints: {len(checkpoints)}")
        print(f"Approximate track length: {self._calculate_track_length(checkpoints):.1f}m")
    
    def _calculate_track_length(self, checkpoints):
        """Calculate total track length"""
        total_length = 0
        for i in range(len(checkpoints)):
            current = checkpoints[i]
            next_cp = checkpoints[(i + 1) % len(checkpoints)]
            
            dx = next_cp.x - current.x
            dy = next_cp.y - current.y
            dz = next_cp.z - current.z
            
            total_length += math.sqrt(dx**2 + dy**2 + dz**2)
        
        return total_length
    
    def generate_difficulty_variants(self, 
                                    track_type='oval',
                                    base_path='tracks/'):
        """
        Generate easy, medium, and hard variants of a track type
        
        Args:
            track_type: Type of track ('oval', 'figure8', 'technical')
            base_path: Base directory for saving tracks
        
        Returns:
            Dictionary with difficulty levels and checkpoint files
        """
        variants = {}
        
        if track_type == 'oval':
            # Easy - large, gentle oval
            variants['easy'] = self.generate_oval_track(
                major_radius=80,
                minor_radius=60,
                num_checkpoints=16,
                save_path=f'{base_path}oval_easy.txt'
            )
            
            # Medium - standard oval
            variants['medium'] = self.generate_oval_track(
                major_radius=60,
                minor_radius=40,
                num_checkpoints=20,
                save_path=f'{base_path}oval_medium.txt'
            )
            
            # Hard - tight oval
            variants['hard'] = self.generate_oval_track(
                major_radius=40,
                minor_radius=25,
                num_checkpoints=24,
                save_path=f'{base_path}oval_hard.txt'
            )
        
        elif track_type == 'figure8':
            # Easy - large figure-8
            variants['easy'] = self.generate_figure_eight_track(
                radius=60,
                num_checkpoints=20,
                save_path=f'{base_path}figure8_easy.txt'
            )
            
            # Medium - standard figure-8
            variants['medium'] = self.generate_figure_eight_track(
                radius=45,
                num_checkpoints=24,
                save_path=f'{base_path}figure8_medium.txt'
            )
            
            # Hard - tight figure-8
            variants['hard'] = self.generate_figure_eight_track(
                radius=30,
                num_checkpoints=28,
                save_path=f'{base_path}figure8_hard.txt'
            )
        
        elif track_type == 'technical':
            # Easy - fewer, gentler corners
            variants['easy'] = self.generate_technical_circuit(
                num_sections=6,
                section_length=35,
                save_path=f'{base_path}technical_easy.txt'
            )
            
            # Medium - balanced difficulty
            variants['medium'] = self.generate_technical_circuit(
                num_sections=8,
                section_length=30,
                save_path=f'{base_path}technical_medium.txt'
            )
            
            # Hard - many sharp corners
            variants['hard'] = self.generate_technical_circuit(
                num_sections=12,
                section_length=25,
                save_path=f'{base_path}technical_hard.txt'
            )
        
        print(f"\nGenerated {len(variants)} difficulty variants for {track_type} track")
        return variants


def main():
    """Example usage of TrackGenerator"""
    # Connect to CARLA
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()
    
    # Create track generator
    generator = TrackGenerator(client, world)
    
    print("=" * 60)
    print("CARLA Custom Track Generator")
    print("=" * 60)
    
    # Menu
    print("\nSelect track type:")
    print("1. Oval Track (Easy)")
    print("2. Figure-8 Track (Challenging)")
    print("3. Technical Circuit (Advanced)")
    print("4. Mountain Pass (Elevation Changes)")
    print("5. Generate All Difficulty Variants")
    print("6. Custom Track from Waypoints")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    if choice == '1':
        checkpoints = generator.generate_oval_track()
        generator.visualize_track(checkpoints, duration=120)
    
    elif choice == '2':
        checkpoints = generator.generate_figure_eight_track()
        generator.visualize_track(checkpoints, duration=120)
    
    elif choice == '3':
        checkpoints = generator.generate_technical_circuit()
        generator.visualize_track(checkpoints, duration=120)
    
    elif choice == '4':
        checkpoints = generator.generate_mountain_pass()
        generator.visualize_track(checkpoints, duration=120)
    
    elif choice == '5':
        print("\nGenerating all difficulty variants...")
        generator.generate_difficulty_variants('oval')
        generator.generate_difficulty_variants('figure8')
        generator.generate_difficulty_variants('technical')
        print("\nAll variants generated!")
    
    elif choice == '6':
        print("\nExample custom waypoints:")
        waypoints = [
            (0, 0, 0.5),
            (50, 0, 0.5),
            (50, 50, 0.5),
            (0, 50, 0.5)
        ]
        checkpoints = generator.generate_from_waypoints(waypoints)
        generator.visualize_track(checkpoints, duration=120)
    
    else:
        print("Invalid choice")
    
    print("\nTrack generation complete!")


if __name__ == "__main__":
    main()
