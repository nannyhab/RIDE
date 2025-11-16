"""
Complete Demo Script
Demonstrates the full workflow: track generation → training → analysis
"""

import os
import sys
import time
import carla


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def check_carla_connection():
    """Check if CARLA is running"""
    print("Checking CARLA connection...")
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(5.0)
        world = client.get_world()
        print("✓ CARLA is running!")
        return True
    except Exception as e:
        print(f"✗ Cannot connect to CARLA: {e}")
        print("\nPlease start CARLA first:")
        print("  Terminal 1: ./CarlaUE4.sh")
        return False


def demo_track_generation():
    """Demo: Generate custom tracks"""
    print_header("STEP 1: Generating Custom Tracks")
    
    from track_generator import TrackGenerator
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    generator = TrackGenerator(client)
    
    print("Creating demonstration tracks...\n")
    
    # 1. Simple oval for initial learning
    print("1️⃣  Creating EASY oval track...")
    oval_checkpoints = generator.generate_oval_track(
        major_radius=70,
        minor_radius=50,
        num_checkpoints=16,
        save_path='demo_oval_easy.txt'
    )
    print("   ✓ Saved to: demo_oval_easy.txt")
    time.sleep(1)
    
    # 2. Technical circuit for advanced training
    print("\n2️⃣  Creating TECHNICAL circuit...")
    tech_checkpoints = generator.generate_technical_circuit(
        num_sections=8,
        section_length=30,
        save_path='demo_technical.txt'
    )
    print("   ✓ Saved to: demo_technical.txt")
    time.sleep(1)
    
    # 3. Visualize the oval track
    print("\n3️⃣  Visualizing oval track in CARLA...")
    print("   Look at the CARLA window to see the track markers!")
    generator.visualize_track(oval_checkpoints, duration=30, draw_line=True)
    
    print("\n✓ Track generation complete!")
    print("\nGenerated tracks:")
    print("  • demo_oval_easy.txt - For initial training")
    print("  • demo_technical.txt - For advanced training")
    
    input("\nPress Enter to continue to training demo...")
    return 'demo_oval_easy.txt'


def demo_training(track_file, num_episodes=50):
    """Demo: Train agent for a few episodes"""
    print_header("STEP 2: Training the Agent")
    
    print(f"Starting training on track: {track_file}")
    print(f"Episodes: {num_episodes}")
    print("\nThe feedback window will show:")
    print("  • Real-time rewards and metrics")
    print("  • Speed and progress indicators")
    print("  • Collision tracking")
    print("  • Lap completion status")
    print("\n⚠️  This is a demo with reduced episodes for speed")
    print("   (Normal training uses 300-500 episodes)")
    
    input("\nPress Enter to start training...")
    
    # Import and run training
    from train_agent import train_agent
    
    try:
        train_agent(
            num_episodes=num_episodes,
            checkpoint_file=track_file,
            save_frequency=10,
            render=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted. Continuing to analysis...")
    
    print("\n✓ Training demo complete!")


def demo_analysis():
    """Demo: Analyze training results"""
    print_header("STEP 3: Analyzing Training Results")
    
    log_path = 'checkpoints/training_log.json'
    
    if not os.path.exists(log_path):
        print(f"✗ Training log not found at {log_path}")
        print("  Skipping analysis demo")
        return
    
    print("Analyzing training results...\n")
    
    from analyze_performance import PerformanceAnalyzer
    
    analyzer = PerformanceAnalyzer(log_path)
    
    # Print statistics
    analyzer.print_summary_statistics()
    
    # Generate analysis plots
    print("\nGenerating analysis plots...")
    analyzer.generate_full_report(save_path='demo_training_analysis.png')
    print("✓ Analysis plot saved to: demo_training_analysis.png")
    
    # Compare specific episodes
    print("\nComparing early vs. late episodes:")
    if len(analyzer.data['episodes']) >= 30:
        episodes_to_compare = [1, 10, 20, 30]
        analyzer.compare_episodes(episodes_to_compare)
    
    print("\n✓ Analysis complete!")


def demo_feedback_explanation():
    """Explain the feedback system"""
    print_header("Understanding the Feedback System")
    
    print("""
The training feedback shows how the agent learns from failures:

📊 REWARD COMPONENTS:
  ✓ Speed Reward: Faster driving = higher reward
  ✓ Progress Reward: Moving toward checkpoints
  ✓ Checkpoint Bonus: +10 for each checkpoint reached
  ✓ Lap Completion: +20 for finishing a lap
  ✓ Best Lap Bonus: +50 for beating previous best

  ✗ Collision Penalty: -20 per crash
  ✗ Off-Track Penalty: Based on distance from center
  ✗ Lane Invasion: -5 per crossing
  ✗ Time Penalty: -0.05 per step (encourages speed)

📈 LEARNING PROGRESSION:
  Episode 1-100:   Learning basic control, many crashes
  Episode 100-300: Completing laps, improving times
  Episode 300+:    Optimized performance, minimal errors

🎯 WHAT THE AGENT LEARNS:
  • Optimal racing lines through corners
  • When to accelerate vs. brake
  • How to avoid collisions
  • Track-specific strategies

🔄 HOW IT LEARNS FROM FAILURES:
  1. Crashes → Large negative reward → Avoid similar actions
  2. Slow laps → Lower rewards → Try different strategies  
  3. Successful laps → High rewards → Repeat similar actions
  4. Best laps → Bonus rewards → Reinforce optimal behavior

The feedback window displays all these metrics in real-time,
allowing you to observe the learning process as it happens!
    """)
    
    input("Press Enter to continue...")


def demo_custom_track_explanation():
    """Explain custom track features"""
    print_header("Custom Track Customization")
    
    print("""
TRACK CUSTOMIZATION OPTIONS:

1️⃣  TRACK TYPES:
   • Oval: Simple circular layout, good for learning basics
   • Figure-8: Adds crossing point, tests planning ahead
   • Technical Circuit: Mixed corners, various difficulty
   • Mountain Pass: Elevation changes, banking effects

2️⃣  DIFFICULTY FACTORS:
   Easy:
     - Wide turns (large radius)
     - Long straights
     - Forgiving track limits
     - Fewer checkpoints
   
   Medium:
     - Mixed corner speeds
     - Standard track width
     - Moderate checkpoint density
   
   Hard:
     - Tight hairpins (small radius)
     - Short straights
     - Narrow sections
     - More checkpoints (precision required)

3️⃣  CUSTOMIZATION PARAMETERS:
   • Track length: Total distance
   • Number of sections: More = more complex
   • Section length: Shorter = tighter corners
   • Checkpoint spacing: Closer = more precise control needed
   • Elevation: Hills and banking angles

4️⃣  EXAMPLE CUSTOMIZATIONS:
   
   For beginner agent:
     generator.generate_oval_track(
         major_radius=80,    # Large, gentle turns
         minor_radius=60,
         num_checkpoints=12  # Fewer checkpoints
     )
   
   For advanced agent:
     generator.generate_technical_circuit(
         num_sections=12,    # Complex layout
         section_length=20,  # Tight corners
     )

The track determines what the agent learns:
  • Simple tracks → Basic driving skills
  • Complex tracks → Advanced racing techniques
    """)
    
    input("Press Enter to continue...")


def main():
    """Run complete demo"""
    print("""
╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║         CARLA REINFORCEMENT LEARNING RACING AGENT DEMO            ║
║                                                                    ║
║  This demo shows:                                                 ║
║    1. How to generate custom tracks with different difficulty    ║
║    2. Training an agent using reinforcement learning             ║
║    3. Real-time feedback showing learning progress               ║
║    4. Performance analysis tracking improvement                  ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\nDEMO MODE: This is a shortened demonstration.")
    print("Full training typically runs for 300-500 episodes.")
    print("This demo will run 50 episodes for speed.\n")
    
    # Check CARLA connection
    if not check_carla_connection():
        print("\n❌ Cannot proceed without CARLA running.")
        print("\nTo start CARLA:")
        print("  1. Terminal 1: srun -A p33058 -p gengpu --gres=gpu:a100:1 --mem=16G -t 2:00:00 --pty bash -l")
        print("  2. conda activate /projects/p33058/envs/carla-py39")
        print("  3. ./CarlaUE4.sh")
        return
    
    time.sleep(2)
    
    # Demo menu
    print("\nSelect demo option:")
    print("  1. Full demo (track generation + training + analysis)")
    print("  2. Just track generation")
    print("  3. Just training (requires existing track)")
    print("  4. Just analysis (requires training log)")
    print("  5. Explain feedback system")
    print("  6. Explain track customization")
    
    choice = input("\nEnter choice (1-6): ").strip()
    
    try:
        if choice == '1':
            # Full demo
            demo_feedback_explanation()
            demo_custom_track_explanation()
            track_file = demo_track_generation()
            demo_training(track_file, num_episodes=50)
            demo_analysis()
            
        elif choice == '2':
            demo_track_generation()
            
        elif choice == '3':
            track_file = input("Enter track checkpoint file (default: oval_track_checkpoints.txt): ").strip()
            if not track_file:
                track_file = 'oval_track_checkpoints.txt'
            
            if not os.path.exists(track_file):
                print(f"✗ Track file not found: {track_file}")
                return
            
            episodes = input("Number of episodes (default: 50): ").strip()
            episodes = int(episodes) if episodes else 50
            
            demo_training(track_file, num_episodes=episodes)
            
        elif choice == '4':
            demo_analysis()
            
        elif choice == '5':
            demo_feedback_explanation()
            
        elif choice == '6':
            demo_custom_track_explanation()
            
        else:
            print("Invalid choice")
            return
        
        print_header("Demo Complete!")
        print("""
Next steps:
  1. Review generated tracks in CARLA
  2. Examine training logs in checkpoints/
  3. Modify hyperparameters in train_agent.py
  4. Create your own custom tracks
  5. Run full training with 300-500 episodes

For full training:
  python train_agent.py --episodes 500 --checkpoint-file your_track.txt

For analysis:
  python analyze_performance.py

Happy training! 🏎️💨
        """)
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
