# Replace the collect_data_point function in track_param_optimizer_robust.py

def collect_data_point(client, track_xodr, track_id, gear_ratio, tire_friction):
    """
    Run autopilot on a track with specific parameters and measure travel time
    FIXED: Better spawn logic to prevent stuck vehicles
    """
    print(f"  Track {track_id}, gear={gear_ratio:.2f}, friction={tire_friction:.2f}", end=" ")
    
    try:
        # Generate world from XODR
        params = carla.OpendriveGenerationParameters(
            vertex_distance=2.0,
            max_road_length=50.0,
            wall_height=1.0,
            additional_width=0.6,
            smooth_junctions=True,
            enable_mesh_visibility=True,
            enable_pedestrian_navigation=True
        )
        
        world = client.generate_opendrive_world(track_xodr, params)
        
        # Sync mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        # Get waypoints
        waypoints = get_track_waypoints(world)
        
        if len(waypoints) < 10:
            print("- No waypoints")
            return None, None, False
        
        # Extract track features
        extractor = TrackFeatureExtractor()
        track_features = extractor.extract_features(waypoints)
        
        if track_features is None:
            print("- No features")
            return None, None, False
        
        # Spawn vehicle - IMPROVED LOGIC
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        
        # Try multiple spawn points along the track
        vehicle = None
        for attempt in range(10):
            spawn_idx = min(5 + attempt * 5, len(waypoints) - 1)
            spawn_point = waypoints[spawn_idx].transform
            spawn_point.location.z += 5.0  # Start high
            
            try:
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                
                # Disable physics temporarily
                vehicle.set_simulate_physics(False)
                world.tick()
                
                # Lower to road level
                spawn_point.location.z = waypoints[spawn_idx].transform.location.z + 1.0
                vehicle.set_transform(spawn_point)
                world.tick()
                
                # Re-enable physics and let settle
                vehicle.set_simulate_physics(True)
                for _ in range(20):
                    world.tick()
                
                # Check if vehicle is actually on ground and not stuck
                velocity = vehicle.get_velocity()
                if abs(velocity.z) < 0.1:  # Not falling
                    break
                else:
                    vehicle.destroy()
                    vehicle = None
                    
            except RuntimeError:
                continue
        
        if vehicle is None:
            print("- Spawn failed")
            return None, None, False
        
        # Set physical parameters
        physics = vehicle.get_physics_control()
        physics.max_rpm = 5000.0 * gear_ratio / 3.0
        
        for wheel in physics.wheels:
            wheel.tire_friction = tire_friction
        
        vehicle.apply_physics_control(physics)
        
        # Wait for physics to stabilize
        for _ in range(10):
            world.tick()
        
        # Enable AUTOPILOT
        vehicle.set_autopilot(True)
        
        # Give autopilot time to start
        for _ in range(10):
            world.tick()
        
        # Run and measure time
        start_time = time.time()
        start_location = vehicle.get_location()
        
        max_steps = int(MAX_RUN_TIME / 0.05)
        last_progress_time = time.time()
        prev_distance = 0.0
        stuck_counter = 0
        
        for step in range(max_steps):
            world.tick()
            
            # Check progress
            current_location = vehicle.get_location()
            distance_traveled = math.sqrt(
                (current_location.x - start_location.x)**2 +
                (current_location.y - start_location.y)**2
            )
            
            # Check velocity to detect stuck
            velocity = vehicle.get_velocity()
            speed = math.sqrt(velocity.x**2 + velocity.y**2)
            
            if speed < 0.5:  # Moving very slowly
                stuck_counter += 1
                if stuck_counter > 100:  # Stuck for 5 seconds
                    print("- Stuck (no movement)")
                    vehicle.destroy()
                    return None, None, False
            else:
                stuck_counter = 0
            
            # Check if making progress
            if distance_traveled - prev_distance > 0.5:
                last_progress_time = time.time()
                prev_distance = distance_traveled
            elif time.time() - last_progress_time > 15.0:
                print("- Stuck (no progress)")
                vehicle.destroy()
                return None, None, False
            
            # Check if completed
            if distance_traveled > track_features['total_length'] * 0.8:
                break
        
        travel_time = time.time() - start_time
        
        # Cleanup
        vehicle.destroy()
        time.sleep(0.5)
        
        print(f"- {travel_time:.2f}s")
        
        return track_features, travel_time, True
        
    except Exception as e:
        print(f"- Error: {e}")
        try:
            if vehicle is not None:
                vehicle.destroy()
        except:
            pass
        return None, None, False
