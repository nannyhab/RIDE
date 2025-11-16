def collect_data_point(client, track_xodr, track_id, gear_ratio, tire_friction):
    """
    Run autopilot on a track with specific parameters and measure travel time
    FIXED: Proper actor lifecycle management
    """
    print(f"  Track {track_id}, gear={gear_ratio:.2f}, friction={tire_friction:.2f}", end=" ")
    
    vehicle = None
    
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
        
        # Spawn vehicle
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        
        # Try multiple spawn points
        for attempt in range(10):
            spawn_idx = min(5 + attempt * 5, len(waypoints) - 1)
            spawn_point = waypoints[spawn_idx].transform
            spawn_point.location.z += 5.0
            
            try:
                vehicle = world.spawn_actor(vehicle_bp, spawn_point)
                
                # Disable physics temporarily
                vehicle.set_simulate_physics(False)
                world.tick()
                
                # Lower to road
                spawn_point.location.z = waypoints[spawn_idx].transform.location.z + 1.0
                vehicle.set_transform(spawn_point)
                world.tick()
                
                # Re-enable physics
                vehicle.set_simulate_physics(True)
                for _ in range(20):
                    world.tick()
                
                # Check if settled
                velocity = vehicle.get_velocity()
                if abs(velocity.z) < 0.1:
                    break
                else:
                    if vehicle.is_alive:
                        vehicle.destroy()
                    vehicle = None
                    
            except RuntimeError:
                if vehicle is not None and vehicle.is_alive:
                    vehicle.destroy()
                vehicle = None
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
        
        # Stabilize
        for _ in range(10):
            world.tick()
        
        # Enable autopilot
        vehicle.set_autopilot(True)
        
        # Autopilot warmup
        for _ in range(10):
            world.tick()
        
        # Measure
        start_time = time.time()
        start_location = vehicle.get_location()
        
        max_steps = int(MAX_RUN_TIME / 0.05)
        last_progress_time = time.time()
        prev_distance = 0.0
        stuck_counter = 0
        
        for step in range(max_steps):
            world.tick()
            
            # Check if vehicle still exists
            if not vehicle.is_alive:
                print("- Vehicle destroyed during run")
                return None, None, False
            
            current_location = vehicle.get_location()
            distance_traveled = math.sqrt(
                (current_location.x - start_location.x)**2 +
                (current_location.y - start_location.y)**2
            )
            
            # Check velocity
            velocity = vehicle.get_velocity()
            speed = math.sqrt(velocity.x**2 + velocity.y**2)
            
            if speed < 0.5:
                stuck_counter += 1
                if stuck_counter > 100:
                    print("- Stuck")
                    if vehicle.is_alive:
                        vehicle.destroy()
                    return None, None, False
            else:
                stuck_counter = 0
            
            # Check progress
            if distance_traveled - prev_distance > 0.5:
                last_progress_time = time.time()
                prev_distance = distance_traveled
            elif time.time() - last_progress_time > 15.0:
                print("- No progress")
                if vehicle.is_alive:
                    vehicle.destroy()
                return None, None, False
            
            # Check completion
            if distance_traveled > track_features['total_length'] * 0.8:
                break
        
        travel_time = time.time() - start_time
        
        # Cleanup safely
        if vehicle is not None and vehicle.is_alive:
            vehicle.set_autopilot(False)
            world.tick()
            vehicle.destroy()
        
        time.sleep(0.5)
        
        print(f"- {travel_time:.2f}s ✓")
        
        return track_features, travel_time, True
        
    except Exception as e:
        print(f"- Error: {e}")
        if vehicle is not None:
            try:
                if vehicle.is_alive:
                    vehicle.destroy()
            except:
                pass
        return None, None, False
