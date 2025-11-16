def collect_all_data():
    """
    Main data collection loop WITH CHECKPOINTING
    FIXED: Don't generate new world until after data is saved
    """
    # Load existing data if available
    checkpoint_file = f'{DATA_DIR}/dataset_checkpoint.pkl'
    
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            dataset = pickle.load(f)
        
        completed_tracks = set(d['track_id'] for d in dataset)
        print(f"✓ RESUMING: Found {len(dataset)} existing data points")
        print(f"✓ Completed tracks: {sorted(completed_tracks)}")
    else:
        dataset = []
        completed_tracks = set()
        print("Starting fresh data collection...")
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(30.0)
    
    generator = RandomTrackGenerator()
    
    print("="*60)
    print("COLLECTING DATA")
    print("="*60)
    
    for track_id in range(NUM_RANDOM_TRACKS):
        if track_id in completed_tracks:
            print(f"\nTrack {track_id + 1}/{NUM_RANDOM_TRACKS} - SKIPPED (already done)")
            continue
        
        print(f"\nTrack {track_id + 1}/{NUM_RANDOM_TRACKS}")
        
        # Generate random track ONCE for this track_id
        track_xodr = generator.generate_random_track(track_id)
        
        # Collect results for ALL parameter combinations on this track
        track_results = []
        
        for gear_ratio in GEAR_RATIOS:
            for tire_friction in TIRE_FRICTIONS:
                features, travel_time, success = collect_data_point_single(
                    client, track_xodr, track_id, gear_ratio, tire_friction
                )
                
                if success and features is not None:
                    track_results.append({
                        'track_id': track_id,
                        'gear_ratio': gear_ratio,
                        'tire_friction': tire_friction,
                        'travel_time': travel_time,
                        **features
                    })
        
        # Add all results for this track to dataset
        dataset.extend(track_results)
        
        # CHECKPOINT: Save after each track
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"  ✓ Checkpoint saved ({len(dataset)} total points, {len(track_results)} from this track)")
    
    # Final save
    with open(f'{DATA_DIR}/dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    
    with open(f'{DATA_DIR}/dataset.json', 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n✓ Collected {len(dataset)} data points")
    print(f"✓ Saved to {DATA_DIR}/")
    
    return dataset


def collect_data_point_single(client, track_xodr, track_id, gear_ratio, tire_friction):
    """
    Run SINGLE test on existing world
    Returns: features, travel_time, success
    """
    print(f"  Track {track_id}, gear={gear_ratio:.2f}, friction={tire_friction:.2f}", end=" ")
    
    vehicle = None
    world = None
    
    try:
        # Generate world
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
        
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        # Get waypoints (only once per track)
        waypoints = get_track_waypoints(world)
        
        if len(waypoints) < 10:
            print("- No waypoints")
            return None, None, False
        
        # Extract features (only once per track)
        extractor = TrackFeatureExtractor()
        track_features = extractor.extract_features(waypoints)
        
        if track_features is None:
            print("- No features")
            return None, None, False
        
        # Spawn vehicle
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        
        spawn_idx = 5
        spawn_point = waypoints[spawn_idx].transform
        spawn_point.location.z += 3.0
        
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        
        vehicle.set_simulate_physics(False)
        world.tick()
        
        spawn_point.location.z = waypoints[spawn_idx].transform.location.z + 0.5
        vehicle.set_transform(spawn_point)
        world.tick()
        
        vehicle.set_simulate_physics(True)
        for _ in range(20):
            world.tick()
        
        # Set physics
        physics = vehicle.get_physics_control()
        physics.max_rpm = 5000.0 * gear_ratio / 3.0
        for wheel in physics.wheels:
            wheel.tire_friction = tire_friction
        vehicle.apply_physics_control(physics)
        
        for _ in range(10):
            world.tick()
        
        # Autopilot
        vehicle.set_autopilot(True)
        for _ in range(10):
            world.tick()
        
        # Run
        start_time = time.time()
        start_location = vehicle.get_location()
        
        max_steps = int(MAX_RUN_TIME / 0.05)
        stuck_counter = 0
        completed = False
        
        for step in range(max_steps):
            world.tick()
            
            current_location = vehicle.get_location()
            velocity = vehicle.get_velocity()
            
            distance_traveled = math.sqrt(
                (current_location.x - start_location.x)**2 +
                (current_location.y - start_location.y)**2
            )
            
            speed = math.sqrt(velocity.x**2 + velocity.y**2)
            
            if speed < 0.5:
                stuck_counter += 1
                if stuck_counter > 100:
                    print("- Stuck")
                    vehicle.set_autopilot(False)
                    world.tick()
                    vehicle.destroy()
                    world.tick()
                    return None, None, False
            else:
                stuck_counter = 0
            
            # Check completion
            if distance_traveled > track_features['total_length'] * 0.8:
                completed = True
                break
        
        # RECORD TIME IMMEDIATELY
        travel_time = time.time() - start_time
        
        # Cleanup
        vehicle.set_autopilot(False)
        world.tick()
        vehicle.destroy()
        world.tick()
        
        if not completed:
            print("- Timeout")
            return None, None, False
        
        print(f"- {travel_time:.2f}s ✓")
        
        # Return immediately with data
        return track_features, travel_time, True
        
    except Exception as e:
        print(f"- Error: {e}")
        if vehicle is not None:
            try:
                vehicle.destroy()
            except:
                pass
        return None, None, False
