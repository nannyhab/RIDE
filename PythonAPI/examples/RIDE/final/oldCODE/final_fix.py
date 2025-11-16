def collect_all_data():
    """Main data collection loop - FIXED to reuse worlds"""
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
        print("Starting fresh...")
    
    client = carla.Client('localhost', 2000)
    client.set_timeout(30.0)
    generator = RandomTrackGenerator()
    
    print("="*60)
    print("COLLECTING DATA")
    print("="*60)
    
    for track_id in range(NUM_RANDOM_TRACKS):
        if track_id in completed_tracks:
            print(f"\nTrack {track_id + 1}/{NUM_RANDOM_TRACKS} - SKIPPED")
            continue
        
        print(f"\nTrack {track_id + 1}/{NUM_RANDOM_TRACKS}")
        
        # Generate XODR
        track_xodr = generator.generate_random_track(track_id)
        
        # Generate world ONCE
        params = carla.OpendriveGenerationParameters(2.0, 50.0, 1.0, 0.6, True, True, True)
        world = client.generate_opendrive_world(track_xodr, params)
        
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)
        
        # Get waypoints and features ONCE
        waypoints = get_track_waypoints(world)
        if len(waypoints) < 10:
            print("  ERROR: Not enough waypoints")
            continue
        
        extractor = TrackFeatureExtractor()
        track_features = extractor.extract_features(waypoints)
        if not track_features:
            print("  ERROR: Could not extract features")
            continue
        
        print(f"  Track: {track_features['total_length']:.0f}m, {track_features['tight_corners_pct']*100:.0f}% curves")
        
        # Test ALL parameters on THIS world
        track_results = []
        
        for gear_ratio in GEAR_RATIOS:
            for tire_friction in TIRE_FRICTIONS:
                result = run_single_test(world, waypoints, track_features, 
                                        track_id, gear_ratio, tire_friction)
                if result:
                    track_results.append(result)
        
        dataset.extend(track_results)
        
        # Save checkpoint
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"  ✓ Saved {len(track_results)} results ({len(dataset)} total)")
    
    # Final save
    with open(f'{DATA_DIR}/dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    with open(f'{DATA_DIR}/dataset.json', 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n✓ Collected {len(dataset)} data points")
    return dataset


def run_single_test(world, waypoints, track_features, track_id, gear_ratio, tire_friction):
    """Run ONE test on existing world - NO world generation here!"""
    print(f"    gear={gear_ratio:.1f}, friction={tire_friction:.1f}", end=" ")
    
    vehicle = None
    
    try:
        # Spawn vehicle
        bp_lib = world.get_blueprint_library()
        vehicle_bp = bp_lib.filter('vehicle.tesla.model3')[0]
        
        spawn_point = waypoints[5].transform
        spawn_point.location.z += 2.0
        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        
        # Settle
        vehicle.set_simulate_physics(False)
        world.tick()
        spawn_point.location.z = waypoints[5].transform.location.z + 0.5
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
        start_loc = vehicle.get_location()
        
        stuck_counter = 0
        completed = False
        
        for step in range(2400):  # 120 seconds
            world.tick()
            
            loc = vehicle.get_location()
            vel = vehicle.get_velocity()
            
            dist = math.sqrt((loc.x - start_loc.x)**2 + (loc.y - start_loc.y)**2)
            speed = math.sqrt(vel.x**2 + vel.y**2)
            
            if speed < 0.5:
                stuck_counter += 1
                if stuck_counter > 100:
                    print("STUCK")
                    vehicle.set_autopilot(False)
                    world.tick()
                    vehicle.destroy()
                    world.tick()
                    return None
            else:
                stuck_counter = 0
            
            if dist > track_features['total_length'] * 0.8:
                completed = True
                break
        
        travel_time = time.time() - start_time
        
        # Cleanup
        vehicle.set_autopilot(False)
        world.tick()
        vehicle.destroy()
        world.tick()
        
        if not completed:
            print("TIMEOUT")
            return None
        
        print(f"{travel_time:.1f}s ✓")
        
        return {
            'track_id': track_id,
            'gear_ratio': gear_ratio,
            'tire_friction': tire_friction,
            'travel_time': travel_time,
            **track_features
        }
        
    except Exception as e:
        print(f"ERROR: {e}")
        if vehicle:
            try:
                vehicle.destroy()
                world.tick()
            except:
                pass
        return None
