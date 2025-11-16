def collect_all_data():
    """Main data collection loop - FIXED to use run_single_test"""
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
