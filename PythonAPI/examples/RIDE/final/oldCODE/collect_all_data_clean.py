def collect_all_data():
    """Main data collection loop WITH CHECKPOINTING"""
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
        track_xodr = generator.generate_random_track(track_id)
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
        
        dataset.extend(track_results)
        
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"  ✓ Checkpoint saved ({len(dataset)} total points, {len(track_results)} from this track)")
    
    with open(f'{DATA_DIR}/dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    
    with open(f'{DATA_DIR}/dataset.json', 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\n✓ Collected {len(dataset)} data points")
    print(f"✓ Saved to {DATA_DIR}/")
    
    return dataset
