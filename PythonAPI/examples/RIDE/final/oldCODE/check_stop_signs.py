"""
Check which CARLA maps have stop signs - SAFE VERSION
Checks one map at a time with proper cleanup
"""

import carla
import sys
import time

def check_map_for_stop_signs(map_name):
    """Check if a map has stop signs - creates new client each time"""
    client = None
    try:
        print(f"\nChecking {map_name}...", flush=True)
        
        # Fresh client connection
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)  # Longer timeout
        
        # Load the map
        world = client.load_world(map_name)
        
        # Wait for map to fully load
        time.sleep(2)
        
        # Get the map
        carla_map = world.get_map()
        
        # Get all landmarks (traffic signs/lights)
        all_landmarks = carla_map.get_all_landmarks()
        
        print(f"  Total landmarks: {len(all_landmarks)}")
        
        # Check different types
        stops = []
        yields = []
        traffic_lights = []
        
        for lm in all_landmarks:
            name_lower = lm.name.lower()
            # Stop signs: OpenDRIVE type 206
            if 'stop' in name_lower or lm.type == '206':
                stops.append(lm)
            # Yield signs: OpenDRIVE type 205  
            elif 'yield' in name_lower or lm.type == '205':
                yields.append(lm)
            # Traffic lights: OpenDRIVE type 1000001
            elif 'traffic' in name_lower or lm.type == '1000001':
                traffic_lights.append(lm)
        
        print(f"  Stop signs: {len(stops)}")
        print(f"  Yield signs: {len(yields)}")
        print(f"  Traffic lights: {len(traffic_lights)}")
        
        if len(stops) > 0:
            print(f"  ✓✓✓ {map_name} HAS STOP SIGNS! ✓✓✓")
            for i, sign in enumerate(stops[:5]):
                loc = sign.transform.location
                print(f"    #{i+1}: ({loc.x:.1f}, {loc.y:.1f}, {loc.z:.1f})")
        
        return {
            'stops': len(stops),
            'yields': len(yields),
            'lights': len(traffic_lights),
            'status': 'success'
        }
        
    except KeyboardInterrupt:
        print("\n  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"  ERROR: {e}")
        return {
            'stops': 0,
            'yields': 0,
            'lights': 0,
            'status': f'error: {str(e)[:50]}'
        }
    finally:
        # Cleanup
        if client:
            del client
        time.sleep(1)


def main():
    # Simple list of maps to check (avoiding crashes)
    maps_to_check = [
        'Town01',
        'Town02', 
        'Town03',
        'Town04',
        'Town05',
        'Town10HD'
    ]
    
    print("="*60)
    print("CHECKING MAPS FOR STOP SIGNS")
    print("="*60)
    print("Checking core maps (not _Opt versions to avoid crashes)")
    print("This may take a few minutes...\n")
    
    results = {}
    
    for map_name in maps_to_check:
        try:
            result = check_map_for_stop_signs(map_name)
            results[map_name] = result
            time.sleep(3)  # Pause between maps
        except KeyboardInterrupt:
            print("\nStopped by user")
            break
        except Exception as e:
            print(f"  CRASH: {e}")
            results[map_name] = {
                'stops': 0,
                'yields': 0, 
                'lights': 0,
                'status': 'crashed'
            }
            time.sleep(5)  # Longer pause after crash
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for map_name in sorted(results.keys()):
        data = results[map_name]
        status = data['status']
        
        if status == 'success':
            stops = data['stops']
            yields = data['yields']
            lights = data['lights']
            
            marker = "✓✓✓" if stops > 0 else "   "
            print(f"{marker} {map_name:12} | Stops: {stops:3} | Yields: {yields:3} | Lights: {lights:3}")
        else:
            print(f"    {map_name:12} | {status}")
    
    # Recommendations
    maps_with_stops = [(name, data['stops']) for name, data in results.items() 
                      if data.get('stops', 0) > 0]
    
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    
    if maps_with_stops:
        best_map = max(maps_with_stops, key=lambda x: x[1])
        print(f"✓ BEST for stop signs: {best_map[0]} ({best_map[1]} stop signs)")
    else:
        print("✗ NO STOP SIGNS found in default CARLA maps")
        print("\nOptions:")
        print("  1. Use traffic lights for stop-and-go (all maps have them)")
        print("  2. Create custom map with RoadRunner")
        print("  3. Use yield signs if available (similar behavior)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
