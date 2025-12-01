import carla
import time

client = carla.Client('localhost', 2000)
client.set_timeout(20.0)

maps = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town10HD']

print("Checking elevation for your available maps...")
print("="*60)

for map_name in maps:
    try:
        world = client.load_world(map_name)
        time.sleep(1)
        
        carla_map = world.get_map()
        spawn_points = carla_map.get_spawn_points()
        
        elevations = [sp.location.z for sp in spawn_points]
        min_z = min(elevations)
        max_z = max(elevations)
        elevation_range = max_z - min_z
        
        print(f"{map_name:10} | Elevation range: {elevation_range:6.1f}m")
    except Exception as e:
        print(f"{map_name:10} | Error: {e}")

print("="*60)
