import carla
import time

client = carla.Client('localhost', 2000)
client.set_timeout(30.0)

print("Checking available maps...")
available_maps = client.get_available_maps()
print(f"\nAvailable maps: {available_maps}\n")

# Test loading each map one by one
test_maps = ['Town01', 'Town02', 'Town03', 'Town04', 'Town05', 'Town10HD']

for map_name in test_maps:
    print(f"Testing {map_name}...", end=" ", flush=True)
    try:
        world = client.load_world(map_name)
        time.sleep(2)
        print("✓ SUCCESS")
    except Exception as e:
        print(f"✗ FAILED: {e}")
    time.sleep(1)

print("\nDone!")
