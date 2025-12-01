import carla
import time

# Connect to the CARLA server
client = carla.Client('localhost', 2000)
client.set_timeout(20.0)  # 20 seconds per request

# Wait until the server is ready
print("Waiting for CARLA server...")
while True:
    try:
        # This will fail if the server isn't ready
        world = client.get_world()
        print("Connected to CARLA server!")
        break
    except Exception as e:
        print("Server not ready yet, retrying...")
        time.sleep(1)

# Load Town03 safely
print("Loading Town03...")
world = client.load_world('Town03')
print("Town03 loaded!")

# Spawn a single vehicle
blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.*')[0]
spawn_points = world.get_map().get_spawn_points()
vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
print(f"Spawned vehicle: {vehicle.type_id}")

# Add a single camera sensor
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
print("Spawned camera sensor")

# Let it run a few seconds
time.sleep(5)

# Clean up
camera.destroy()
vehicle.destroy()
print("Actors destroyed, done!")
