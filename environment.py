import carla
import random

# Manipulating CARLA through the Python 
# Have not tested

client = carla.Client('localhost', 2000)
world = client.get_world()

client.load_world('Town07')
vehicle_blueprints = world.get_blueprint_library().filter('vehicle.audi.tt')

spawn_points = world.get_map().get_spawn_points()

ego_vehicle = world.spawn_actor(random.choice(vehicle_blueprints), random.choice(spawn_points))

# Create a transform to place the camera on top of the vehicle
camera_init_trans = carla.Transform(carla.Location(z=1.5))

# We create the camera through a blueprint that defines its properties
camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')

# We spawn the camera and attach it to our ego vehicle
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)

for vehicle in world.get_actors().filter('*vehicle*'):
    vehicle.set_autopilot(True)