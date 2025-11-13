
#!/usr/bin/env python3
# Standalone CARLA route test: drive from point A to point B and stop.
import argparse
import math
import time
import random
from contextlib import contextmanager
import sys

sys.path.append('../Downloads/PythonAPI/carla/')
import carla
from agents.navigation.basic_agent import BasicAgent

def distance(a: carla.Location, b: carla.Location) -> float:
    dx, dy, dz = a.x - b.x, a.y - b.y, a.z - b.z
    return math.sqrt(dx*dx + dy*dy + dz*dz)

@contextmanager
def sync_mode(world, fps=20.0):
    settings = world.get_settings()
    original_settings = carla.WorldSettings(
        no_rendering_mode=settings.no_rendering_mode,
        synchronous_mode=settings.synchronous_mode,
        fixed_delta_seconds=settings.fixed_delta_seconds,
        substepping=settings.substepping,
        max_substep_delta_time=settings.max_substep_delta_time,
        max_substeps=settings.max_substeps,
    )
    new_settings = world.get_settings()
    new_settings.synchronous_mode = True
    new_settings.fixed_delta_seconds = 1.0 / fps
    world.apply_settings(new_settings)
    try:
        yield
    finally:
        world.apply_settings(original_settings)

def main():
    parser = argparse.ArgumentParser(description="Drive from spawn A to spawn B using BasicAgent and stop at goal.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town03")
    parser.add_argument("--start", type=int, default=0, help="index of spawn point for start")
    parser.add_argument("--goal", type=int, default=10, help="index of spawn point for goal")
    parser.add_argument("--speed", type=float, default=20.0, help="target speed in km/h for the agent")
    parser.add_argument("--arrival-thresh", type=float, default=5.0, help="meters to goal to consider arrived")
    parser.add_argument("--no-render", action="store_true", help="disable rendering for speed")
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    world = client.load_world(args.town)
    if args.no_render:
        s = world.get_settings()
        s.no_rendering_mode = True
        world.apply_settings(s)

    blueprint_library = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    if len(spawn_points) < 2:
        raise RuntimeError("Not enough spawn points in this town.")
    start_tf = spawn_points[min(args.start, len(spawn_points)-1)]
    goal_tf  = spawn_points[min(args.goal, len(spawn_points)-1)]

    vehicle_bp = blueprint_library.filter("vehicle.*model3*")
    bp = vehicle_bp[0] if vehicle_bp else blueprint_library.filter("vehicle.*")[0]
    if bp.has_attribute("color"):
        color = random.choice(bp.get_attribute("color").recommended_values)
        bp.set_attribute("color", color)

    vehicle = None
    try:
        vehicle = world.spawn_actor(bp, start_tf)
        print(f"Spawned at index {args.start}: {start_tf.location} → goal index {args.goal}: {goal_tf.location}")

        world.debug.draw_string(goal_tf.location, "GOAL", draw_shadow=False,
                                color=carla.Color(0,255,0), life_time=60.0)

        agent = BasicAgent(vehicle, target_speed=args.speed)
        agent.set_destination(goal_tf.location, start_tf.location)

        arrived = False
        with sync_mode(world, fps=20.0):
            for step in range(2000):  # 100 seconds horizon
                world.tick()
                control = agent.run_step()
                vehicle.apply_control(control)

                vloc = vehicle.get_location()
                d = distance(vloc, goal_tf.location)
                if d < args.arrival_thresh:
                    arrived = True
                    print(f"Arrived within {args.arrival_thresh} m of goal (d={d:.2f} m).")
                    break

            # Stop and hold brakes
            for _ in range(40):  # 2 seconds braking
                world.tick()
                vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
            print("Stopped at goal.")
    finally:
        if vehicle is not None:
            vehicle.destroy()
        print("Cleaned up and exiting.")

if __name__ == "__main__":
    main()
