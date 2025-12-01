"""Diagnose CARLA connection issues"""
import carla
import time
import sys

print("="*60)
print("CARLA DIAGNOSTIC")
print("="*60)

print("\n1. Attempting to connect to CARLA...")
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    print("   ✓ Connected to CARLA server")
except Exception as e:
    print(f"   ✗ FAILED to connect: {e}")
    sys.exit(1)

print("\n2. Getting CARLA version...")
try:
    version = client.get_server_version()
    print(f"   ✓ CARLA version: {version}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

print("\n3. Listing available maps...")
try:
    maps = client.get_available_maps()
    print(f"   ✓ Found {len(maps)} maps:")
    for m in maps:
        print(f"      - {m}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

print("\n4. Loading Town01...")
try:
    world = client.load_world('Town01')
    time.sleep(2)
    print("   ✓ Town01 loaded successfully")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

print("\n5. Getting world settings...")
try:
    settings = world.get_settings()
    print(f"   ✓ Current settings:")
    print(f"      Synchronous mode: {settings.synchronous_mode}")
    print(f"      Fixed delta: {settings.fixed_delta_seconds}")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

print("\n6. Initializing Traffic Manager...")
try:
    traffic_manager = client.get_trafficmanager(8000)
    print("   ✓ Traffic Manager initialized")
except Exception as e:
    print(f"   ✗ FAILED HERE: {e}")
    print("\n   This is where your script crashes!")
    sys.exit(1)

print("\n7. Setting Traffic Manager to synchronous mode...")
try:
    traffic_manager.set_synchronous_mode(True)
    print("   ✓ Synchronous mode enabled")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)
