#!/bin/bash
# GROUP 3: ELEVATION (scenarios 10,11,12,13,14)

cd /projects/p33058/carla/PythonAPI/examples/RIDE/final

echo "============================================================"
echo "GROUP 3: ELEVATION"
echo "Scenarios: 10,11,12,13,14"
echo "============================================================"

# Create unique data directory for this group
mkdir -p track_optimization_data_physics_elevation

# Run with modified DATA_DIR
python3 << 'PYEOF'
import sys
sys.argv = ['', '--scenarios', '10,11,12,13,14']

# Override DATA_DIR before importing
import track_param_optimizer_PHYSICS as script
script.DATA_DIR = "track_optimization_data_physics_elevation"
import os
os.makedirs(script.DATA_DIR, exist_ok=True)

script.main()
PYEOF

echo ""
echo "✓ GROUP 3 COMPLETE!"
