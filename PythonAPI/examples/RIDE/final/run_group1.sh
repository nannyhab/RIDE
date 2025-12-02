#!/bin/bash
# GROUP 1: STRAIGHT_SPEED (scenarios 0,1,2,3,4)

cd /projects/p33058/carla/PythonAPI/examples/RIDE/final

echo "============================================================"
echo "GROUP 1: STRAIGHT_SPEED"
echo "Scenarios: 0,1,2,3,4"
echo "============================================================"

# Create unique data directory for this group
mkdir -p track_optimization_data_physics_straight

# Run with modified DATA_DIR
python3 << 'PYEOF'
import sys
sys.argv = ['', '--scenarios', '0,1,2,3,4']

# Override DATA_DIR before importing
import track_param_optimizer_PHYSICS as script
script.DATA_DIR = "track_optimization_data_physics_straight"
import os
os.makedirs(script.DATA_DIR, exist_ok=True)

script.main()
PYEOF

echo ""
echo "✓ GROUP 1 COMPLETE!"
