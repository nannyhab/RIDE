#!/bin/bash
# GROUP 5: MIXED (scenarios 20,21,22,23,24)

cd /projects/p33058/carla/PythonAPI/examples/RIDE/final

echo "============================================================"
echo "GROUP 5: MIXED"
echo "Scenarios: 20,21,22,23,24"
echo "============================================================"

# Create unique data directory for this group
mkdir -p track_optimization_data_physics_mixed

# Run with modified DATA_DIR
python3 << 'PYEOF'
import sys
sys.argv = ['', '--scenarios', '20,21,22,23,24']

# Override DATA_DIR before importing
import track_param_optimizer_PHYSICS as script
script.DATA_DIR = "track_optimization_data_physics_mixed"
import os
os.makedirs(script.DATA_DIR, exist_ok=True)

script.main()
PYEOF

echo ""
echo "✓ GROUP 5 COMPLETE!"
