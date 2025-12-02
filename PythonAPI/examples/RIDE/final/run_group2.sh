#!/bin/bash
# GROUP 2: TIGHT_CURVES (scenarios 5,6,7,8,9)

cd /projects/p33058/carla/PythonAPI/examples/RIDE/final

echo "============================================================"
echo "GROUP 2: TIGHT_CURVES"
echo "Scenarios: 5,6,7,8,9"
echo "============================================================"

# Create unique data directory for this group
mkdir -p track_optimization_data_physics_curves

# Run with modified DATA_DIR
python3 << 'PYEOF'
import sys
sys.argv = ['', '--scenarios', '5,6,7,8,9']

# Override DATA_DIR before importing
import track_param_optimizer_PHYSICS as script
script.DATA_DIR = "track_optimization_data_physics_curves"
import os
os.makedirs(script.DATA_DIR, exist_ok=True)

script.main()
PYEOF

echo ""
echo "✓ GROUP 2 COMPLETE!"
