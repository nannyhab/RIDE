#!/bin/bash
# FRICTION TESTING GROUP: Sharp curves (scenarios 50-59)

cd /projects/p33058/carla/PythonAPI/examples/RIDE/final

echo "============================================================"
echo "FRICTION GROUP: TIRE FRICTION TESTS"
echo "Scenarios: 50-59"
echo "============================================================"

# Create unique data directory for friction tests
mkdir -p track_optimization_data_physics_friction

# Run with data directory override
python3 << 'PYEOF'
import sys
sys.argv = ['', '--scenarios', '50,51,52,53,54,55,56,57,58,59']

# Override DATA_DIR
import track_param_optimizer_PHYSICS as script
script.DATA_DIR = "track_optimization_data_physics_friction"
import os
os.makedirs(script.DATA_DIR, exist_ok=True)

script.main()
PYEOF

echo ""
echo "✓ FRICTION GROUP COMPLETE!"
