#!/bin/bash
# TIRE FRICTION TESTING - 10 sharp Town02 routes

cd /projects/p33058/carla/PythonAPI/examples/RIDE/final

echo "============================================================"
echo "TIRE FRICTION TESTING"
echo "Scenarios: 50-59 (10 sharpest Town02 routes)"
echo "Friction values: [0.5, 1.0, 2.0, 3.0] (same as existing data)"
echo "============================================================"

# Create separate data directory
mkdir -p track_optimization_data_friction

# Run with modified DATA_DIR
python3 << 'PYEOF'
import sys
sys.argv = ['', '--scenarios', '50,51,52,53,54,55,56,57,58,59']

# Override DATA_DIR before importing
import track_param_optimizer_PHYSICS as script
script.DATA_DIR = "track_optimization_data_friction"
import os
os.makedirs(script.DATA_DIR, exist_ok=True)

script.main()
PYEOF

echo ""
echo "✓ FRICTION TESTS COMPLETE!"
echo ""
echo "Collected: 10 scenarios × 16 configs = 160 points"
echo "Data: track_optimization_data_friction/"
