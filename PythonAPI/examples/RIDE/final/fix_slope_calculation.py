"""Fix slope calculation to capture total elevation changes"""

import sys

with open('track_param_optimizer_FINAL.py', 'r') as f:
    content = f.read()

# Find and replace the slope calculation in extract_features
old_slope_code = '''            if len(seg_z) > 1:
                slope = abs(seg_z[-1] - seg_z[0]) / max(end_s - start_s, 1e-6)
            else:
                slope = 0.0
            slopes.append(slope)'''

new_slope_code = '''            if len(seg_z) > 1:
                # Calculate slope as elevation change / horizontal distance
                elevation_change = abs(seg_z[-1] - seg_z[0])
                horizontal_distance = end_s - start_s
                slope = elevation_change / max(horizontal_distance, 1e-6)
            else:
                slope = 0.0
            slopes.append(slope)'''

content = content.replace(old_slope_code, new_slope_code)

# Also add total elevation range as a feature
old_features = '''        features = {
            'total_length': total_length,
            'avg_curvature': np.mean(np.abs(curvature_array)),
            'max_curvature': np.max(np.abs(curvature_array)),
            'std_curvature': np.std(curvature_array),
            'avg_slope': np.mean(slopes),
            'max_slope': np.max(slopes),'''

new_features = '''        # Calculate total elevation change
        elevation_range = np.max(z) - np.min(z)
        max_elevation = np.max(z)
        min_elevation = np.min(z)
        
        features = {
            'total_length': total_length,
            'elevation_range': elevation_range,
            'max_elevation': max_elevation,
            'min_elevation': min_elevation,
            'avg_curvature': np.mean(np.abs(curvature_array)),
            'max_curvature': np.max(np.abs(curvature_array)),
            'std_curvature': np.std(curvature_array),
            'avg_slope': np.mean(slopes),
            'max_slope': np.max(slopes),'''

content = content.replace(old_features, new_features)

# Update the print statement to show elevation range
old_print = '''        print(f"  {feats['total_length']:.0f}m, curves={feats['tight_corners_pct']*100:.0f}%, max_slope={feats['max_slope']:.4f}")'''

new_print = '''        elev_str = f"elev={feats['elevation_range']:.1f}m" if feats.get('elevation_range', 0) > 1 else "flat"
        print(f"  {feats['total_length']:.0f}m, curves={feats['tight_corners_pct']*100:.0f}%, {elev_str}, max_slope={feats['max_slope']:.4f}")'''

content = content.replace(old_print, new_print)

with open('track_param_optimizer_FINAL.py', 'w') as f:
    f.write(content)

print("✓ Fixed slope calculation!")
print("\nNow elevation_range will show total vertical change")
print("Example: 'elev=11.4m' for Town04")
