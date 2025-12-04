from flask import Flask, render_template, request, jsonify
import time

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

from optimize_setup import optimize_setup_for_flags

@app.route("/optimize", methods=["POST"])
def optimize():
    data = request.get_json() or {}
    flags_dict = data.get("environment_flags", {})
    
    # Map frontend keys to backend keys
    # Frontend keys: right_angle_turns, steep_elevation_changes, high_speed_sections, 
    #                tight_curves, long_straights, narrow_lanes, wide_lanes
    # Backend keys: intersections_90, steep_elevation, high_speed_sections, 
    #               tight_curves, long_straightaways, narrow_lanes, wide_multi_lane
    
    selected_flags = []
    if flags_dict.get("right_angle_turns"):
        selected_flags.append("intersections_90")
    if flags_dict.get("steep_elevation_changes"):
        selected_flags.append("steep_elevation")
    if flags_dict.get("high_speed_sections"):
        selected_flags.append("high_speed_sections")
    if flags_dict.get("tight_curves"):
        selected_flags.append("tight_curves")
    if flags_dict.get("long_straights"):
        selected_flags.append("long_straightaways")
    if flags_dict.get("narrow_lanes"):
        selected_flags.append("narrow_lanes")
    if flags_dict.get("wide_lanes"):
        selected_flags.append("wide_multi_lane")
        
    # Call the optimization routine
    try:
        result = optimize_setup_for_flags(selected_flags)
        return jsonify(result)
    except Exception as e:
        print(f"Error during optimization: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)