from flask import Flask, render_template, request, jsonify
import time

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/optimize", methods=["POST"])
def optimize():
    data = request.get_json() or {}
    flags = data.get("environment_flags", {})
    # TODO: call your CARLA/ML parameter search here
    time.sleep(2)
    best_params = {
        "profile_summary": ", ".join(
            k.replace("_", " ") for k, v in flags.items() if v
        ) or "No specific features selected",
        "tire_friction": 1.2,
        "gear_ratio": 3.5,
        "predicted_time": 123.4,
        "max lateral force": "20G"
    }
    return jsonify(best_params)

if __name__ == "__main__":
    app.run(debug=True)