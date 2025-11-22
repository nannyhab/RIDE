from flask import Flask, render_template, request, jsonify
import time

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/optimize", methods=["POST"])
def optimize():
    data = request.get_json()
    slider_value = data["env_mix"]   # 0.0 = city, 1.0 = highway
    print(slider_value)
    # TODO: call your CARLA/ML parameter search here
    time.sleep(2)
    best_params = {
        "slider value": slider_value,
        "tire_friction": 1.2,
        "gear_ratio": 3.5,
        "predicted_time": 123.4,
        "max lateral force": "20G"
    }
    return jsonify(best_params)

if __name__ == "__main__":
    app.run(debug=True)