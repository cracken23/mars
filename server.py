"""
Flask API server for the Karate Trainer web frontend.

Exposes the existing pose matching and feedback logic as REST endpoints.
Run with: python server.py
"""
import importlib.util
import json
import sys
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

# Load pose_matcher and feedback_engine directly from their files so we
# don't trigger core/__init__.py (which imports cv2/mediapipe).
_CORE_DIR = Path(__file__).parent / "core"


def _load_module(name: str, filepath: Path):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_pm = _load_module("core.pose_matcher", _CORE_DIR / "pose_matcher.py")
_fe = _load_module("core.feedback_engine", _CORE_DIR / "feedback_engine.py")
similarity_score = _pm.similarity_score
compute_angle_differences = _pm.compute_angle_differences
generate_detailed_feedback = _fe.generate_detailed_feedback

app = Flask(__name__, static_folder="web", static_url_path="")

# Load levels data at startup
LEVELS_FILE = Path(__file__).parent / "data" / "levels.json"
LEVELS: list[dict] = []

try:
    with open(LEVELS_FILE, "r") as f:
        LEVELS = json.load(f)
except Exception as e:
    print(f"Warning: could not load {LEVELS_FILE}: {e}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory("web", "index.html")


@app.route("/api/levels")
def get_levels():
    """Return all levels without the heavy reference_pose data."""
    summary = [
        {"id": lv["id"], "name": lv["name"], "type": lv["type"], "threshold": lv["threshold"]}
        for lv in LEVELS
    ]
    return jsonify(summary)


@app.route("/api/levels/<int:level_id>")
def get_level(level_id: int):
    """Return a single level including its reference_pose."""
    for lv in LEVELS:
        if lv["id"] == level_id:
            return jsonify(lv)
    return jsonify({"error": "Level not found"}), 404


@app.route("/api/score", methods=["POST"])
def score():
    """
    Score user landmarks against a level's reference pose.

    Expects JSON: {"level_id": int, "landmarks": [{"x": float, "y": float}, ...]}
    Returns JSON:  {"score": float, "status": str, "feedback": [str, ...]}
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid JSON body"}), 400

    level_id = data.get("level_id")
    user_landmarks = data.get("landmarks")

    if level_id is None or not user_landmarks:
        return jsonify({"error": "level_id and landmarks are required"}), 400

    # Find the level
    level = None
    for lv in LEVELS:
        if lv["id"] == level_id:
            level = lv
            break

    if level is None:
        return jsonify({"error": "Level not found"}), 404

    ref_landmarks = level["reference_pose"]

    # Compute score
    sc = similarity_score(user_landmarks, ref_landmarks)
    threshold = level.get("threshold", 60)

    # Status text
    if sc >= threshold:
        status = "Good \u2014 matched!"
    elif sc >= threshold * 0.8:
        status = "Close \u2014 adjust posture"
    else:
        status = "Try again \u2014 align to reference"

    # Detailed feedback
    diffs = compute_angle_differences(user_landmarks, ref_landmarks)
    feedback = generate_detailed_feedback(diffs, threshold_degrees=15.0)

    return jsonify({
        "score": round(sc, 1),
        "threshold": threshold,
        "status": status,
        "feedback": feedback[:3],
    })


# ---------------------------------------------------------------------------
# CORS support (simple approach for local development)
# ---------------------------------------------------------------------------

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


if __name__ == "__main__":
    print(f"Loaded {len(LEVELS)} levels from {LEVELS_FILE}")
    print("Starting Karate Trainer API at http://localhost:5000")
    app.run(debug=True, port=5000)
