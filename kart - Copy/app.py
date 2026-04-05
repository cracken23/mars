"""
app.py  —  KarateTrainer Web Backend
======================================
Replaces the PySide6 Qt launcher (original app.py + ui/pose_canvas.py) with a
Flask + WebSocket server. Uses the EXACT same core/ functions PoseCanvas calls:

    core.pose_estimator  -> PoseEstimator            (MediaPipe wrapper)
    core.pose_matcher    -> similarity_score()        (angle-based scoring)
    core.feedback_engine -> generate_feedback()       (joint feedback list)
    data/levels.json     -> loaded directly (no PoseLoader class)

Key facts from reading the actual source:
  - PoseCanvas.update_frame() calls:
        similarity_score(user_landmarks, ref)          from core.pose_matcher
        convert_landmarks(mp_landmarks)                local helper
  - Scoring thresholds: pass ≥ threshold, close ≥ threshold*0.8
  - levels.json lives at  data/levels.json  (relative to project root)
  - Each level has: id, name, type, threshold, reference_pose, animation
  - animation is a list of 3 frames, each a list of 33 {x,y,z,visibility} dicts
    (used to draw the reference skeleton side-panel in the Qt app)

Run:
    pip install flask flask-sock flask-cors opencv-python mediapipe
    python app.py
Open:
    http://localhost:5000
"""

import os
import json
import threading
import time
import base64
import math

import cv2
import numpy as np
from flask import Flask, render_template, jsonify, request
from flask_sock import Sock
from flask_cors import CORS

# ── Core imports (identical to what PoseCanvas imports) ──────────────────────
from core.pose_estimator import PoseEstimator
from core.pose_matcher import similarity_score      # module-level function
from core.feedback_engine import generate_feedback  # module-level function

# ── Flask setup ───────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
app.config["SECRET_KEY"] = "karate-trainer-secret"
CORS(app)
sock = Sock(app)

# ── Instantiate only what needs instantiation ─────────────────────────────────
pose_estimator = PoseEstimator()

# ── Data path (mirrors PoseCanvas.load_levels()) ─────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
LEVELS_PATH = os.path.join(BASE_DIR, "data", "levels.json")

# ── Shared state ──────────────────────────────────────────────────────────────
camera_lock      = threading.Lock()
cap              = None
camera_active    = False
current_level_idx = 0   # 0-based index into levels list (matches PoseCanvas)

session_attempts  = 0
session_passed    = 0
best_score        = 0.0
latest_score      = 0.0
latest_feedback   = []
latest_passed     = False

# Score smoothing — same as PoseCanvas (deque maxlen=5)
from collections import deque
score_history = deque(maxlen=5)


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_levels():
    """Load levels.json — mirrors PoseCanvas.load_levels()."""
    if not os.path.exists(LEVELS_PATH):
        return []
    with open(LEVELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def convert_landmarks(mp_landmarks):
    """
    Convert MediaPipe landmarks to list of {x, y} dicts.
    Exact copy of PoseCanvas.convert_landmarks().
    """
    if mp_landmarks is None:
        return []
    if hasattr(mp_landmarks, "landmark"):
        mp_landmarks = mp_landmarks.landmark
    result = []
    for lm in mp_landmarks:
        if hasattr(lm, "x"):
            result.append({"x": float(lm.x), "y": float(lm.y)})
        elif isinstance(lm, dict):
            result.append({"x": float(lm["x"]), "y": float(lm["y"])})
    return result


def get_current_level():
    """Return the level dict at current_level_idx."""
    levels = load_levels()
    if not levels:
        return None
    idx = max(0, min(current_level_idx, len(levels) - 1))
    return levels[idx]


def get_feedback_for_score(avg_score, threshold):
    """
    Replicate PoseCanvas's feedback text logic exactly:
        >= threshold       → "Good ✓ matched!"
        >= threshold * 0.8 → "Close — adjust posture"
        else               → "Try again — align to reference"
    Also calls generate_feedback() for per-joint detail when score is low.
    """
    if avg_score >= threshold:
        return ["Good ✓ matched!"], True
    elif avg_score >= threshold * 0.8:
        return ["Close — adjust posture"], False
    else:
        return ["Try again — align to reference"], False


def run_pipeline(frame, draw=True):
    """
    Mirrors PoseCanvas.update_frame() logic exactly:
      1. pose_estimator.process_frame()
      2. convert_landmarks()
      3. similarity_score() from core.pose_matcher
      4. deque-averaged score
      5. threshold check + feedback text
    Returns (annotated_frame, avg_score, passed, feedback_lines)
    """
    global score_history

    annotated, mp_landmarks = pose_estimator.process_frame(frame, draw=draw)
    user_landmarks = convert_landmarks(mp_landmarks)

    level = get_current_level()
    if not level or not user_landmarks:
        score_history.clear()
        return annotated, 0.0, False, ["No pose detected — step into frame"]

    ref = level.get("reference_pose", [])
    threshold = level.get("threshold", 60)

    if not ref:
        return annotated, 0.0, False, ["No reference pose — run capture_reference_poses.py first"]

    # Score using the exact same function PoseCanvas uses
    score = similarity_score(user_landmarks, ref)
    score_history.append(score)
    avg_score = sum(score_history) / len(score_history)

    feedback_lines, passed = get_feedback_for_score(avg_score, threshold)
    return annotated, round(avg_score, 1), passed, feedback_lines


# ── REST routes ───────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/levels")
def api_levels():
    """All levels + animation data for the frontend skeleton panel."""
    try:
        levels = load_levels()
    except Exception as e:
        print(f"[/api/levels] {e}")
        return jsonify([])
    return jsonify([
        {
            "id":            lv.get("id"),
            "name":          lv.get("name", f"Level {lv.get('id')}"),
            "type":          lv.get("type", ""),
            "threshold":     lv.get("threshold", 60),
            "has_reference": len(lv.get("reference_pose", [])) > 0,
            "animation":     lv.get("animation", []),   # ← sent to frontend for skeleton panel
        }
        for lv in levels
    ])


@app.route("/api/session")
def api_session():
    level = get_current_level()
    return jsonify({
        "attempts":          session_attempts,
        "passed":            session_passed,
        "best_score":        best_score,
        "current_level_idx": current_level_idx,
        "current_level":     level.get("id") if level else None,
        "level_name":        level.get("name") if level else "",
        "latest_score":      latest_score,
        "latest_passed":     latest_passed,
        "feedback":          latest_feedback,
        "camera_active":     camera_active,
    })


@app.route("/api/level", methods=["POST"])
def api_set_level():
    """Switch level — mirrors PoseCanvas.on_level_change()."""
    global current_level_idx, score_history
    data   = request.get_json(silent=True) or {}
    levels = load_levels()
    max_idx = len(levels) - 1 if levels else 0
    # Accept either 0-based index or 1-based id
    if "index" in data:
        current_level_idx = max(0, min(int(data["index"]), max_idx))
    elif "level" in data:
        # find by id
        target_id = int(data["level"])
        for i, lv in enumerate(levels):
            if lv.get("id") == target_id:
                current_level_idx = i
                break
    score_history.clear()   # mirrors PoseCanvas.score_history.clear() on level change
    level = get_current_level()
    return jsonify({
        "current_level_idx": current_level_idx,
        "level_name": level.get("name") if level else "",
    })


@app.route("/api/camera/start", methods=["POST"])
def api_camera_start():
    global cap, camera_active
    with camera_lock:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return jsonify({"error": "Could not open camera. Check permissions."}), 500
        camera_active = True
    score_history.clear()
    return jsonify({"status": "started"})


@app.route("/api/camera/stop", methods=["POST"])
def api_camera_stop():
    global cap, camera_active
    with camera_lock:
        camera_active = False
        if cap and cap.isOpened():
            cap.release()
            cap = None
    score_history.clear()
    return jsonify({"status": "stopped"})


@app.route("/api/capture", methods=["POST"])
def api_capture():
    """Snapshot + score — updates session stats."""
    global session_attempts, session_passed, best_score
    global latest_score, latest_feedback, latest_passed

    if not camera_active or cap is None:
        return jsonify({"error": "Camera not active"}), 400

    with camera_lock:
        ret, frame = cap.read()
    if not ret:
        return jsonify({"error": "Frame read failed"}), 500

    _, avg_score, passed, feedback_lines = run_pipeline(frame, draw=False)

    session_attempts += 1
    if passed:
        session_passed += 1
    if avg_score > best_score:
        best_score = avg_score
    latest_score    = avg_score
    latest_passed   = passed
    latest_feedback = feedback_lines

    level = get_current_level()
    return jsonify({
        "score":          avg_score,
        "threshold":      level.get("threshold", 60) if level else 60,
        "passed":         passed,
        "feedback":       feedback_lines,
        "attempts":       session_attempts,
        "session_passed": session_passed,
        "best_score":     best_score,
    })


# ── WebSocket ─────────────────────────────────────────────────────────────────

@sock.route("/ws/stream")
def ws_stream(ws):
    """
    Live frame loop mirroring PoseCanvas.update_frame() at ~20fps.
    Sends JSON:
      { type:"frame",  image:<b64 JPEG>, score, threshold, passed,
                       feedback, level_idx, level_name }
      { type:"status", camera_active:false }
    """
    global latest_score, latest_passed, latest_feedback

    while True:
        try:
            if not camera_active or cap is None or not cap.isOpened():
                ws.send(json.dumps({"type": "status", "camera_active": False}))
                time.sleep(0.4)
                continue

            with camera_lock:
                ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            # No zoom manipulation needed in browser — CSS scales the feed naturally.
            # The Qt zoom-out was only needed because PoseCanvas window was small.
            # Flip horizontally so it feels like a mirror (more natural for training).
            frame = cv2.flip(frame, 1)

            annotated, avg_score, passed, feedback_lines = run_pipeline(frame, draw=True)

            latest_score    = avg_score
            latest_passed   = passed
            latest_feedback = feedback_lines

            level     = get_current_level()
            threshold = level.get("threshold", 60) if level else 60
            lv_name   = level.get("name", "") if level else ""

            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 72])
            b64    = base64.b64encode(buf).decode("utf-8")

            ws.send(json.dumps({
                "type":          "frame",
                "image":         b64,
                "score":         avg_score,
                "threshold":     threshold,
                "passed":        passed,
                "feedback":      feedback_lines,
                "level_idx":     current_level_idx,
                "level_name":    lv_name,
            }))

            time.sleep(0.05)   # ~20 fps

        except Exception as e:
            print(f"[WS] disconnected: {e}")
            break


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("  KarateTrainer  →  http://localhost:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)