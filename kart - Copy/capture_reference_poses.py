"""
capture_reference_poses.py
---------------------------------
Capture 6 reference karate poses from webcam and save them as
karate_trainer/data/levels.json, ready for multi-level training.

Press SPACE to capture a pose.
Press Q to quit early.

This version converts MediaPipe NormalizedLandmarkList into serializable dicts.
"""

import os
import cv2
import json
from datetime import datetime
from core.pose_estimator import PoseEstimator

# Define 6 levels (can modify names/types easily)
LEVELS_TEMPLATE = [
    {"id": 1, "name": "Front Kick (Right)", "type": "kick", "threshold": 65},
    {"id": 2, "name": "Side Kick (Left)", "type": "kick", "threshold": 65},
    {"id": 3, "name": "Straight Punch (Right)", "type": "punch", "threshold": 60},
    {"id": 4, "name": "Hook Punch (Left)", "type": "punch", "threshold": 60},
    {"id": 5, "name": "Defense Stance A", "type": "defense", "threshold": 70},
    {"id": 6, "name": "Defense Stance B", "type": "defense", "threshold": 70},
]

# Output file path
OUT_PATH = os.path.join("karate_trainer", "data", "levels.json")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)


def serialize_landmarks(landmarks):
    """
    Convert MediaPipe landmarks (NormalizedLandmarkList or similar) into a list
    of simple dicts with float values: [{'x':..,'y':..,'z':..,'visibility':..}, ...].
    Accepts:
      - a MediaPipe NormalizedLandmarkList (has .landmark)
      - a list/iterable of objects with x,y (and optionally z, visibility)
      - a list of dicts already (will be returned as-is)
    """
    if landmarks is None:
        return []

    # If it's already a plain list of dicts (serializable), return a shallow copy
    if isinstance(landmarks, list) and landmarks and isinstance(landmarks[0], dict):
        # But ensure keys are limited to numeric types
        out = []
        for lm in landmarks:
            d = {}
            # keep x,y,z,visibility if present
            for k in ("x", "y", "z", "visibility"):
                if k in lm:
                    # ensure floats
                    try:
                        d[k] = float(lm[k])
                    except Exception:
                        d[k] = lm[k]
            out.append(d)
        return out

    # If it has a .landmark attribute (MediaPipe NormalizedLandmarkList)
    if hasattr(landmarks, "landmark"):
        iterable = landmarks.landmark
    else:
        # assume it's a general iterable of objects/lists/tuples
        iterable = landmarks

    serial = []
    for lm in iterable:
        # If the item is already a dict-like with x,y
        if isinstance(lm, dict):
            item = {}
            for k in ("x", "y", "z", "visibility"):
                if k in lm:
                    try:
                        item[k] = float(lm[k])
                    except Exception:
                        item[k] = lm[k]
            serial.append(item)
            continue

        # If lm has attributes x,y (MediaPipe landmark object)
        if hasattr(lm, "x") and hasattr(lm, "y"):
            item = {}
            try:
                item["x"] = float(lm.x)
            except Exception:
                item["x"] = lm.x
            try:
                item["y"] = float(lm.y)
            except Exception:
                item["y"] = lm.y
            # optional fields
            if hasattr(lm, "z"):
                try:
                    item["z"] = float(lm.z)
                except Exception:
                    item["z"] = lm.z
            if hasattr(lm, "visibility"):
                try:
                    item["visibility"] = float(lm.visibility)
                except Exception:
                    item["visibility"] = lm.visibility
            serial.append(item)
            continue

        # If lm is a tuple/list like (x,y) or (x,y,z)
        if isinstance(lm, (list, tuple)):
            item = {}
            if len(lm) >= 1:
                item["x"] = float(lm[0])
            if len(lm) >= 2:
                item["y"] = float(lm[1])
            if len(lm) >= 3:
                item["z"] = float(lm[2])
            serial.append(item)
            continue

        # Fallback: try to convert to str (not ideal) — skip
    return serial


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    pose_est = PoseEstimator()
    print("✅ Webcam opened.")
    print("➡️  Press SPACE to capture pose, Q to quit early.")

    current_idx = 0
    total = len(LEVELS_TEMPLATE)
    captured = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated, landmarks = pose_est.process_frame(frame, draw=True)

        label = f"Level {current_idx + 1}/{total}: {LEVELS_TEMPLATE[current_idx]['name']}"
        cv2.putText(
            annotated, label, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2
        )
        cv2.putText(
            annotated, "Press SPACE to capture | R to retake | Q to quit",
            (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )
        cv2.imshow("Karate Pose Capture", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):  # capture
            if landmarks:
                serial = serialize_landmarks(landmarks)
                LEVELS_TEMPLATE[current_idx]["reference_pose"] = serial
                captured += 1
                print(f"✅ Captured {LEVELS_TEMPLATE[current_idx]['name']}")
                current_idx += 1
                if current_idx >= total:
                    break
            else:
                print("⚠️ No pose detected — try again (adjust camera/lighting).")

        elif key == ord("r"):  # retake current (optional convenience)
            print("🔁 Retake current pose — press SPACE when ready.")

        elif key == ord("q"):
            print("❎ Quit early.")
            break

    cap.release()
    cv2.destroyAllWindows()
    pose_est.release()

    # Fill any uncaptured poses with empty lists
    for i in range(len(LEVELS_TEMPLATE)):
        if "reference_pose" not in LEVELS_TEMPLATE[i]:
            LEVELS_TEMPLATE[i]["reference_pose"] = []

    # Save
    try:
        with open(OUT_PATH, "w") as f:
            json.dump(LEVELS_TEMPLATE, f, indent=2)
        print(f"\n✅ Saved {captured}/{total} poses to {OUT_PATH}")
        print("You can now run your main app to try all levels!")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        print("❌ Failed to save levels.json:", e)


if __name__ == "__main__":
    main()
