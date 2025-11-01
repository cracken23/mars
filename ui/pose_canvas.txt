# karate_trainer/ui/pose_canvas.py
import json
import cv2
from collections import deque
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap, QFont

from core.pose_estimator import PoseEstimator
from core.pose_matcher import (
    similarity_score,
    landmarks_to_2d_list,
    compute_angle_differences,
    KEY_ANGLE_TRIPLETS,
)

# utility to draw reference skeleton (expects list of landmark dicts with x,y normalized)
POSE_CONNECTIONS = [
    (11, 13), (13, 15), (12, 14), (14, 16),  # arms
    (11, 12), (23, 24), (11, 23), (12, 24),  # shoulders-hips
    (23, 25), (25, 27), (24, 26), (26, 28),  # legs
    (27, 31), (28, 32)
]


def mp_landmarks_to_simple_list(mp_landmarks):
    """
    Convert MediaPipe NormalizedLandmarkList (or list/tuple/dict forms)
    into a list of simple dicts: [{'x': float, 'y': float, 'z':..., 'visibility':...}, ...]
    """
    if mp_landmarks is None:
        return []

    # MediaPipe NormalizedLandmarkList has .landmark
    if hasattr(mp_landmarks, "landmark"):
        iterable = mp_landmarks.landmark
    else:
        iterable = mp_landmarks

    out = []
    for lm in iterable:
        # already a dict-like object
        if isinstance(lm, dict):
            d = {}
            for k in ("x", "y", "z", "visibility"):
                if k in lm:
                    try:
                        d[k] = float(lm[k])
                    except Exception:
                        d[k] = lm[k]
            out.append(d)
            continue

        # object with attributes (MediaPipe landmark)
        if hasattr(lm, "x") and hasattr(lm, "y"):
            d = {}
            try:
                d["x"] = float(lm.x)
                d["y"] = float(lm.y)
            except Exception:
                d["x"], d["y"] = lm.x, lm.y
            if hasattr(lm, "z"):
                try:
                    d["z"] = float(lm.z)
                except Exception:
                    d["z"] = lm.z
            if hasattr(lm, "visibility"):
                try:
                    d["visibility"] = float(lm.visibility)
                except Exception:
                    d["visibility"] = lm.visibility
            out.append(d)
            continue

        # tuple/list like (x,y) or (x,y,z)
        if isinstance(lm, (list, tuple)):
            d = {}
            if len(lm) >= 1:
                d["x"] = float(lm[0])
            if len(lm) >= 2:
                d["y"] = float(lm[1])
            if len(lm) >= 3:
                d["z"] = float(lm[2])
            out.append(d)
            continue

        # unknown type -> skip
    return out


class PoseCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # UI elements
        self.cam_label = QLabel()
        self.cam_label.setAlignment(Qt.AlignCenter)

        self.score_label = QLabel("Score: -")
        self.score_label.setFont(QFont("Arial", 18))
        self.feedback_label = QLabel("")
        self.feedback_label.setFont(QFont("Arial", 14))

        # Level selector + next/prev
        controls = QHBoxLayout()
        self.level_combo = QComboBox()
        self.prev_btn = QPushButton("Prev")
        self.next_btn = QPushButton("Next")
        controls.addWidget(self.prev_btn)
        controls.addWidget(self.level_combo)
        controls.addWidget(self.next_btn)

        v = QVBoxLayout()
        v.addLayout(controls)
        v.addWidget(self.cam_label)
        v.addWidget(self.score_label)
        v.addWidget(self.feedback_label)
        self.setLayout(v)

        self.prev_btn.clicked.connect(self.prev_level)
        self.next_btn.clicked.connect(self.next_level)
        self.level_combo.currentIndexChanged.connect(self.on_level_change)

        # Pose estimator
        self.pose_estimator = PoseEstimator()
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Warning: Webcam not opened")

        # timer loop for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

        # load levels
        self.levels = []
        self.current_level_index = 0
        self.current_level = None
        self.load_levels()

        # smoothing history for score
        self.score_history = deque(maxlen=5)

    def load_levels(self):
        import os
        base = os.path.dirname(os.path.dirname(__file__))  # karate_trainer/ui -> karate_trainer
        levels_path = os.path.join(base, "data", "levels.json")
        try:
            with open(levels_path, 'r') as f:
                self.levels = json.load(f)
        except Exception as e:
            print("Could not load levels.json:", e)
            # fallback to single existing reference if present
            try:
                ref = json.load(open(os.path.join(base, "reference_pose.json")))
                self.levels = [{"id": 1, "name": "Reference", "type": "ref", "threshold": 60, "reference_pose": ref}]
            except Exception:
                self.levels = []

        # populate level combo
        self.level_combo.clear()
        for lv in self.levels:
            self.level_combo.addItem(lv.get('name', f"Level {lv.get('id', '?')}"))
        if self.levels:
            self.current_level_index = 0
            self.current_level = self.levels[0]

    def prev_level(self):
        if not self.levels:
            return
        self.current_level_index = max(0, self.current_level_index - 1)
        self.level_combo.setCurrentIndex(self.current_level_index)

    def next_level(self):
        if not self.levels:
            return
        self.current_level_index = min(len(self.levels) - 1, self.current_level_index + 1)
        self.level_combo.setCurrentIndex(self.current_level_index)

    def on_level_change(self, idx):
        if idx < 0 or idx >= len(self.levels):
            return
        self.current_level_index = idx
        self.current_level = self.levels[idx]

    def draw_reference_overlay(self, frame, ref_landmarks, alpha=0.5):
        """
        Draws reference landmarks and connections on the frame (BGR).
        ref_landmarks: list of dicts with normalized x,y (0..1) coordinates (MediaPipe style)
        """
        h, w = frame.shape[:2]
        overlay = frame.copy()
        pts = []
        for i, lm in enumerate(ref_landmarks):
            # safe check for x,y
            if 'x' not in lm or 'y' not in lm:
                pts.append((0, 0))
                continue
            x = int(lm['x'] * w)
            y = int(lm['y'] * h)
            pts.append((x, y))
            cv2.circle(overlay, (x, y), 3, (0, 200, 200), -1)

        for (a, b) in POSE_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(overlay, pts[a], pts[b], (0, 200, 200), 2)

        # blend overlay
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # get annotated frame and raw landmarks from your PoseEstimator
        try:
            annotated_frame, mp_landmarks = self.pose_estimator.process_frame(frame, draw=True)
        except Exception as e:
            # If pose estimator fails, still show raw frame
            annotated_frame = frame.copy()
            mp_landmarks = None
            print("PoseEstimator error:", e)

        score_text = "-"
        feedback = ""

        # Convert MediaPipe or arbitrary landmarks to simple list of dicts
        user_landmarks = mp_landmarks_to_simple_list(mp_landmarks)
        # For matching we only need x,y pairs in the expected dict format
        user_lm_xy = [{'x': lm['x'], 'y': lm['y']} for lm in user_landmarks] if user_landmarks else []

        if self.current_level and user_lm_xy:
            ref = self.current_level.get('reference_pose')
            if ref:
                # compute similarity score safely
                try:
                    s = similarity_score(user_lm_xy, ref)
                except Exception as e:
                    print("Similarity error:", e)
                    s = 0.0

                # smoothing
                try:
                    self.score_history.append(s)
                    smoothed = sum(self.score_history) / len(self.score_history)
                except Exception:
                    smoothed = s

                score_text = f"{smoothed:.1f}%"
                threshold = self.current_level.get('threshold', 60)

                # feedback rules
                if smoothed >= threshold:
                    feedback = "Good — matched!"
                elif smoothed >= threshold * 0.8:
                    feedback = "Close — adjust posture"
                else:
                    feedback = "Try again — align to reference"

                # draw reference overlay (semi-transparent skeleton)
                try:
                    annotated_frame = self.draw_reference_overlay(annotated_frame, ref, alpha=0.5)
                except Exception:
                    pass

                # optional: highlight worst contributing angles
                try:
                    diffs = compute_angle_differences(user_lm_xy, ref)
                    # find top 2 worst diffs
                    diffs_with_idx = [(i, r, u, d) for (i, r, u, d) in diffs if d is not None]
                    diffs_with_idx.sort(key=lambda x: -x[3])  # descending by diff
                    worst = diffs_with_idx[:2]
                    # draw small red circles at the angle vertex points for feedback
                    h, w = annotated_frame.shape[:2]
                    for (triplet_idx, ra, ua, diff) in worst:
                        a, b, c = KEY_ANGLE_TRIPLETS[triplet_idx]
                        # b is the vertex index — draw marker if present in ref or user
                        if b < len(ref) and 'x' in ref[b] and 'y' in ref[b]:
                            rx = int(ref[b]['x'] * w)
                            ry = int(ref[b]['y'] * h)
                            cv2.circle(annotated_frame, (rx, ry), 8, (0, 0, 255), -1)
                except Exception:
                    # angle highlighting is optional — ignore errors
                    pass

        # Update Qt widget with annotated_frame
        try:
            rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.cam_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                self.cam_label.width(), self.cam_label.height(), Qt.KeepAspectRatio))
        except Exception as e:
            # If conversion fails, skip showing the frame but update labels
            print("Frame->QImage error:", e)

        # Update score/feedback UI
        self.score_label.setText(f"Score: {score_text}")
        self.feedback_label.setText(feedback)

    def closeEvent(self, event):
        self.timer.stop()
        if hasattr(self, "cap") and self.cap:
            self.cap.release()
        try:
            self.pose_estimator.release()
        except Exception:
            pass
        super().closeEvent(event)
