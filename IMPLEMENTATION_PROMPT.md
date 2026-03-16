# Karate Trainer Application - Complete Implementation Prompt

## Project Overview

Build a **real-time karate pose training application** that uses webcam input and MediaPipe pose estimation to compare the user's body pose against reference poses. The application provides visual feedback with a similarity score and suggestions for pose adjustment.

### Key Technologies
- **Python 3.12+**
- **PySide6** (Qt for Python) - Desktop GUI framework
- **MediaPipe 0.10.x** - Pose estimation (using Tasks API)
- **OpenCV** - Webcam capture and image processing
- **NumPy** - Numerical computations

---

## Project Structure

```
karate_trainer/
├── app.py                      # Entry point
├── config.py                   # Configuration dataclasses
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── core/
│   ├── __init__.py
│   ├── pose_estimator.py       # MediaPipe wrapper
│   ├── pose_matcher.py         # Angle-based pose comparison
│   ├── pose_loader.py          # JSON reference pose I/O
│   ├── feedback_engine.py      # Human-readable feedback generation
│   └── pose_landmarker.task    # MediaPipe model file
├── ui/
│   ├── __init__.py
│   ├── main_window.py          # Main application window
│   ├── pose_canvas.py          # Webcam display and pose overlay
│   └── feedback_panel.py       # Detailed feedback widget
├── data/
│   └── levels.json             # Reference poses for each level
└── scripts/
    ├── capture_reference_poses.py
    └── save_reference_pose.py
```

---

## Step 1: Entry Point (`app.py`)

Create a minimal entry point that initializes the Qt application and shows the main window.

```python
import sys
from PySide6.QtWidgets import QApplication
from ui.main_window import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
```

---

## Step 2: Configuration (`config.py`)

Define configuration using Python dataclasses for type safety and defaults.

```python
from dataclasses import dataclass, field
from typing import List, Tuple
import os

@dataclass
class CameraConfig:
    """Camera/webcam settings."""
    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    backends: List[str] = field(default_factory=lambda: ["V4L2", "ANY"])

@dataclass
class PoseEstimationConfig:
    """MediaPipe pose estimation settings."""
    model_complexity: int = 1  # 0=lite, 1=full, 2=heavy
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    enable_segmentation: bool = False
    model_path: str = "core/pose_landmarker.task"

@dataclass
class MatchingConfig:
    """Pose matching algorithm settings."""
    score_smoothing_window: int = 5
    angle_difference_threshold: float = 15.0  # degrees
    reference_overlay_alpha: float = 0.5

@dataclass
class UIConfig:
    """User interface settings."""
    window_title: str = "Karate Trainer"
    window_width: int = 1280
    window_height: int = 720
    score_font_size: int = 18
    feedback_font_size: int = 14

@dataclass
class PathConfig:
    """File paths."""
    levels_file: str = "data/levels.json"
    reference_pose_file: str = "reference_pose.json"
    
    def get_absolute_path(self, relative_path: str) -> str:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_dir, relative_path)

@dataclass
class AppConfig:
    """Main application configuration."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    pose_estimation: PoseEstimationConfig = field(default_factory=PoseEstimationConfig)
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    paths: PathConfig = field(default_factory=PathConfig)

# Global configuration instance
config = AppConfig()
```

---

## Step 3: Pose Estimator (`core/pose_estimator.py`)

Wrap MediaPipe pose detection. **Important**: MediaPipe 0.10.x uses the new Tasks API requiring a `.task` model file.

```python
import cv2
import numpy as np

class PoseEstimator:
    """
    Wrapper for MediaPipe Pose detection.
    Supports both legacy API (mp.solutions.pose) and new Tasks API (0.10.x+).
    """
    
    def __init__(self, model_path: str = None):
        self._use_tasks_api = False
        self._pose = None
        self._landmarker = None
        self._latest_result = None
        
        # Try new Tasks API first
        if model_path and self._try_init_tasks_api(model_path):
            self._use_tasks_api = True
        else:
            self._init_legacy_api()
    
    def _try_init_tasks_api(self, model_path: str) -> bool:
        """Initialize using MediaPipe Tasks API (0.10.x+)."""
        try:
            import mediapipe as mp
            import os
            
            # Resolve model path
            if not os.path.isabs(model_path):
                base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                model_path = os.path.join(base, model_path)
            
            if not os.path.exists(model_path):
                return False
            
            BaseOptions = mp.tasks.BaseOptions
            PoseLandmarker = mp.tasks.vision.PoseLandmarker
            PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
            VisionRunningMode = mp.tasks.vision.RunningMode
            
            def result_callback(result, output_image, timestamp_ms):
                self._latest_result = result
            
            options = PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=VisionRunningMode.LIVE_STREAM,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False,
                result_callback=result_callback
            )
            
            self._landmarker = PoseLandmarker.create_from_options(options)
            self._timestamp = 0
            return True
        except Exception as e:
            print(f"Tasks API init failed: {e}")
            return False
    
    def _init_legacy_api(self):
        """Initialize using legacy MediaPipe API."""
        import mediapipe as mp
        self._mp_pose = mp.solutions.pose
        self._mp_drawing = mp.solutions.drawing_utils
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def process_frame(self, frame, draw: bool = True):
        """
        Process a BGR frame and return (annotated_frame, landmarks).
        
        Returns:
            tuple: (annotated_frame, landmarks_list or None)
        """
        if self._use_tasks_api:
            return self._process_tasks_api(frame, draw)
        else:
            return self._process_legacy_api(frame, draw)
    
    def _process_tasks_api(self, frame, draw: bool):
        import mediapipe as mp
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        self._timestamp += 33  # ~30 FPS
        self._landmarker.detect_async(mp_image, self._timestamp)
        
        annotated = frame.copy()
        landmarks = None
        
        if self._latest_result and self._latest_result.pose_landmarks:
            landmarks = self._latest_result.pose_landmarks[0]
            
            if draw:
                self._draw_landmarks_on_frame(annotated, landmarks)
        
        return annotated, landmarks
    
    def _process_legacy_api(self, frame, draw: bool):
        import mediapipe as mp
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._pose.process(rgb)
        
        annotated = frame.copy()
        landmarks = None
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            if draw:
                self._mp_drawing.draw_landmarks(
                    annotated,
                    results.pose_landmarks,
                    self._mp_pose.POSE_CONNECTIONS
                )
        
        return annotated, landmarks
    
    def _draw_landmarks_on_frame(self, frame, landmarks):
        """Draw pose landmarks and connections on frame."""
        h, w = frame.shape[:2]
        
        CONNECTIONS = [
            (11, 13), (13, 15), (12, 14), (14, 16),  # arms
            (11, 12), (23, 24), (11, 23), (12, 24),  # torso
            (23, 25), (25, 27), (24, 26), (26, 28),  # legs
        ]
        
        points = []
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            points.append((x, y))
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        
        for a, b in CONNECTIONS:
            if a < len(points) and b < len(points):
                cv2.line(frame, points[a], points[b], (0, 255, 0), 2)
    
    def release(self):
        """Release resources."""
        if self._landmarker:
            self._landmarker.close()
        if self._pose:
            self._pose.close()
```

**Model File**: Download the pose landmark model:
```bash
curl -o core/pose_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
```

---

## Step 4: Pose Matcher (`core/pose_matcher.py`)

Implement angle-based pose comparison. This compares joint angles rather than absolute positions for scale/position invariance.

```python
import math
from typing import List, Dict, Tuple, Optional

# Key angle triplets: (point_a, vertex, point_b) for joint angle calculation
# These are MediaPipe landmark indices
KEY_ANGLE_TRIPLETS = [
    (11, 13, 15),  # Left arm (shoulder-elbow-wrist)
    (12, 14, 16),  # Right arm
    (13, 11, 23),  # Left shoulder (elbow-shoulder-hip)
    (14, 12, 24),  # Right shoulder
    (23, 25, 27),  # Left leg (hip-knee-ankle)
    (24, 26, 28),  # Right leg
    (11, 23, 25),  # Left hip (shoulder-hip-knee)
    (12, 24, 26),  # Right hip
    (25, 27, 31),  # Left foot (knee-ankle-foot)
    (26, 28, 32),  # Right foot
    (11, 12, 24),  # Torso
]

def landmarks_to_2d_list(landmarks) -> List[Dict[str, float]]:
    """Convert various landmark formats to list of {x, y} dicts."""
    result = []
    
    if hasattr(landmarks, 'landmark'):
        landmarks = landmarks.landmark
    
    for lm in landmarks:
        if isinstance(lm, dict):
            result.append({'x': float(lm.get('x', 0)), 'y': float(lm.get('y', 0))})
        elif hasattr(lm, 'x') and hasattr(lm, 'y'):
            result.append({'x': float(lm.x), 'y': float(lm.y)})
        elif isinstance(lm, (list, tuple)) and len(lm) >= 2:
            result.append({'x': float(lm[0]), 'y': float(lm[1])})
    
    return result

def compute_angle(p1: Dict, p2: Dict, p3: Dict) -> Optional[float]:
    """
    Compute angle at p2 formed by p1-p2-p3 in degrees.
    Returns None if points are too close.
    """
    v1 = (p1['x'] - p2['x'], p1['y'] - p2['y'])
    v2 = (p3['x'] - p2['x'], p3['y'] - p2['y'])
    
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    if mag1 < 1e-6 or mag2 < 1e-6:
        return None
    
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    cos_angle = max(-1, min(1, dot / (mag1 * mag2)))
    
    return math.degrees(math.acos(cos_angle))

def compute_angle_differences(
    user_landmarks: List[Dict],
    ref_landmarks: List[Dict]
) -> List[Tuple[int, Optional[float], Optional[float], Optional[float]]]:
    """
    Compute angle differences for all key triplets.
    
    Returns:
        List of (triplet_idx, ref_angle, user_angle, abs_difference)
    """
    results = []
    
    for idx, (a, b, c) in enumerate(KEY_ANGLE_TRIPLETS):
        ref_angle = None
        user_angle = None
        diff = None
        
        if a < len(ref_landmarks) and b < len(ref_landmarks) and c < len(ref_landmarks):
            ref_angle = compute_angle(ref_landmarks[a], ref_landmarks[b], ref_landmarks[c])
        
        if a < len(user_landmarks) and b < len(user_landmarks) and c < len(user_landmarks):
            user_angle = compute_angle(user_landmarks[a], user_landmarks[b], user_landmarks[c])
        
        if ref_angle is not None and user_angle is not None:
            diff = abs(ref_angle - user_angle)
        
        results.append((idx, ref_angle, user_angle, diff))
    
    return results

def similarity_score(
    user_landmarks: List[Dict],
    ref_landmarks: List[Dict],
    max_angle_diff: float = 90.0
) -> float:
    """
    Compute similarity score (0-100) based on angle matching.
    
    Higher score = better match.
    """
    diffs = compute_angle_differences(user_landmarks, ref_landmarks)
    
    valid_diffs = [d for _, _, _, d in diffs if d is not None]
    
    if not valid_diffs:
        return 0.0
    
    avg_diff = sum(valid_diffs) / len(valid_diffs)
    
    # Convert to 0-100 score (0° diff = 100%, max_angle_diff = 0%)
    score = max(0, 100 * (1 - avg_diff / max_angle_diff))
    
    return score
```

---

## Step 5: Pose Loader (`core/pose_loader.py`)

Handle loading and saving reference poses from JSON.

```python
import json
import os
from typing import List, Dict, Optional

def load_levels(file_path: str) -> List[Dict]:
    """Load levels from JSON file."""
    if not os.path.exists(file_path):
        return []
    
    with open(file_path, 'r') as f:
        return json.load(f)

def load_reference_pose(file_path: str) -> Optional[List[Dict]]:
    """Load a single reference pose from JSON."""
    if not os.path.exists(file_path):
        return None
    
    with open(file_path, 'r') as f:
        return json.load(f)

def save_reference_pose(landmarks: List[Dict], file_path: str) -> None:
    """Save landmarks to JSON file."""
    with open(file_path, 'w') as f:
        json.dump(landmarks, f, indent=2)
```

---

## Step 6: Feedback Engine (`core/feedback_engine.py`)

Generate human-readable feedback from pose differences.

```python
from typing import List, Tuple, Optional

JOINT_NAMES = {
    0: "left arm",
    1: "right arm",
    2: "left shoulder",
    3: "right shoulder",
    4: "left leg",
    5: "right leg",
    6: "left hip",
    7: "right hip",
    8: "left foot",
    9: "right foot",
    10: "torso",
}

def generate_feedback(
    angle_diffs: List[Tuple[int, Optional[float], Optional[float], Optional[float]]],
    threshold: float = 15.0
) -> List[str]:
    """
    Generate feedback messages from angle differences.
    
    Args:
        angle_diffs: List of (triplet_idx, ref_angle, user_angle, diff)
        threshold: Minimum difference in degrees to report
        
    Returns:
        List of feedback strings
    """
    feedback = []
    
    # Filter and sort by difference (worst first)
    valid = [(idx, ra, ua, d) for idx, ra, ua, d in angle_diffs if d is not None and d > threshold]
    valid.sort(key=lambda x: -x[3])
    
    for triplet_idx, ref_angle, user_angle, diff in valid[:3]:
        joint_name = JOINT_NAMES.get(triplet_idx, f"joint {triplet_idx}")
        
        if user_angle < ref_angle:
            direction = "extend"
        else:
            direction = "bend"
        
        feedback.append(f"Adjust your {joint_name} - {direction} more ({diff:.0f}° off)")
    
    return feedback

def generate_status(score: float, threshold: float) -> Tuple[str, bool]:
    """
    Generate status message based on score.
    
    Returns:
        (message, is_success)
    """
    if score >= threshold:
        return "Great! Pose matched!", True
    elif score >= threshold * 0.8:
        return "Almost there - adjust slightly", False
    else:
        return "Keep trying - align to the reference", False
```

---

## Step 7: Main Window (`ui/main_window.py`)

Simple container for the pose canvas widget.

```python
from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from ui.pose_canvas import PoseCanvas

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Karate Trainer")
        self.setGeometry(100, 100, 1280, 720)
        
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        
        self.pose_canvas = PoseCanvas()
        layout.addWidget(self.pose_canvas)
        
        self.setCentralWidget(central_widget)
```

---

## Step 8: Pose Canvas (`ui/pose_canvas.py`)

The main widget handling webcam display, pose detection, and score calculation.

### Key Features:
1. **Webcam Capture**: Uses OpenCV with fallback camera backends
2. **Level Selection**: Dropdown to select different karate poses
3. **Score Display**: Shows smoothed similarity score
4. **Reference Overlay**: Draws semi-transparent reference skeleton
5. **Worst Angle Highlighting**: Marks joints that need most adjustment

```python
import json
import cv2
from collections import deque
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap, QFont

from core.pose_estimator import PoseEstimator
from core.pose_matcher import similarity_score, compute_angle_differences, KEY_ANGLE_TRIPLETS

# Skeleton connections for drawing reference overlay
POSE_CONNECTIONS = [
    (11, 13), (13, 15), (12, 14), (14, 16),  # arms
    (11, 12), (23, 24), (11, 23), (12, 24),  # torso
    (23, 25), (25, 27), (24, 26), (26, 28),  # legs
]

class PoseCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # UI Setup
        self.cam_label = QLabel()
        self.cam_label.setAlignment(Qt.AlignCenter)
        
        self.score_label = QLabel("Score: -")
        self.score_label.setFont(QFont("Arial", 18))
        
        self.feedback_label = QLabel("")
        self.feedback_label.setFont(QFont("Arial", 14))
        
        # Level controls
        controls = QHBoxLayout()
        self.level_combo = QComboBox()
        self.prev_btn = QPushButton("Prev")
        self.next_btn = QPushButton("Next")
        controls.addWidget(self.prev_btn)
        controls.addWidget(self.level_combo)
        controls.addWidget(self.next_btn)
        
        layout = QVBoxLayout()
        layout.addLayout(controls)
        layout.addWidget(self.cam_label)
        layout.addWidget(self.score_label)
        layout.addWidget(self.feedback_label)
        self.setLayout(layout)
        
        # Connect signals
        self.prev_btn.clicked.connect(self.prev_level)
        self.next_btn.clicked.connect(self.next_level)
        self.level_combo.currentIndexChanged.connect(self.on_level_change)
        
        # Initialize pose estimator
        self.pose_estimator = PoseEstimator(model_path="core/pose_landmarker.task")
        
        # Open camera
        self.cap = self._try_open_camera()
        
        # Frame update timer (30 FPS)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)
        
        # Load levels and state
        self.levels = []
        self.current_level_index = 0
        self.current_level = None
        self.score_history = deque(maxlen=5)  # Smoothing
        self.load_levels()
    
    def _try_open_camera(self):
        """Try to open webcam with multiple backends."""
        backends = [(cv2.CAP_V4L2, "V4L2"), (cv2.CAP_ANY, "ANY")]
        
        for idx in range(3):
            for backend, name in backends:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    print(f"Opened camera {idx} with {name}")
                    return cap
                cap.release()
        
        print("Warning: No webcam found")
        return None
    
    def load_levels(self):
        """Load levels from JSON file."""
        import os
        base = os.path.dirname(os.path.dirname(__file__))
        levels_path = os.path.join(base, "data", "levels.json")
        
        try:
            with open(levels_path, 'r') as f:
                self.levels = json.load(f)
        except Exception as e:
            print(f"Could not load levels: {e}")
            self.levels = []
        
        # Populate combo box
        self.level_combo.clear()
        for lv in self.levels:
            self.level_combo.addItem(lv.get('name', f"Level {lv.get('id')}"))
        
        if self.levels:
            self.current_level = self.levels[0]
    
    def prev_level(self):
        if self.levels:
            self.current_level_index = (self.current_level_index - 1) % len(self.levels)
            self.level_combo.setCurrentIndex(self.current_level_index)
    
    def next_level(self):
        if self.levels:
            self.current_level_index = (self.current_level_index + 1) % len(self.levels)
            self.level_combo.setCurrentIndex(self.current_level_index)
    
    def on_level_change(self, idx):
        if 0 <= idx < len(self.levels):
            self.current_level_index = idx
            self.current_level = self.levels[idx]
            self.score_history.clear()
    
    def update_frame(self):
        """Main frame update loop."""
        if not self.cap or not self.cap.isOpened():
            self.score_label.setText("Score: No camera")
            return
        
        ret, frame = self.cap.read()
        if not ret:
            return
        
        # Get pose estimation
        annotated_frame, mp_landmarks = self.pose_estimator.process_frame(frame, draw=True)
        
        # Convert landmarks to dict format
        user_landmarks = self._convert_landmarks(mp_landmarks)
        
        score_text = "-"
        feedback = ""
        
        if self.current_level and user_landmarks:
            ref = self.current_level.get('reference_pose')
            if ref:
                # Calculate score
                score = similarity_score(user_landmarks, ref)
                self.score_history.append(score)
                smoothed = sum(self.score_history) / len(self.score_history)
                
                score_text = f"{smoothed:.1f}%"
                threshold = self.current_level.get('threshold', 60)
                
                # Generate feedback
                if smoothed >= threshold:
                    feedback = "Good — matched!"
                elif smoothed >= threshold * 0.8:
                    feedback = "Close — adjust posture"
                else:
                    feedback = "Try again — align to reference"
                
                # Draw reference overlay
                self._draw_reference_overlay(annotated_frame, ref)
                
                # Highlight worst joints
                self._highlight_worst_joints(annotated_frame, user_landmarks, ref)
        
        # Display frame
        self._display_frame(annotated_frame)
        
        # Update labels
        self.score_label.setText(f"Score: {score_text}")
        self.feedback_label.setText(feedback)
    
    def _convert_landmarks(self, landmarks):
        """Convert MediaPipe landmarks to list of dicts."""
        if landmarks is None:
            return []
        
        result = []
        iterable = landmarks.landmark if hasattr(landmarks, 'landmark') else landmarks
        
        for lm in iterable:
            if hasattr(lm, 'x') and hasattr(lm, 'y'):
                result.append({'x': float(lm.x), 'y': float(lm.y)})
            elif isinstance(lm, dict):
                result.append({'x': float(lm['x']), 'y': float(lm['y'])})
        
        return result
    
    def _draw_reference_overlay(self, frame, ref_landmarks, alpha=0.5):
        """Draw semi-transparent reference skeleton."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        pts = [(int(lm['x'] * w), int(lm['y'] * h)) for lm in ref_landmarks]
        
        for pt in pts:
            cv2.circle(overlay, pt, 3, (0, 200, 200), -1)
        
        for a, b in POSE_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(overlay, pts[a], pts[b], (0, 200, 200), 2)
        
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    def _highlight_worst_joints(self, frame, user_landmarks, ref_landmarks):
        """Draw red circles on joints with biggest angle differences."""
        diffs = compute_angle_differences(user_landmarks, ref_landmarks)
        valid = [(i, d) for i, _, _, d in diffs if d is not None]
        valid.sort(key=lambda x: -x[1])
        
        h, w = frame.shape[:2]
        for triplet_idx, _ in valid[:2]:  # Top 2 worst
            _, vertex, _ = KEY_ANGLE_TRIPLETS[triplet_idx]
            if vertex < len(ref_landmarks):
                x = int(ref_landmarks[vertex]['x'] * w)
                y = int(ref_landmarks[vertex]['y'] * h)
                cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)
    
    def _display_frame(self, frame):
        """Convert OpenCV frame to Qt and display."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self.cam_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
            self.cam_label.width(), self.cam_label.height(), Qt.KeepAspectRatio))
    
    def closeEvent(self, event):
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.pose_estimator.release()
        super().closeEvent(event)
```

---

## Step 9: Feedback Panel (`ui/feedback_panel.py`)

A detailed feedback widget showing score bar and adjustment suggestions.

```python
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QProgressBar, QFrame
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

class FeedbackPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Score section
        self.score_label = QLabel("Score: --")
        self.score_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.score_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.score_label)
        
        self.score_bar = QProgressBar()
        self.score_bar.setRange(0, 100)
        self.score_bar.setValue(0)
        layout.addWidget(self.score_bar)
        
        # Status
        self.status_label = QLabel("Ready")
        self.status_label.setFont(QFont("Arial", 16))
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        
        # Adjustments
        self.feedback_list = QLabel("")
        self.feedback_list.setWordWrap(True)
        layout.addWidget(self.feedback_list)
        
        layout.addStretch()
    
    def update_score(self, score: float, threshold: float = 60.0):
        self.score_label.setText(f"Score: {score:.1f}%")
        self.score_bar.setValue(int(score))
        
        # Color based on score
        if score >= threshold:
            color = "#4CAF50"  # Green
        elif score >= threshold * 0.8:
            color = "#FFC107"  # Yellow
        else:
            color = "#F44336"  # Red
        
        self.score_bar.setStyleSheet(f"""
            QProgressBar::chunk {{ background-color: {color}; }}
        """)
    
    def update_feedback(self, messages: list):
        self.feedback_list.setText("\n".join(f"• {msg}" for msg in messages))
```

---

## Step 10: Reference Pose Data (`data/levels.json`)

JSON file containing reference poses. Each level has:
- `id`: Unique identifier
- `name`: Display name
- `type`: Category (kick, punch, stance)
- `threshold`: Minimum score to pass (0-100)
- `reference_pose`: Array of 33 landmarks (MediaPipe format)

```json
[
  {
    "id": 1,
    "name": "Front Kick (Right)",
    "type": "kick",
    "threshold": 65,
    "reference_pose": [
      {"x": 0.64, "y": 0.56, "z": -0.87, "visibility": 0.99},
      // ... 32 more landmarks (33 total for MediaPipe Pose)
    ]
  },
  {
    "id": 2,
    "name": "Side Kick (Left)",
    "type": "kick",
    "threshold": 65,
    "reference_pose": [...]
  }
]
```

### Landmark Structure
Each landmark follows MediaPipe's 33-point pose model:
- `x`, `y`: Normalized coordinates (0-1) relative to frame
- `z`: Depth (relative to hips)
- `visibility`: Confidence score (0-1)

---

## Step 11: Package Initialization

### `core/__init__.py`
```python
from .pose_estimator import PoseEstimator
from .pose_matcher import similarity_score, compute_angle_differences
from .pose_loader import load_levels, load_reference_pose
from .feedback_engine import generate_feedback

__all__ = [
    'PoseEstimator',
    'similarity_score',
    'compute_angle_differences',
    'load_levels',
    'load_reference_pose',
    'generate_feedback',
]
```

### `ui/__init__.py`
```python
from .main_window import MainWindow
from .pose_canvas import PoseCanvas
from .feedback_panel import FeedbackPanel

__all__ = ['MainWindow', 'PoseCanvas', 'FeedbackPanel']
```

---

## Step 12: Dependencies (`requirements.txt`)

```
PySide6>=6.5.0
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
```

---

## Step 13: Running the Application

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download MediaPipe model
curl -o core/pose_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
```

### Run
```bash
python app.py
```

---

## Algorithm Details

### Pose Matching Algorithm

The matching algorithm uses **angle-based comparison** rather than absolute position comparison. This provides:

1. **Scale Invariance**: Works regardless of user's distance from camera
2. **Position Invariance**: User doesn't need to be centered exactly
3. **Intuitive Feedback**: Angles correspond to joint positions

**Key Angle Triplets** (11 total):
| Index | Triplet | Body Part |
|-------|---------|-----------|
| 0 | (11, 13, 15) | Left arm |
| 1 | (12, 14, 16) | Right arm |
| 2 | (13, 11, 23) | Left shoulder |
| 3 | (14, 12, 24) | Right shoulder |
| 4 | (23, 25, 27) | Left leg |
| 5 | (24, 26, 28) | Right leg |
| 6 | (11, 23, 25) | Left hip |
| 7 | (12, 24, 26) | Right hip |
| 8 | (25, 27, 31) | Left foot |
| 9 | (26, 28, 32) | Right foot |
| 10 | (11, 12, 24) | Torso |

### Score Calculation
```
average_angle_diff = mean(|ref_angle - user_angle| for each triplet)
score = 100 * (1 - average_angle_diff / 90°)
```

Score range: 0-100 where 100 = perfect match

### Smoothing
Scores are smoothed using a rolling average of the last 5 frames to reduce jitter.

---

## Future Enhancements

1. **Progress Tracking**: Save user's best scores per level
2. **Tutorial Mode**: Step-by-step pose breakdown
3. **Voice Feedback**: Text-to-speech for hands-free training
4. **Timing Challenges**: Hold pose for X seconds
5. **Multi-person Mode**: Train with a partner
6. **Recording/Playback**: Record sessions for review
7. **Difficulty Scaling**: Adjust thresholds dynamically
