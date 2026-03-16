"""
Pose estimation module using MediaPipe.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# Try new MediaPipe API first, fall back to legacy
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision
    from mediapipe import solutions
    USE_LEGACY_API = False
except (ImportError, AttributeError):
    USE_LEGACY_API = True

# Check if solutions is available (legacy API)
try:
    import mediapipe as mp
    _ = mp.solutions.pose
    USE_LEGACY_API = True
except AttributeError:
    USE_LEGACY_API = False


class PoseEstimator:
    """Wrapper around MediaPipe Pose for pose estimation from video frames."""
    
    def __init__(self, detection_confidence: float = 0.5, tracking_confidence: float = 0.5) -> None:
        """
        Initialize the pose estimator.
        
        Args:
            detection_confidence: Minimum confidence for person detection (0.0-1.0)
            tracking_confidence: Minimum confidence for landmark tracking (0.0-1.0)
        """
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        
        if USE_LEGACY_API:
            # Legacy MediaPipe API (< 0.10.x)
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                min_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence
            )
            self.mp_drawing = mp.solutions.drawing_utils
            self._use_legacy = True
        else:
            # New MediaPipe Tasks API (>= 0.10.x)
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.components import containers
            
            base_options = mp_tasks.BaseOptions(
                model_asset_path=self._get_model_path()
            )
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                min_pose_detection_confidence=detection_confidence,
                min_tracking_confidence=tracking_confidence,
            )
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
            self._use_legacy = False
            self._frame_timestamp = 0
    
    def _get_model_path(self) -> str:
        """Get the path to the pose landmarker model."""
        import os
        # Check for local model first
        local_model = os.path.join(os.path.dirname(__file__), "pose_landmarker.task")
        if os.path.exists(local_model):
            return local_model
        # Default model path
        return "pose_landmarker.task"

    def process_frame(self, frame: np.ndarray, draw: bool = True) -> tuple[np.ndarray, Any]:
        """
        Process a BGR frame and extract pose landmarks.
        
        Args:
            frame: BGR image as numpy array
            draw: Whether to draw landmarks on the frame
            
        Returns:
            Tuple of (annotated_frame, pose_landmarks)
            pose_landmarks is None if no pose detected
        """
        if self._use_legacy:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)
            
            landmarks = results.pose_landmarks
            if draw and landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, landmarks, self.mp_pose.POSE_CONNECTIONS
                )
            return frame, landmarks
        else:
            # New Tasks API
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            
            self._frame_timestamp += 33  # ~30fps
            result = self.landmarker.detect_for_video(mp_image, self._frame_timestamp)
            
            landmarks = None
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]
                if draw:
                    self._draw_landmarks_new_api(frame, landmarks)
            
            return frame, landmarks
    
    def _draw_landmarks_new_api(self, frame: np.ndarray, landmarks: list) -> None:
        """Draw landmarks using the new API format."""
        h, w = frame.shape[:2]
        
        # Define pose connections
        POSE_CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10),
            (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
            (11, 23), (12, 24), (23, 24),
            (23, 25), (25, 27), (27, 29), (27, 31),
            (24, 26), (26, 28), (28, 30), (28, 32),
        ]
        
        # Draw landmarks
        for lm in landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
        
        # Draw connections
        for start, end in POSE_CONNECTIONS:
            if start < len(landmarks) and end < len(landmarks):
                x1 = int(landmarks[start].x * w)
                y1 = int(landmarks[start].y * h)
                x2 = int(landmarks[end].x * w)
                y2 = int(landmarks[end].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def draw_landmarks(self, frame: np.ndarray, landmarks: Any) -> np.ndarray:
        """
        Draw pose landmarks on a frame with custom styling.
        
        Args:
            frame: BGR image to draw on
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Frame with landmarks drawn
        """
        if self._use_legacy:
            self.mp_drawing.draw_landmarks(
                frame,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            )
        else:
            self._draw_landmarks_new_api(frame, landmarks)
        return frame

    def release(self) -> None:
        """Release resources used by the pose estimator."""
        if self._use_legacy:
            self.pose.close()
        else:
            self.landmarker.close()
if __name__ == "__main__":
    # Quick standalone test
    cap = cv2.VideoCapture(0)
    estimator = PoseEstimator()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot access webcam")
            break

        annotated_frame, landmarks = estimator.process_frame(frame, draw=True)
        cv2.imshow("Pose Estimator Test", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    estimator.release()
