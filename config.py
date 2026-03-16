"""
Configuration settings for the Karate Trainer application.

This module centralizes all configurable parameters for easy tuning.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# Base paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
LEVELS_FILE = DATA_DIR / "levels.json"


@dataclass
class CameraConfig:
    """Camera/webcam configuration."""
    device_index: int = 0
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 30


@dataclass
class PoseEstimationConfig:
    """MediaPipe pose estimation configuration."""
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


@dataclass
class PoseMatchingConfig:
    """Pose matching/scoring configuration."""
    max_angle_tolerance_degrees: float = 60.0
    score_smoothing_window: int = 5
    default_passing_threshold: float = 60.0


@dataclass
class UIConfig:
    """User interface configuration."""
    window_title: str = "Karate Trainer"
    window_width: int = 1280
    window_height: int = 720
    frame_update_interval_ms: int = 30
    reference_overlay_alpha: float = 0.5


@dataclass
class FeedbackConfig:
    """Feedback generation configuration."""
    angle_diff_threshold_degrees: float = 15.0
    max_feedback_items: int = 3
    show_detailed_angles: bool = False


@dataclass
class AppConfig:
    """Main application configuration."""
    camera: CameraConfig = field(default_factory=CameraConfig)
    pose_estimation: PoseEstimationConfig = field(default_factory=PoseEstimationConfig)
    pose_matching: PoseMatchingConfig = field(default_factory=PoseMatchingConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    
    # Paths
    data_dir: Path = DATA_DIR
    levels_file: Path = LEVELS_FILE
    
    def __post_init__(self):
        """Ensure data directory exists."""
        self.data_dir.mkdir(parents=True, exist_ok=True)


# Global config instance (can be overridden for testing)
config = AppConfig()


def load_config_from_env() -> AppConfig:
    """
    Load configuration from environment variables.
    
    Environment variables override default values:
    - KARATE_CAMERA_DEVICE: Camera device index
    - KARATE_DETECTION_CONFIDENCE: Pose detection confidence
    - KARATE_TRACKING_CONFIDENCE: Pose tracking confidence
    - KARATE_WINDOW_WIDTH: Window width
    - KARATE_WINDOW_HEIGHT: Window height
    
    Returns:
        AppConfig instance with environment overrides applied
    """
    cfg = AppConfig()
    
    if device := os.environ.get("KARATE_CAMERA_DEVICE"):
        cfg.camera.device_index = int(device)
        
    if det_conf := os.environ.get("KARATE_DETECTION_CONFIDENCE"):
        cfg.pose_estimation.min_detection_confidence = float(det_conf)
        
    if track_conf := os.environ.get("KARATE_TRACKING_CONFIDENCE"):
        cfg.pose_estimation.min_tracking_confidence = float(track_conf)
        
    if width := os.environ.get("KARATE_WINDOW_WIDTH"):
        cfg.ui.window_width = int(width)
        
    if height := os.environ.get("KARATE_WINDOW_HEIGHT"):
        cfg.ui.window_height = int(height)
        
    return cfg
