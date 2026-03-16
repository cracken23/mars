# UI module - PySide6 widgets and windows
from .main_window import MainWindow
from .pose_canvas import PoseCanvas
from .feedback_panel import FeedbackPanel, generate_feedback_text

__all__ = [
    "MainWindow",
    "PoseCanvas",
    "FeedbackPanel",
    "generate_feedback_text",
]
