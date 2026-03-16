# ui/feedback_panel.py
"""
Feedback panel widget for displaying pose comparison feedback to the user.
Shows score, textual feedback, and highlights body parts that need adjustment.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QFrame
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QColor, QPalette


# MediaPipe landmark indices to human-readable names
LANDMARK_NAMES = {
    0: "nose",
    1: "left eye (inner)",
    2: "left eye",
    3: "left eye (outer)",
    4: "right eye (inner)",
    5: "right eye",
    6: "right eye (outer)",
    7: "left ear",
    8: "right ear",
    9: "mouth (left)",
    10: "mouth (right)",
    11: "left shoulder",
    12: "right shoulder",
    13: "left elbow",
    14: "right elbow",
    15: "left wrist",
    16: "right wrist",
    17: "left pinky",
    18: "right pinky",
    19: "left index",
    20: "right index",
    21: "left thumb",
    22: "right thumb",
    23: "left hip",
    24: "right hip",
    25: "left knee",
    26: "right knee",
    27: "left ankle",
    28: "right ankle",
    29: "left heel",
    30: "right heel",
    31: "left foot index",
    32: "right foot index",
}

# Key body parts for feedback (subset of landmarks)
KEY_BODY_PARTS = {
    11: "left shoulder",
    12: "right shoulder",
    13: "left elbow",
    14: "right elbow",
    15: "left wrist",
    16: "right wrist",
    23: "left hip",
    24: "right hip",
    25: "left knee",
    26: "right knee",
    27: "left ankle",
    28: "right ankle",
}


class FeedbackPanel(QWidget):
    """
    A widget that displays real-time feedback about pose matching.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Score section
        score_frame = QFrame()
        score_frame.setFrameStyle(QFrame.StyledPanel)
        score_layout = QVBoxLayout(score_frame)
        
        self.score_label = QLabel("Score: --")
        self.score_label.setFont(QFont("Arial", 24, QFont.Bold))
        self.score_label.setAlignment(Qt.AlignCenter)
        score_layout.addWidget(self.score_label)
        
        self.score_bar = QProgressBar()
        self.score_bar.setRange(0, 100)
        self.score_bar.setValue(0)
        self.score_bar.setTextVisible(True)
        self.score_bar.setFormat("%v%")
        self.score_bar.setMinimumHeight(25)
        score_layout.addWidget(self.score_bar)
        
        layout.addWidget(score_frame)
        
        # Status/feedback section
        self.status_label = QLabel("Ready")
        self.status_label.setFont(QFont("Arial", 16))
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        
        # Detailed feedback section
        feedback_frame = QFrame()
        feedback_frame.setFrameStyle(QFrame.StyledPanel)
        feedback_layout = QVBoxLayout(feedback_frame)
        
        feedback_title = QLabel("Adjustments:")
        feedback_title.setFont(QFont("Arial", 12, QFont.Bold))
        feedback_layout.addWidget(feedback_title)
        
        self.feedback_list = QLabel("")
        self.feedback_list.setFont(QFont("Arial", 11))
        self.feedback_list.setWordWrap(True)
        self.feedback_list.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        feedback_layout.addWidget(self.feedback_list)
        
        layout.addWidget(feedback_frame)
        layout.addStretch()
        
    def update_score(self, score: float, threshold: float = 60.0):
        """
        Update the displayed score and progress bar.
        
        Args:
            score: The similarity score (0-100)
            threshold: The passing threshold for the current level
        """
        self.score_label.setText(f"Score: {score:.1f}%")
        self.score_bar.setValue(int(score))
        
        # Color the progress bar based on score vs threshold
        if score >= threshold:
            self._set_bar_color("#4CAF50")  # Green
        elif score >= threshold * 0.8:
            self._set_bar_color("#FFC107")  # Yellow/amber
        else:
            self._set_bar_color("#F44336")  # Red
            
    def _set_bar_color(self, hex_color: str):
        """Set the progress bar chunk color."""
        self.score_bar.setStyleSheet(f"""
            QProgressBar {{
                border: 1px solid #ccc;
                border-radius: 5px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {hex_color};
                border-radius: 4px;
            }}
        """)
        
    def update_status(self, status: str, is_success: bool = False):
        """
        Update the status message.
        
        Args:
            status: The status text to display
            is_success: Whether this is a success state (affects styling)
        """
        self.status_label.setText(status)
        if is_success:
            self.status_label.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            self.status_label.setStyleSheet("")
            
    def update_feedback(self, angle_diffs: list, max_items: int = 3):
        """
        Update the detailed feedback based on angle differences.
        
        Args:
            angle_diffs: List of (triplet_idx, ref_angle, user_angle, diff) tuples
            max_items: Maximum number of feedback items to show
        """
        # Filter out None diffs and sort by difference (worst first)
        valid_diffs = [(idx, ra, ua, d) for idx, ra, ua, d in angle_diffs if d is not None]
        valid_diffs.sort(key=lambda x: -x[3])
        
        feedback_lines = []
        for triplet_idx, ref_angle, user_angle, diff in valid_diffs[:max_items]:
            if diff > 15:  # Only show significant differences
                body_part = self._get_body_part_from_triplet(triplet_idx)
                direction = "higher" if user_angle < ref_angle else "lower"
                feedback_lines.append(f"• Adjust {body_part} ({diff:.0f}° off)")
                
        if feedback_lines:
            self.feedback_list.setText("\n".join(feedback_lines))
        else:
            self.feedback_list.setText("Looking good!")
            
    def _get_body_part_from_triplet(self, triplet_idx: int) -> str:
        """Map a triplet index to a human-readable body part name."""
        # These correspond to KEY_ANGLE_TRIPLETS in pose_matcher.py
        triplet_names = [
            "left arm",
            "right arm",
            "left shoulder",
            "right shoulder",
            "left leg",
            "right leg",
            "left hip",
            "right hip",
            "left foot",
            "right foot",
            "torso",
        ]
        if triplet_idx < len(triplet_names):
            return triplet_names[triplet_idx]
        return f"joint {triplet_idx}"
        
    def reset(self):
        """Reset the panel to initial state."""
        self.score_label.setText("Score: --")
        self.score_bar.setValue(0)
        self.status_label.setText("Ready")
        self.status_label.setStyleSheet("")
        self.feedback_list.setText("")


def generate_feedback_text(angle_diffs: list, threshold_deg: float = 15.0) -> list[str]:
    """
    Generate human-readable feedback from angle differences.
    
    Args:
        angle_diffs: List of (triplet_idx, ref_angle, user_angle, diff) tuples
        threshold_deg: Minimum angle difference to report
        
    Returns:
        List of feedback strings
    """
    triplet_names = [
        "left arm",
        "right arm", 
        "left shoulder",
        "right shoulder",
        "left leg",
        "right leg",
        "left hip",
        "right hip",
        "left foot",
        "right foot",
        "torso",
    ]
    
    feedback = []
    for triplet_idx, ref_angle, user_angle, diff in angle_diffs:
        if diff is None or diff < threshold_deg:
            continue
            
        body_part = triplet_names[triplet_idx] if triplet_idx < len(triplet_names) else f"joint {triplet_idx}"
        
        if user_angle < ref_angle:
            feedback.append(f"Raise your {body_part} slightly")
        else:
            feedback.append(f"Lower your {body_part} slightly")
            
    return feedback
