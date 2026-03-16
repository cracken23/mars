"""
Feedback engine for generating human-readable pose adjustment suggestions.
"""
from __future__ import annotations


# MediaPipe Pose landmark names (indices 0-32)
JOINT_NAMES: list[str] = [
    "nose",
    "left eye (inner)",
    "left eye",
    "left eye (outer)",
    "right eye (inner)",
    "right eye",
    "right eye (outer)",
    "left ear",
    "right ear",
    "mouth (left)",
    "mouth (right)",
    "left shoulder",
    "right shoulder",
    "left elbow",
    "right elbow",
    "left wrist",
    "right wrist",
    "left pinky",
    "right pinky",
    "left index",
    "right index",
    "left thumb",
    "right thumb",
    "left hip",
    "right hip",
    "left knee",
    "right knee",
    "left ankle",
    "right ankle",
    "left heel",
    "right heel",
    "left foot index",
    "right foot index",
]


def generate_feedback(joint_status: list[bool]) -> list[str]:
    """
    Generate feedback messages for joints that need adjustment.
    
    Args:
        joint_status: List of booleans indicating if each joint is correctly positioned.
                      True = correct, False = needs adjustment.
                      
    Returns:
        List of feedback strings for incorrect joints.
    """
    feedback = []
    for idx, ok in enumerate(joint_status):
        if not ok and idx < len(JOINT_NAMES):
            feedback.append(f"Adjust your {JOINT_NAMES[idx]}")
    return feedback


def generate_detailed_feedback(
    angle_diffs: list[tuple[int, float | None, float | None, float | None]],
    threshold_degrees: float = 15.0
) -> list[str]:
    """
    Generate detailed feedback based on angle differences.
    
    Args:
        angle_diffs: List of (triplet_idx, ref_angle, user_angle, diff) tuples
        threshold_degrees: Minimum angle difference to report
        
    Returns:
        List of specific feedback messages
    """
    # Body part names corresponding to KEY_ANGLE_TRIPLETS in pose_matcher.py
    triplet_names = [
        "left arm (elbow)",
        "right arm (elbow)",
        "left shoulder",
        "right shoulder",
        "left leg (knee)",
        "right leg (knee)",
        "left hip",
        "right hip",
        "left foot",
        "right foot",
        "torso rotation",
    ]
    
    feedback = []
    for triplet_idx, ref_angle, user_angle, diff in angle_diffs:
        if diff is None or diff < threshold_degrees:
            continue
            
        body_part = triplet_names[triplet_idx] if triplet_idx < len(triplet_names) else f"joint {triplet_idx}"
        
        if user_angle is not None and ref_angle is not None:
            if user_angle < ref_angle:
                feedback.append(f"Extend your {body_part} more ({diff:.0f}° difference)")
            else:
                feedback.append(f"Bend your {body_part} more ({diff:.0f}° difference)")
        else:
            feedback.append(f"Adjust your {body_part}")
            
    return feedback
