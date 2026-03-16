"""
Pose matching module for comparing user poses against reference poses.

Uses angle-based comparison for robustness to distance and position differences.
"""
from __future__ import annotations

import math
from typing import Any

# Type aliases for clarity
Point2D = tuple[float, float]
LandmarkDict = dict[str, float]
AngleDiff = tuple[int, float | None, float | None, float | None]

# Landmarks indices follow MediaPipe Pose (0..32).
# We'll compute angles for key triplets: (a,b,c) -> angle at b formed by ba and bc
# Choose triplets meaningful for kicks/punches/defense
KEY_ANGLE_TRIPLETS: list[tuple[int, int, int]] = [
    # arms (elbow angles)
    (11, 13, 15),  # left shoulder-elbow-wrist (angle at 13)
    (12, 14, 16),  # right shoulder-elbow-wrist
    # shoulder rotation (upper arm relative to torso)
    (23, 11, 13),  # left hip - left shoulder - left elbow (torso/shoulder)
    (24, 12, 14),  # right hip - right shoulder - right elbow
    # hips / legs
    (23, 25, 27),  # left hip-knee-ankle (angle at 25)
    (24, 26, 28),  # right hip-knee-ankle
    # torso lean / hip rotation
    (11, 23, 25),  # left shoulder-hip-knee
    (12, 24, 26),  # right shoulder-hip-knee
    # knees / small adjustments
    (25, 27, 31),  # left knee-ankle-foot-index
    (26, 28, 32),  # right knee-ankle-foot-index
    # optional: neck/head orientation
    (11, 23, 24),  # left shoulder-hip-right hip (torso twist)
]

# Weights for each angle (same length as KEY_ANGLE_TRIPLETS)
DEFAULT_WEIGHTS: list[float] = [1.0] * len(KEY_ANGLE_TRIPLETS)

# Angle difference that maps to zero score for that joint
MAX_ANGLE_TOLERANCE_DEGREES: float = 60.0


def _angle_between(a: Point2D, b: Point2D, c: Point2D) -> float | None:
    """
    Compute angle at point b formed by points a-b-c in degrees.
    
    Args:
        a: First point (x, y)
        b: Vertex point (x, y) - angle is measured here
        c: Third point (x, y)
        
    Returns:
        Angle in degrees (0..180), or None if computation fails
    """
    bax = a[0] - b[0]
    bay = a[1] - b[1]
    cbx = c[0] - b[0]
    cby = c[1] - b[1]

    # Compute dot product and magnitudes
    dot = bax * cbx + bay * cby
    mag1 = math.hypot(bax, bay)
    mag2 = math.hypot(cbx, cby)
    
    if mag1 == 0 or mag2 == 0:
        return None
        
    cosang = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    ang = math.degrees(math.acos(cosang))
    return ang


def landmarks_to_2d_list(landmarks: list[Any]) -> list[Point2D]:
    """
    Convert landmarks to a list of 2D points.
    
    Args:
        landmarks: List of dicts with 'x','y' keys, or list of tuples/lists
        
    Returns:
        List of (x, y) tuples
    """
    if not landmarks:
        return []
        
    if isinstance(landmarks[0], (tuple, list)):
        return [(float(p[0]), float(p[1])) for p in landmarks]
        
    # Assume dicts with x, y keys
    out = []
    for lm in landmarks:
        out.append((lm.get('x', 0.0), lm.get('y', 0.0)))
    return out


def compute_angle_differences(
    user_landmarks: list[Any],
    ref_landmarks: list[Any],
    triplets: list[tuple[int, int, int]] = KEY_ANGLE_TRIPLETS
) -> list[AngleDiff]:
    """
    Compute angle differences between user and reference poses.
    
    Args:
        user_landmarks: User's detected landmarks
        ref_landmarks: Reference pose landmarks
        triplets: List of (a, b, c) index triplets defining angles
        
    Returns:
        List of (triplet_index, ref_angle, user_angle, abs_diff_degrees) tuples.
        Angles may be None if computation failed.
    """
    u = landmarks_to_2d_list(user_landmarks)
    r = landmarks_to_2d_list(ref_landmarks)
    
    results: list[AngleDiff] = []
    for idx, (a, b, c) in enumerate(triplets):
        try:
            ra = _angle_between(r[a], r[b], r[c])
            ua = _angle_between(u[a], u[b], u[c])
        except (IndexError, TypeError):
            ra = ua = None
            
        if ra is None or ua is None:
            diff = None
        else:
            diff = abs(ra - ua)
            
        results.append((idx, ra, ua, diff))
        
    return results


def score_from_diffs(
    diffs: list[AngleDiff],
    weights: list[float] | None = None,
    max_tolerance: float = MAX_ANGLE_TOLERANCE_DEGREES
) -> float:
    """
    Calculate similarity score from angle differences.
    
    Args:
        diffs: List of (idx, ref_angle, user_angle, diff) tuples
        weights: Optional weights for each angle (defaults to equal weights)
        max_tolerance: Maximum angle difference before score becomes 0
        
    Returns:
        Normalized score from 0 to 100
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
        
    total_weight = 0.0
    acc = 0.0
    
    for i, (idx, ra, ua, diff) in enumerate(diffs):
        w = weights[i] if i < len(weights) else 1.0
        total_weight += w
        
        if diff is None:
            # Missing data - penalize by assuming max diff
            normalized = 0.0
        else:
            clipped = min(diff, max_tolerance)
            # 1 when diff=0, 0 when diff>=max_tolerance
            normalized = max(0.0, 1.0 - (clipped / max_tolerance))
            
        acc += w * normalized
        
    if total_weight == 0:
        return 0.0
        
    score = (acc / total_weight) * 100.0
    return float(score)


def similarity_score(
    user_landmarks: list[Any],
    ref_landmarks: list[Any],
    weights: list[float] | None = None,
    max_tolerance: float = MAX_ANGLE_TOLERANCE_DEGREES
) -> float:
    """
    Compute similarity score between user pose and reference pose.
    
    This is a convenience function that combines compute_angle_differences
    and score_from_diffs into a single call.
    
    Args:
        user_landmarks: User's detected landmarks
        ref_landmarks: Reference pose landmarks  
        weights: Optional weights for each angle
        max_tolerance: Maximum angle difference before score becomes 0
        
    Returns:
        Similarity score from 0 to 100
    """
    diffs = compute_angle_differences(user_landmarks, ref_landmarks)
    return score_from_diffs(diffs, weights=weights, max_tolerance=max_tolerance)
