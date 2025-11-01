# karate_trainer/core/pose_matcher.py
import math

# Landmarks indices follow MediaPipe Pose (0..32).
# We'll compute angles for key triplets: (a,b,c) -> angle at b formed by ba and bc
# Choose triplets meaningful for kicks/punches/defense
KEY_ANGLE_TRIPLETS = [
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

# weights for each angle (same length as KEY_ANGLE_TRIPLETS)
DEFAULT_WEIGHTS = [1.0] * len(KEY_ANGLE_TRIPLETS)

MAX_ANGLE_TOLERANCE_DEGREES = 60.0  # angle difference that maps to zero score for that joint

def _angle_between(a, b, c):
    """
    Compute angle at point b formed by points a-b-c in degrees.
    Input: a,b,c are (x,y) tuples (normalized coords fine).
    Returns angle in degrees (0..180)
    """
    bax = a[0] - b[0]
    bay = a[1] - b[1]
    cbx = c[0] - b[0]
    cby = c[1] - b[1]

    # compute dot product and norms
    dot = bax * cbx + bay * cby
    mag1 = math.hypot(bax, bay)
    mag2 = math.hypot(cbx, cby)
    if mag1 == 0 or mag2 == 0:
        return None
    cosang = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    ang = math.degrees(math.acos(cosang))
    return ang

def landmarks_to_2d_list(landmarks):
    """
    Convert MediaPipe-style landmarks (list of dicts with 'x','y') to list of tuples.
    If input is already a list of tuples, it returns as-is.
    """
    if not landmarks:
        return []
    if isinstance(landmarks[0], tuple) or isinstance(landmarks[0], list):
        return [(p[0], p[1]) for p in landmarks]
    # assume dicts with x,y
    out = []
    for lm in landmarks:
        out.append((lm.get('x', 0.0), lm.get('y', 0.0)))
    return out

def compute_angle_differences(user_landmarks, ref_landmarks, triplets=KEY_ANGLE_TRIPLETS):
    """
    Returns list of (index, ref_angle, user_angle, abs_diff_deg).
    user_landmarks/ref_landmarks: list-like of (x,y) or dicts with x,y
    """
    u = landmarks_to_2d_list(user_landmarks)
    r = landmarks_to_2d_list(ref_landmarks)
    results = []
    for idx, (a,b,c) in enumerate(triplets):
        try:
            ra = _angle_between(r[a], r[b], r[c])
            ua = _angle_between(u[a], u[b], u[c])
        except Exception:
            ra = ua = None
        if ra is None or ua is None:
            diff = None
        else:
            diff = abs(ra - ua)
        results.append((idx, ra, ua, diff))
    return results

def score_from_diffs(diffs, weights=None, max_tolerance=MAX_ANGLE_TOLERANCE_DEGREES):
    """
    diffs: list of tuples (idx, ref_angle, user_angle, diff)
    weights: list of floats same length; default uses DEFAULT_WEIGHTS
    returns normalized score 0..100
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    total_weight = 0.0
    acc = 0.0
    for i, (idx, ra, ua, diff) in enumerate(diffs):
        w = weights[i] if i < len(weights) else 1.0
        total_weight += w
        if diff is None:
            # if missing, penalize moderately by assuming max diff
            normalized = 0.0
        else:
            clipped = min(diff, max_tolerance)
            normalized = max(0.0, 1.0 - (clipped / max_tolerance))  # 1 when diff=0, 0 when diff>=max_tolerance
        acc += w * normalized
    if total_weight == 0:
        return 0.0
    score = (acc / total_weight) * 100.0
    return float(score)

# Convenience function to produce final score in one call
def similarity_score(user_landmarks, ref_landmarks, weights=None, max_tolerance=MAX_ANGLE_TOLERANCE_DEGREES):
    diffs = compute_angle_differences(user_landmarks, ref_landmarks)
    return score_from_diffs(diffs, weights=weights, max_tolerance=max_tolerance)
