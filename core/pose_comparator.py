# core/pose_comparator.py
import math

class PoseComparator:
    def __init__(self):
        pass

    def compare(self, landmarks1, landmarks2):
        """
        Compare two sets of pose landmarks and return a similarity score (0–100).
        landmarks1 and landmarks2 must each have a `.landmark` list.
        """
        if not landmarks1 or not landmarks2:
            return 0.0

        total_diff = 0.0
        count = 0

        for lm1, lm2 in zip(landmarks1.landmark, landmarks2.landmark):
            dx = lm1.x - lm2.x
            dy = lm1.y - lm2.y
            dz = lm1.z - lm2.z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            total_diff += dist
            count += 1

        avg_diff = total_diff / count if count else 0
        # The multiplier (5) controls sensitivity — tweak for tighter/looser scoring
        similarity = max(0.0, min(100.0, 100 * (1 - avg_diff * 5)))
        return similarity
