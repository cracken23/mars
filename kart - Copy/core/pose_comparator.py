import numpy as np


class PoseComparator:

    def normalize_landmarks(self, landmarks):
        """
        Convert MediaPipe landmarks or dicts into a consistent list of dicts.
        """
        normalized = []

        if hasattr(landmarks, "landmark"):  # MediaPipe object
            for lm in landmarks.landmark:
                normalized.append({
                    "x": float(lm.x),
                    "y": float(lm.y),
                    "z": float(lm.z),
                    "visibility": float(lm.visibility)
                })

        elif isinstance(landmarks, list):  # Already list
            for lm in landmarks:
                if isinstance(lm, dict) and "x" in lm and "y" in lm:
                    normalized.append({
                        "x": float(lm["x"]),
                        "y": float(lm["y"])
                    })

        return normalized

    # ================= ANGLE METHOD =================
    def angle_similarity(self, user, ref):
        """
        Simple angle-based similarity (fallback method)
        """
        if len(user) != len(ref):
            return 0

        total_diff = 0
        for u, r in zip(user, ref):
            dx = u["x"] - r["x"]
            dy = u["y"] - r["y"]
            total_diff += abs(dx) + abs(dy)

        score = max(0, 100 - total_diff * 100)
        return score

    # ================= EUCLIDEAN =================
    def euclidean_similarity(self, user, ref):
        if len(user) != len(ref):
            return 0

        dist = 0
        for u, r in zip(user, ref):
            dx = u["x"] - r["x"]
            dy = u["y"] - r["y"]
            dist += (dx ** 2 + dy ** 2)

        score = max(0, 100 - dist * 100)
        return score

    # ================= COSINE =================
    def cosine_similarity(self, user, ref):
        try:
            u_vec = np.array([v for lm in user for v in (lm["x"], lm["y"])])
            r_vec = np.array([v for lm in ref for v in (lm["x"], lm["y"])])

            dot = np.dot(u_vec, r_vec)
            norm = np.linalg.norm(u_vec) * np.linalg.norm(r_vec)

            if norm == 0:
                return 0

            return (dot / norm) * 100
        except:
            return 0

    # ================= PROCRUSTES =================
    def procrustes_similarity(self, user, ref):
        try:
            u = np.array([[lm["x"], lm["y"]] for lm in user])
            r = np.array([[lm["x"], lm["y"]] for lm in ref])

            # normalize (center)
            u -= u.mean(axis=0)
            r -= r.mean(axis=0)

            # scale
            u /= np.linalg.norm(u)
            r /= np.linalg.norm(r)

            # similarity
            diff = np.linalg.norm(u - r)
            score = max(0, 100 - diff * 100)

            return score
        except:
            return 0

    # ================= MAIN FUNCTION =================
    def compare_pose(self, user_landmarks, ref_landmarks, method="angle"):
        """
        Main entry point used by UI + server
        """
        user = self.normalize_landmarks(user_landmarks)
        ref = self.normalize_landmarks(ref_landmarks)

        if not user or not ref:
            return 0

        method = method.lower()

        if method == "angle":
            return self.angle_similarity(user, ref)

        elif method == "euclidean":
            return self.euclidean_similarity(user, ref)

        elif method == "cosine":
            return self.cosine_similarity(user, ref)

        elif method == "procrustes":
            return self.procrustes_similarity(user, ref)

        return self.angle_similarity(user, ref)