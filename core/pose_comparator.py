def normalize_landmarks(self, landmarks):
    """
    Convert MediaPipe landmarks or dicts into a consistent list of dicts.
    """
    normalized = []

    if hasattr(landmarks, "landmark"):  # MediaPipe object
        for lm in landmarks.landmark:
            normalized.append({
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            })
    elif isinstance(landmarks, list):  # Already list of dicts
        for lm in landmarks:
            if isinstance(lm, dict):
                normalized.append(lm)
    return normalized
