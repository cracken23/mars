# core/pose_loader.py
import json
from types import SimpleNamespace

def load_reference_pose(path):
    """
    Loads reference pose from JSON and returns a Mediapipe-like object
    with `.landmark` attribute containing objects with x, y, z.
    """
    with open(path, "r") as f:
        data = json.load(f)

    landmarks = []
    for lm in data:
        landmarks.append(SimpleNamespace(
            x=lm.get('x', 0.0),
            y=lm.get('y', 0.0),
            z=lm.get('z', 0.0),
            visibility=lm.get('visibility', 0.0)
        ))

    return SimpleNamespace(landmark=landmarks)
