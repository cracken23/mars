"""
Pose loader module for loading reference poses from JSON files.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any


def load_reference_pose(path: str) -> SimpleNamespace:
    """
    Load a reference pose from a JSON file.
    
    Args:
        path: Path to the JSON file containing pose landmarks
        
    Returns:
        A SimpleNamespace object with a .landmark attribute containing
        the list of landmarks (each with x, y, z, visibility attributes)
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
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


def save_reference_pose(landmarks: list[dict[str, float]], path: str) -> None:
    """
    Save pose landmarks to a JSON file.
    
    Args:
        landmarks: List of landmark dictionaries with x, y, z, visibility keys
        path: Output file path
    """
    with open(path, "w") as f:
        json.dump(landmarks, f, indent=2)

