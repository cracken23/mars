# Karate Trainer 🥋

A real-time pose training application that helps users learn and practice karate poses using computer vision and machine learning.

## Features

- **Real-time pose detection** using MediaPipe
- **Pose comparison** with reference poses using angle-based matching
- **Visual feedback** with skeleton overlay and score display
- **Multiple training levels** with different karate techniques (kicks, punches, defense stances)
- **Adaptive scoring** with configurable thresholds

## Requirements

- Python 3.10+
- Webcam
- Linux/Windows/macOS (tested on Ubuntu via WSL)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd mars
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Application

```bash
python app.py
```

### Controls

- **Next/Prev buttons**: Navigate between training levels
- **Level dropdown**: Select a specific pose to practice
- **Q key**: Quit the application (in standalone test modes)

### Capturing New Reference Poses

Use the capture script to record new reference poses:

```bash
python scripts/capture_reference_poses.py
```

Press **SPACE** to capture a pose, **Q** to quit.

## Project Structure

```
mars/
├── app.py                 # Main application entry point
├── config.py              # Configuration settings
├── requirements.txt       # Python dependencies
│
├── core/                  # Core logic modules
│   ├── __init__.py
│   ├── pose_estimator.py  # MediaPipe pose detection wrapper
│   ├── pose_matcher.py    # Angle-based pose comparison
│   ├── pose_loader.py     # JSON pose file loader
│   └── feedback_engine.py # Human-readable feedback generation
│
├── ui/                    # PySide6 UI components
│   ├── __init__.py
│   ├── main_window.py     # Main application window
│   ├── pose_canvas.py     # Webcam display + pose overlay widget
│   └── feedback_panel.py  # Score and feedback display widget
│
├── data/                  # Data files
│   └── levels.json        # Training levels with reference poses
│
└── scripts/               # Utility scripts
    ├── capture_reference_poses.py  # Record new poses
    └── save_reference_pose.py      # Quick single pose capture
```

## How It Works

### Pose Detection
The app uses [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html) to detect 33 body landmarks from the webcam feed in real-time.

### Pose Matching
Instead of comparing raw landmark positions (which vary with distance and camera angle), the app calculates **joint angles** at key body parts:

- Elbow angles (arm flexion)
- Shoulder angles (arm position relative to torso)
- Knee angles (leg flexion)
- Hip angles (leg position)
- Torso rotation

This makes the comparison robust to:
- User's distance from the camera
- User's position in the frame
- Body size differences

### Scoring
Each angle difference is compared against a tolerance threshold:
- **0° difference** → 100% match for that joint
- **≥60° difference** → 0% match for that joint

The final score is a weighted average of all joint scores.

## Configuration

Edit `config.py` to customize:

```python
# Camera settings
camera.device_index = 0  # Change for different webcam

# Pose detection
pose_estimation.min_detection_confidence = 0.5
pose_estimation.min_tracking_confidence = 0.5

# Scoring
pose_matching.max_angle_tolerance_degrees = 60.0
pose_matching.score_smoothing_window = 5

# UI
ui.window_width = 1280
ui.window_height = 720
```

Or use environment variables:
```bash
export KARATE_CAMERA_DEVICE=1
export KARATE_DETECTION_CONFIDENCE=0.7
python app.py
```

## Adding New Training Levels

1. Edit `data/levels.json` or use the capture script
2. Each level needs:
   - `id`: Unique identifier
   - `name`: Display name
   - `type`: Category (kick, punch, defense)
   - `threshold`: Passing score (0-100)
   - `reference_pose`: Array of 33 landmarks with x, y, z, visibility

Example:
```json
{
  "id": 7,
  "name": "Roundhouse Kick",
  "type": "kick",
  "threshold": 65,
  "reference_pose": [...]
}
```

## Troubleshooting

### Webcam not detected
- Check if another application is using the webcam
- Try a different device index in `config.py`
- On Linux, ensure you have permissions: `sudo usermod -a -G video $USER`

### Low FPS / Lag
- Reduce window size in config
- Ensure good lighting for faster pose detection
- Close other resource-intensive applications

### Pose not detected
- Ensure your full body is visible in the frame
- Stand further from the camera
- Improve lighting conditions

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests (when available)
5. Submit a pull request
