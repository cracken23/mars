# save_reference_pose.py
from core.pose_estimator import PoseEstimator
import cv2
import json

cap = cv2.VideoCapture(0)
pose_estimator = PoseEstimator()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get both the annotated frame and the landmarks
    annotated_frame, landmarks = pose_estimator.process_frame(frame, draw=True)

    cv2.imshow("Capture Reference Pose", annotated_frame)

    # Press 's' to save
    if cv2.waitKey(1) & 0xFF == ord('s'):
        if landmarks:
            # Extract (x, y, z, visibility) for each landmark
            data = [
                {
                    "x": lm.x,
                    "y": lm.y,
                    "z": lm.z,
                    "visibility": lm.visibility
                }
                for lm in landmarks.landmark
            ]

            with open("reference_pose.json", "w") as f:
                json.dump(data, f, indent=2)

            print("Reference pose saved!")
            break

cap.release()
cv2.destroyAllWindows()
pose_estimator.release()
