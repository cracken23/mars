import cv2
import json
from core.pose_estimator import PoseEstimator
from core.pose_comparator import PoseComparator

# Load reference pose
with open("reference_pose.json", "r") as f:
    reference_landmarks = json.load(f)
    
print("Current Landmarks:", current_landmarks[:2])  # show first 2
print("Reference Landmarks:", reference_landmarks[:2])


pose_estimator = PoseEstimator()
comparator = PoseComparator()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    annotated_frame, landmarks = pose_estimator.process_frame(frame)

    if landmarks:
        # Convert landmarks into a list of dicts (x, y, z, visibility)
        current_landmarks = [
            {
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility
            } for lm in landmarks.landmark
        ]

        # Compare similarity
        score = comparator.compare(current_landmarks, reference_landmarks)
        cv2.putText(
            annotated_frame,
            f"Similarity: {score:.2f}%",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow("Pose Match Test", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose_estimator.release()
