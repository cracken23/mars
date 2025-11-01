
import cv2
from core.pose_estimator import PoseEstimator
from core.pose_loader import load_reference_pose
from core.pose_comparator import PoseComparator


pose_estimator = PoseEstimator()
reference_landmarks = load_reference_pose("reference_pose.json")
comparator = PoseComparator()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get processed frame and landmarks
    frame, landmarks = pose_estimator.process_frame(frame, draw=True)

    # Compare only if landmarks are detected
    if landmarks:
        score = comparator.compare(landmarks, reference_landmarks)
        cv2.putText(frame, f"Similarity: {score:.2f}%", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Pose Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pose_estimator.release()
print(type(landmarks), landmarks)
print(type(reference_landmarks), reference_landmarks)
