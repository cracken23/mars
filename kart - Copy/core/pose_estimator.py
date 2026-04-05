import cv2
import mediapipe as mp

class PoseEstimator:
    def __init__(self, detection_confidence=0.5, tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def process_frame(self, frame, draw=True):
        """
        Processes a BGR frame, returns annotated frame and pose landmarks.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)

        if draw and results.pose_landmarks:
            self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)

        return frame, results.pose_landmarks
    

    def release(self):
        self.pose.close()
    def draw_landmarks(self, frame, landmarks):
        self.mp_drawing.draw_landmarks(
            frame,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
            self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
        )
        return frame
if __name__ == "__main__":
    # Quick standalone test
    cap = cv2.VideoCapture(0)
    estimator = PoseEstimator()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot access webcam")
            break

        annotated_frame, landmarks = estimator.process_frame(frame, draw=True)
        cv2.imshow("Pose Estimator Test", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    estimator.release()
