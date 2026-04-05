# save_reference.py (run in project root)
import cv2, json
from core.pose_estimator import PoseEstimator

cap = cv2.VideoCapture(0)
est = PoseEstimator()
print("Press SPACE to capture a reference pose (q to quit)")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    annotated, lm = est.process_frame(frame, draw=True)
    cv2.imshow("Capture", annotated)
    k = cv2.waitKey(1) & 0xFF
    if k == ord(' '):  # space -> save
        # lm is list of dicts with x,y,z,visibility
        json.dump(lm, open("karate_trainer/data/ref_capture.json","w"), indent=2)
        print("Saved to karate_trainer/data/ref_capture.json")
    if k == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
est.release()
