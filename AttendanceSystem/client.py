from ultralytics import YOLO
import cv2
import torch
import face_recognition
import pickle
import requests
import time
import os

# Load the YOLOv10x model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov10s.pt').to(device)  # Load model to GPU if available

# Load known face encodings and names from the pickle file
ENCODINGS_FILE = 'encodings.pickle'

if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, 'rb') as f:
        known_encodings, known_names = pickle.load(f)  # Unpack the tuple correctly
else:
    print(f"{ENCODINGS_FILE} not found. Please run the train_faces.py script to create it.")
    exit()

# Open a connection to the webcam for recognition
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame.")
        break

    # Run YOLO face detection
    results = model(frame)

    # Check if results are returned correctly
    if results:
        result = results[0]  # Get the first output
        if hasattr(result, 'boxes'):
            boxes = result.boxes.data.numpy()  # Get the bounding box data
        else:
            print("No boxes detected.")
            boxes = []  # Set boxes to an empty list
    else:
        print("No results returned from YOLO model.")
        boxes = []  # Ensure boxes is empty

    # Iterate through each detected box
    for box in boxes:
        x1, y1, x2, y2 = map(int, box[:4])
        face_frame = frame[y1:y2, x1:x2]
        rgb_face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(rgb_face_frame)

        if face_encodings:
            face_encoding = face_encodings[0]
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                match_index = matches.index(True)
                name = known_names[match_index]

            # Annotate the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Send attendance data to the server
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            attendance_payload = {'student_name': name, 'timestamp': timestamp}
            try:
                response = requests.post('http://127.0.0.1:5000/attendance', json=attendance_payload)
                if response.status_code == 200:
                    print(response.json()['message'])
            except Exception as e:
                print(f"Error sending attendance data: {e}")

    # Display the annotated frame
    cv2.imshow('Attendance System', frame)

    # Check if the window is being updated correctly
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
