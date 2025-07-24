# Used to detect faces in a frame using YOLOv10 model
import torch
import numpy as np

# Load the YOLOv10 model once (only at the start)
def load_yolo_model():
    # Load the PyTorch YOLOv10 model from the .pt file
    model = torch.load("yolov10s.pt")
    model.eval()  # Set the model to evaluation mode for inference
    return model

# Model loaded only once globally for efficiency
model = load_yolo_model()

def detect_faces(frame):
    # Convert OpenCV frame (BGR) to RGB and then to tensor
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_tensor = torch.from_numpy(rgb_frame).float().permute(2, 0, 1).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        results = model(input_tensor)

    # Assuming results contain bounding boxes and class labels
    # Extracting bounding box data from the results
    faces = []  # Placeholder for detected faces
    for detection in results:
        # Extract bounding box coordinates (x, y, width, height)
        x, y, w, h = detection['bbox']
        faces.append((x, y, w, h))
    
    return faces
