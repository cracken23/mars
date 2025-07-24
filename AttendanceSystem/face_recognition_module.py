#Used by client.py to recognize faces in images

import face_recognition
import pickle
import os

# Path to the pickle file containing known face encodings
ENCODINGS_FILE = "encodings.pickle"

def load_known_faces():
    """Load known face encodings and names from the pickle file."""
    if not os.path.exists(ENCODINGS_FILE):
        print(f"{ENCODINGS_FILE} not found. Returning empty lists.")
        return [], []  # Return empty lists if the file does not exist

    try:
        with open(ENCODINGS_FILE, 'rb') as f:
            known_face_encodings, known_face_names = pickle.load(f)
        return known_face_encodings, known_face_names
    except Exception as e:
        print(f"Error loading {ENCODINGS_FILE}: {e}")
        return [], []

def recognize_face(face_img):
    """Recognize a face in the provided image."""
    known_face_encodings, known_face_names = load_known_faces()

    face_encodings = face_recognition.face_encodings(face_img)

    if face_encodings:
        # Compare the detected face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encodings[0])
        if True in matches:
            match_index = matches.index(True)
            return known_face_names[match_index]

    return None
