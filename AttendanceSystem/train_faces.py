# Used to train the face recognition model with new images

import face_recognition
import os
import pickle

# Path to the directory containing known faces
KNOWN_FACES_DIR = 'dataset'
ENCODINGS_FILE = 'encodings.pickle'

def load_known_faces():
    """Load existing face encodings from the pickle file if it exists."""
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, 'rb') as f:
            return pickle.load(f)
    return [], []

def save_face_encodings(known_face_encodings, known_face_names):
    """Save the face encodings and names to a pickle file."""
    with open(ENCODINGS_FILE, 'wb') as f:
        pickle.dump((known_face_encodings, known_face_names), f)

def train_faces():
    known_face_encodings, known_face_names = load_known_faces()

    # Loop through each folder in the known_faces_dir
    for student_folder in os.listdir(KNOWN_FACES_DIR):
        folder_path = os.path.join(KNOWN_FACES_DIR, student_folder)

        # Ensure it's a directory
        if os.path.isdir(folder_path):
            # Check if the student name already exists
            if student_folder in known_face_names:
                continue  # Skip this folder if already processed
            
            # Load each image in the folder
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    img_path = os.path.join(folder_path, filename)
                    try:
                        image = face_recognition.load_image_file(img_path)
                        encodings = face_recognition.face_encodings(image)

                        if encodings:  # Check if encoding was found
                            known_face_encodings.append(encodings[0])
                            known_face_names.append(student_folder)
                        else:
                            print(f"No faces found in image: {img_path}")
                    except Exception as e:
                        print(f"Error processing image {img_path}: {e}")

    # Save the updated encodings to the pickle file
    save_face_encodings(known_face_encodings, known_face_names)

if __name__ == '__main__':
    train_faces()
