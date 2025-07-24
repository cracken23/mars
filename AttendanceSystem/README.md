
# Attendance Marking System

## Overview

The Attendance Marking System is a facial recognition application that automates attendance tracking for students. It uses computer vision and deep learning techniques to identify students in real-time and record their attendance.


## Features

- Real-time facial recognition using YOLO and `face_recognition` libraries.
- Automatic attendance marking with timestamps.
- Web interface for displaying attendance records.
- SQLite database for persistent storage of attendance data.
- Easy setup and configuration.

## Getting Started

### Prerequisites

- Python 3.x
- Git
- Required Python packages (see `requirements.txt`)


## Installation

1. Clone the repository

```bash
git clone https://github.com/cracken23/miniproject2A
```
2. Navigate to the project directory:
```bash
cd AttendanceSystem
```
3. Install dependencies
```bash
pip install -r requirements.txt
```


### Setting Up the Database

 Run the database setup script to create the `attendance.db` file:
   ```bash
   python setup_database.py
```


### Training the Model

1. Prepare your dataset of student images in the `dataset` folder.
2. Run the training script to generate face encodings:
   ```bash
   python train_faces.py
   ```

### Running the Server

1. Start the Flask server to handle attendance data:
   ```bash
   python server/server.py
    ```
2. The server will be accessible at :
   ```arduino
   http://127.0.0.1:5000
   ```

### Running the Client

1. Start the client application:
   ```bash
   python client.py
   ```
2. Open your web browser and navigate to:
    ```bash
    http://127.0.0.1:5000
    ```



   



## Usage

Once the server is running, you can start the client. The webcam will activate, and the application will start recognizing faces. When a student's face is detected, their attendance will be recorded in the database, and the web interface will display the attendance records.



## License

[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgments

- [YOLO](https://github.com/AlexeyAB/darknet) for object detection.
- [face_recognition](https://github.com/ageitgey/face_recognition) for facial recognition.
- [Flask](https://flask.palletsprojects.com/) for the web framework.
