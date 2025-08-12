import cv2
from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from core.pose_estimator import PoseEstimator

class PoseCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # UI
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        self.setLayout(layout)

        # Pose Estimator
        self.pose_estimator = PoseEstimator()

        # Webcam
        self.cap = cv2.VideoCapture(0)

        # Timer to update frames
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # ~33 FPS

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Process frame with pose detection
        annotated_frame, _ = self.pose_estimator.process_frame(frame, draw=True)

        # Convert frame to QImage for QLabel
        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event):
        """Cleanup when window closes."""
        self.timer.stop()
        self.cap.release()
        self.pose_estimator.release()
        super().closeEvent(event)
