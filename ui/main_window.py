from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from ui.pose_canvas import PoseCanvas


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Karate Trainer")
        self.setGeometry(100, 100, 1280, 720)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        self.pose_canvas = PoseCanvas()
        layout.addWidget(self.pose_canvas)

        self.setCentralWidget(central_widget)
