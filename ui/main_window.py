from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QLabel
from PySide6.QtGui import QFont

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Karate Trainer MVP")
        self.setGeometry(100, 100, 1280, 720)  # x, y, width, height

        # Central widget
        central_widget = QWidget()
        layout = QVBoxLayout()

        # Temporary placeholder
        from PySide6.QtWidgets import QMainWindow, QWidget, QVBoxLayout
from ui.pose_canvas import PoseCanvas

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Karate Trainer")

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)

        self.pose_canvas = PoseCanvas()
        layout.addWidget(self.pose_canvas)

        self.setCentralWidget(central_widget)


        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
