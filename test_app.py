#!/usr/bin/env python3
"""Test script to debug application startup."""
import sys
import traceback

print("Starting app...")

try:
    from PySide6.QtWidgets import QApplication
    print("PySide6 imported OK")
    
    from ui.main_window import MainWindow
    print("MainWindow imported OK")
    
    app = QApplication(sys.argv)
    print("QApplication created OK")
    
    window = MainWindow()
    print("MainWindow created OK")
    
    window.show()
    print("Window shown - entering event loop")
    
    sys.exit(app.exec())
    
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
