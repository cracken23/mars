#!/usr/bin/env python3
"""Test camera with different settings."""
import cv2
import time

print("Testing camera devices...")

for idx in [0, 2]:
    print(f"\nTrying /dev/video{idx}...")
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print(f"  Cannot open")
        continue
    
    # Try setting lower resolution for better USB bandwidth
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    # Try multiple reads (sometimes first frames fail)
    for attempt in range(5):
        ret, frame = cap.read()
        if ret:
            channels = frame.shape[2] if len(frame.shape) == 3 else 1
            print(f"  SUCCESS on attempt {attempt+1}: {frame.shape[1]}x{frame.shape[0]}, {channels}ch")
            break
        print(f"  Attempt {attempt+1} failed, retrying...")
        time.sleep(0.5)
    else:
        print(f"  All attempts failed")
    
    cap.release()
