#!/usr/bin/env python3
"""
Data Collection Script for Face Mask Detection
Capture face images and save them for training.
"""

import os
import cv2
import numpy as np
from face_mask_detection import FaceDetector

# Configuration
DATA_DIR = "data"
MASK_DIR = os.path.join(DATA_DIR, "mask")
NO_MASK_DIR = os.path.join(DATA_DIR, "no_mask")
SAMPLES_PER_CLASS = 100

# Create directories
os.makedirs(MASK_DIR, exist_ok=True)
os.makedirs(NO_MASK_DIR, exist_ok=True)


def count_images(directory):
    """Count existing images in directory."""
    if os.path.exists(directory):
        return len([f for f in os.listdir(directory) if f.endswith(('.jpg', '.png', '.jpeg'))])
    return 0


def capture_and_save():
    """Capture face images from webcam."""
    print("=" * 50)
    print("Face Mask Detection - Data Collection")
    print("=" * 50)
    print()

    # Check existing counts
    mask_count = count_images(MASK_DIR)
    no_mask_count = count_images(NO_MASK_DIR)
    print(f"Current dataset: {mask_count} mask images, {no_mask_count} no-mask images")
    print()

    # Initialize webcam and face detector
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam!")
        return

    face_detector = FaceDetector()

    print("Instructions:")
    print("  1. Press 'M' - Capture a WITH MASK image")
    print("  2. Press 'N' - Capture a WITHOUT MASK image")
    print("  3. Press 'Q' - Quit and save")
    print()

    saved_mask = 0
    saved_no_mask = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Detect faces
        faces = face_detector.detect_faces(frame)

        # Draw face rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Display info
        cv2.putText(frame, f"MASK images: {mask_count + saved_mask}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"NO MASK images: {no_mask_count + saved_no_mask}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "M=MASK  N=NO MASK  Q=Quit",
                    (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if len(faces) > 0:
            cv2.putText(frame, f"Faces detected: {len(faces)}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("Data Collection - Face Mask Detection", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('m') and len(faces) > 0:
            # Save mask image
            x, y, w, h = faces[0]
            pad = int(w * 0.1)
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
            face_img = frame[y1:y2, x1:x2]

            filename = os.path.join(MASK_DIR, f"mask_{mask_count + saved_mask + 1}.jpg")
            cv2.imwrite(filename, face_img)
            saved_mask += 1
            print(f"  Saved MASK image: {filename}")

        elif key == ord('n') and len(faces) > 0:
            # Save no mask image
            x, y, w, h = faces[0]
            pad = int(w * 0.1)
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(frame.shape[1], x + w + pad), min(frame.shape[0], y + h + pad)
            face_img = frame[y1:y2, x1:x2]

            filename = os.path.join(NO_MASK_DIR, f"nomask_{no_mask_count + saved_no_mask + 1}.jpg")
            cv2.imwrite(filename, face_img)
            saved_no_mask += 1
            print(f"  Saved NO MASK image: {filename}")

    cap.release()
    cv2.destroyAllWindows()

    print()
    print("=" * 50)
    print(f"Data collection complete!")
    print(f"Saved {saved_mask} MASK images")
    print(f"Saved {saved_no_mask} NO MASK images")
    print(f"Total MASK: {mask_count + saved_mask}")
    print(f"Total NO MASK: {no_mask_count + saved_no_mask}")
    print("=" * 50)


if __name__ == "__main__":
    capture_and_save()