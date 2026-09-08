#!/usr/bin/env python3
"""
Generate improved synthetic face images for training.
Creates more realistic-looking synthetic faces with varied features.
"""

import os
import cv2
import numpy as np


DATA_DIR = "data"
MASK_DIR = os.path.join(DATA_DIR, "mask")
NO_MASK_DIR = os.path.join(DATA_DIR, "no_mask")
NUM_SAMPLES = 300  # More samples for better training


def add_noise(img, amount=10):
    """Add random noise to image."""
    noise = np.random.randint(-amount, amount + 1, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def add_blur(img, probability=0.3):
    """Apply random blur."""
    if np.random.random() < probability:
        ksize = np.random.choice([3, 5])
        return cv2.GaussianBlur(img, (ksize, ksize), 0)
    return img


def create_realistic_face(variation_seed, with_mask=True):
    """Create a more realistic-looking synthetic face."""
    np.random.seed(variation_seed)

    img = np.ones((128, 128, 3), dtype=np.uint8) * 220

    # Random skin tone variations
    skin_base = np.random.randint(180, 240)
    skin_tone = (skin_base, skin_base - 30, skin_base - 60)

    # Face oval
    face_w = 45 + np.random.randint(-5, 5)
    face_h = 55 + np.random.randint(-5, 5)
    center = (64 + np.random.randint(-3, 3), 64 + np.random.randint(-3, 3))

    # Draw face with gradient
    cv2.ellipse(img, center, (face_w, face_h), 0, 0, 360, skin_tone, -1)

    # Add face shading
    overlay = img.copy()
    cv2.ellipse(overlay, center, (face_w - 5, face_h - 5), 0, 0, 360,
                 (min(255, skin_tone[0] + 15), min(255, skin_tone[1] + 15), min(255, skin_tone[2] + 15)), -1)
    img = cv2.addWeighted(img, 0.8, overlay, 0.2, 0)

    # Hair
    hair_color = (np.random.randint(20, 80), np.random.randint(10, 50), np.random.randint(5, 30))
    cv2.ellipse(img, (center[0], center[1] - 35), (face_w + 5, 25), 0, 0, 180, hair_color, -1)

    # Eyes
    eye_y = center[1] - 10
    eye_spacing = 18
    eye_size = 5 + np.random.randint(-2, 2)

    # Eye whites
    cv2.ellipse(img, (center[0] - eye_spacing, eye_y), (eye_size + 2, eye_size), 0, 0, 360, (240, 240, 240), -1)
    cv2.ellipse(img, (center[0] + eye_spacing, eye_y), (eye_size + 2, eye_size), 0, 0, 360, (240, 240, 240), -1)

    # Irises
    iris_color = (np.random.randint(50, 150), np.random.randint(80, 180), np.random.randint(100, 200))
    cv2.circle(img, (center[0] - eye_spacing, eye_y), eye_size - 1, iris_color, -1)
    cv2.circle(img, (center[0] + eye_spacing, eye_y), eye_size - 1, iris_color, -1)

    # Pupils
    cv2.circle(img, (center[0] - eye_spacing, eye_y), 2, (20, 20, 20), -1)
    cv2.circle(img, (center[0] + eye_spacing, eye_y), 2, (20, 20, 20), -1)

    # Eye highlights
    cv2.circle(img, (center[0] - eye_spacing - 1, eye_y - 1), 1, (255, 255, 255), -1)
    cv2.circle(img, (center[0] + eye_spacing - 1, eye_y - 1), 1, (255, 255, 255), -1)

    # Eyebrows
    brow_color = (skin_tone[0] - 40, skin_tone[1] - 30, skin_tone[2] - 20)
    brow_thickness = 2 + np.random.randint(0, 2)
    cv2.line(img, (center[0] - eye_spacing - 6, eye_y - 12), (center[0] - eye_spacing + 6, eye_y - 10), brow_color, brow_thickness)
    cv2.line(img, (center[0] + eye_spacing - 6, eye_y - 10), (center[0] + eye_spacing + 6, eye_y - 12), brow_color, brow_thickness)

    # Nose
    nose_x = center[0]
    nose_y1 = eye_y + 8
    nose_y2 = nose_y1 + 18
    cv2.line(img, (nose_x, nose_y1), (nose_x - 3, nose_y2), (skin_tone[0] - 20, skin_tone[1] - 20, skin_tone[2] - 10), 2)
    cv2.line(img, (nose_x, nose_y1), (nose_x + 3, nose_y2), (skin_tone[0] - 20, skin_tone[1] - 20, skin_tone[2] - 10), 2)

    # Mouth area
    mouth_y = center[1] + 28

    if with_mask:
        # Medical mask - blue/white
        mask_colors = [
            (200, 220, 255),   # Light blue
            (180, 200, 230),   # Blue
            (255, 255, 255),   # White
            (220, 230, 245),   # Pale blue
        ]
        mask_color = mask_colors[np.random.randint(0, len(mask_colors))]

        # Mask body (covers mouth and chin)
        mask_y1 = mouth_y - 8
        mask_y2 = mouth_y + 20
        mask_x1 = center[0] - 30 - np.random.randint(-2, 2)
        mask_x2 = center[0] + 30 + np.random.randint(-2, 2)

        cv2.rectangle(img, (mask_x1, mask_y1), (mask_x2, mask_y2), mask_color, -1)

        # Mask shading
        cv2.rectangle(img, (mask_x1, mask_y1), (mask_x2, mask_y2),
                     (mask_color[0] - 30, mask_color[1] - 30, mask_color[2] - 20), 2)

        # Mask pleats
        for i in range(4):
            y = mask_y1 + 5 + i * 6
            cv2.line(img, (mask_x1 + 5, y), (mask_x2 - 5, y),
                    (mask_color[0] - 20, mask_color[1] - 20, mask_color[2] - 10), 1)

        # Ear loops
        cv2.line(img, (mask_x1, mask_y1 + 5), (mask_x1 - 12, mask_y1 - 5), (30, 30, 30), 2)
        cv2.line(img, (mask_x2, mask_y1 + 5), (mask_x2 + 12, mask_y1 - 5), (30, 30, 30), 2)

        # Nose clip (top of mask)
        cv2.line(img, (center[0] - 15, mask_y1), (center[0] + 15, mask_y1),
                (mask_color[0] - 15, mask_color[1] - 15, mask_color[2] - 5), 2)

        # Some variation - mask folds
        if np.random.random() > 0.5:
            cv2.line(img, (center[0] + 10, mask_y1), (center[0] + 15, mask_y2),
                    (mask_color[0] - 25, mask_color[1] - 25, mask_color[2] - 15), 1)

    else:
        # No mask - visible mouth with variations
        mouth_open = np.random.random() > 0.6

        if mouth_open:
            # Open mouth
            cv2.ellipse(img, (center[0], mouth_y), (12, 6 + np.random.randint(0, 4)), 0, 0, 180,
                       (80, 40, 40), -1)
            cv2.ellipse(img, (center[0], mouth_y - 2), (10, 4), 0, 0, 180,
                       (180, 80, 80), -1)  # Tongue
        else:
            # Closed mouth / slight smile
            lip_color = (150 + np.random.randint(-20, 20),
                        60 + np.random.randint(-15, 15),
                        60 + np.random.randint(-15, 15))
            cv2.ellipse(img, (center[0], mouth_y), (14, 4 + np.random.randint(0, 3)), 0, 0, 180,
                       lip_color, -1)
            # Upper lip line
            cv2.line(img, (center[0] - 12, mouth_y), (center[0] + 12, mouth_y),
                    (lip_color[0] - 20, lip_color[1] - 15, lip_color[2] - 15), 1)

        # Maybe add facial hair for some no-mask images
        if np.random.random() > 0.75:
            beard_color = (40 + np.random.randint(-10, 10),
                         25 + np.random.randint(-10, 10),
                         15 + np.random.randint(-5, 5))
            cv2.ellipse(img, (center[0], mouth_y + 12), (20, 12), 0, 0, 180,
                       beard_color, -1)

    # Add overall variations
    img = add_noise(img, 8)
    img = add_blur(img, 0.2)

    # Random brightness
    brightness = np.random.randint(-20, 20)
    img = np.clip(img.astype(np.int16) + brightness, 0, 255).astype(np.uint8)

    # Random contrast
    contrast = 0.9 + np.random.random() * 0.3
    img = np.clip(((img.astype(np.float32) - 128) * contrast + 128), 0, 255).astype(np.uint8)

    return img


def generate_dataset():
    """Generate synthetic training data."""
    os.makedirs(MASK_DIR, exist_ok=True)
    os.makedirs(NO_MASK_DIR, exist_ok=True)

    print("=" * 50)
    print("Generating Improved Synthetic Training Data")
    print("=" * 50)
    print(f"Creating {NUM_SAMPLES} images per class...")
    print()

    # Generate mask images
    print("Generating MASK images...")
    for i in range(NUM_SAMPLES):
        variation_seed = 1000 + i
        img = create_realistic_face(variation_seed, with_mask=True)

        # Random transforms
        if np.random.random() > 0.4:
            img = cv2.flip(img, 1)
        if np.random.random() > 0.5:
            angle = np.random.randint(-12, 12)
            M = cv2.getRotationMatrix2D((64, 64), angle, 1.0)
            img = cv2.warpAffine(img, M, (128, 128))
        if np.random.random() > 0.7:
            shift_x = np.random.randint(-5, 5)
            shift_y = np.random.randint(-5, 5)
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            img = cv2.warpAffine(img, M, (128, 128))

        filename = os.path.join(MASK_DIR, f"mask_{i+1}.jpg")
        cv2.imwrite(filename, img)

        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{NUM_SAMPLES} mask images")

    # Generate no mask images
    print("Generating NO MASK images...")
    for i in range(NUM_SAMPLES):
        variation_seed = 2000 + i
        img = create_realistic_face(variation_seed, with_mask=False)

        # Random transforms
        if np.random.random() > 0.4:
            img = cv2.flip(img, 1)
        if np.random.random() > 0.5:
            angle = np.random.randint(-12, 12)
            M = cv2.getRotationMatrix2D((64, 64), angle, 1.0)
            img = cv2.warpAffine(img, M, (128, 128))
        if np.random.random() > 0.7:
            shift_x = np.random.randint(-5, 5)
            shift_y = np.random.randint(-5, 5)
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            img = cv2.warpAffine(img, M, (128, 128))

        filename = os.path.join(NO_MASK_DIR, f"nomask_{i+1}.jpg")
        cv2.imwrite(filename, img)

        if (i + 1) % 50 == 0:
            print(f"  Generated {i + 1}/{NUM_SAMPLES} no-mask images")

    print()
    print("=" * 50)
    print("Dataset generation complete!")
    print(f"Total MASK images: {NUM_SAMPLES}")
    print(f"Total NO MASK images: {NUM_SAMPLES}")
    print("=" * 50)


if __name__ == "__main__":
    generate_dataset()