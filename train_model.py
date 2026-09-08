#!/usr/bin/env python3
"""Train the face mask detection model and save it.

Uses the real images in:
    data/mask/   (label 0)
    data/no_mask/  (label 1)

The actual class_indices from the data generator is printed
during training and saved to:
    models/class_indices.json

This JSON file is loaded at prediction time so we never
hard-code which output means MASK vs NO MASK.
"""

import os
import sys
import json
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, Dense, Conv2D, MaxPooling2D,
                                     Flatten, Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator


IMG_SIZE = (128, 128)
MODEL_PATH = "models/face_mask_model.keras"
CLASS_INDICES_PATH = "models/class_indices.json"


def build_model(input_shape=(128, 128, 3)):
    """Build the face mask detection CNN model with BatchNorm."""
    model = Sequential([
        # Block 1
        Input(shape=input_shape),
        Conv2D(32, (3, 3), padding='same'),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 2
        Conv2D(64, (3, 3), padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Block 3
        Conv2D(128, (3, 3), padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Classifier head
        Flatten(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def check_dataset(data_dir="data"):
    """Check that both class folders exist and have images."""
    mask_dir = os.path.join(data_dir, "mask")
    no_mask_dir = os.path.join(data_dir, "no_mask")

    def count(d):
        if not os.path.isdir(d):
            return 0
        return len([f for f in os.listdir(d)
                    if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

    mask_count = count(mask_dir)
    no_mask_count = count(no_mask_dir)
    has_data = mask_count > 0 and no_mask_count > 0
    return has_data, mask_count, no_mask_count


def train_and_save_model(model_path=MODEL_PATH,
                         class_indices_path=CLASS_INDICES_PATH,
                         data_dir="data", img_size=IMG_SIZE,
                         batch_size=16, epochs=30):
    """Train the face mask detection model and save it together with class_indices."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    has_data, mask_count, no_mask_count = check_dataset(data_dir)

    print("=" * 50)
    print("STEP 1: CHECK TRAINING DATA")
    print("=" * 50)
    print(f"MASK images: {mask_count}")
    print(f"NO MASK images: {no_mask_count}")
    print(f"Total: {mask_count + no_mask_count}")
    print("=" * 50)
    print()

    if not has_data:
        print("ERROR: No training data found (or one of the folders is empty).")
        print()
        print("Required folder structure:")
        print("  data/")
        print("    mask/     <-- MASK images (labelled 0)")
        print("    no_mask/  <-- NO MASK images (labelled 1)")
        print()
        print("To create real training data, run:")
        print("  python collect_data.py")
        sys.exit(1)

    # ---- Build the data generator so we can read the TRUE class_indices ----
    print("=" * 50)
    print("STEP 2: BUILD DATA GENERATOR")
    print("=" * 50)
    print("Image size:", img_size, "(used for BOTH training and prediction)")
    print("Rescaling: 1./255  (used for BOTH training and prediction)")
    print("Color format: BGR (OpenCV default) -> BGR input to model")
    print("Note: generator reads as BGR by default in tf.keras >= 2.18 unless")
    print("      color_mode='rgb' is set. We will set color_mode='rgb' so that")
    print("      training and prediction read images the same way.")
    print()

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb',
        subset='training',
        shuffle=True
    )

    val_gen = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        color_mode='rgb',
        subset='validation',
        shuffle=False
    )

    # Print the actual class_indices from the generator
    # This is the SOURCE OF TRUTH for class mapping.
    class_indices = train_gen.class_indices
    # class_indices looks like: {'mask': 0, 'no_mask': 1} (or the other way)
    inv_class_indices = {v: k for k, v in class_indices.items()}

    print()
    print("=" * 50)
    print("STEP 3: CLASS INDICES (from data generator)")
    print("=" * 50)
    print("class_indices (folder -> index):", class_indices)
    print("inverted     (index -> folder):", inv_class_indices)
    print()
    for folder_name, idx in class_indices.items():
        print(f"  output {idx:.1f} ~ 1.0  ==>  {folder_name.upper()}")
        print(f"  output {idx:.1f} ~ 0.0  ==>  {inv_class_indices[1 - idx].upper()}")
    print("=" * 50)
    print()

    # Save the mapping so the prediction code can use the same mapping
    with open(class_indices_path, 'w') as f:
        json.dump({
            "class_indices": class_indices,
            "inv_class_indices": inv_class_indices,
            "img_size": list(img_size),
            "rescale": 1.0 / 255.0,
            "color_mode": "rgb"
        }, f, indent=2)
    print(f"Saved class mapping to: {class_indices_path}")
    print()

    # ---- Build and train the model ----
    print("=" * 50)
    print("STEP 4: BUILD MODEL")
    print("=" * 50)
    model = build_model(input_shape=(*img_size, 3))
    model.summary()
    print("=" * 50)
    print()

    early_stop = EarlyStopping(monitor='val_accuracy', patience=5,
                               restore_best_weights=True, verbose=1)
    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy',
                                 save_best_only=True, verbose=1)

    print("=" * 50)
    print("STEP 5: TRAINING")
    print("=" * 50)
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    model.save(model_path)

    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])

    print()
    print("=" * 50)
    print("TRAINING COMPLETE")
    print("=" * 50)
    print(f"Model saved to: {model_path}")
    print(f"Class mapping saved to: {class_indices_path}")
    print(f"Final training accuracy:   {final_train_acc:.2%}")
    print(f"Final validation accuracy: {final_val_acc:.2%}")
    print(f"Best validation accuracy:  {best_val_acc:.2%}")
    print("=" * 50)
    print()
    print("Now run the application:")
    print("  python face_mask_detection.py")
    print()

    # Quick sanity check on the saved model
    print("=" * 50)
    print("STEP 6: SANITY CHECK ON SAVED MODEL")
    print("=" * 50)
    test_model_on_saved_data(model_path, class_indices, data_dir, img_size, n=5)
    return model


def test_model_on_saved_data(model_path, class_indices, data_dir, img_size, n=5):
    """Reload the saved model and verify it predicts correctly on training data."""
    from tensorflow.keras.models import load_model
    model = load_model(model_path)
    print(f"Reloaded model from: {model_path}")
    print(f"Using class_indices: {class_indices}")
    print()

    for folder_name, idx in class_indices.items():
        folder = os.path.join(data_dir, folder_name)
        files = sorted([f for f in os.listdir(folder)
                        if f.lower().endswith(('.jpg', '.png', '.jpeg'))])[:n]
        print(f"--- {folder_name} (expected index={idx}) ---")
        for f in files:
            img = cv2.imread(os.path.join(folder, f))
            if img is None:
                continue
            # Match training pipeline: BGR -> RGB, resize, /255
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            x = img.astype('float32') / 255.0
            x = np.expand_dims(x, 0)
            raw = model.predict(x, verbose=0)[0][0]
            predicted_idx = 1 if raw >= 0.5 else 0
            predicted_label = [k for k, v in class_indices.items() if v == predicted_idx][0]
            correct = "OK " if predicted_idx == idx else "BAD"
            print(f"  [{correct}] {f}: raw={raw:.4f}  predicted={predicted_label.upper()}")
        print()


if __name__ == "__main__":
    print()
    train_and_save_model()
