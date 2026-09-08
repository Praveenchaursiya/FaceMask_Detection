#!/usr/bin/env python3
"""
Face Mask Detection - Real-time Webcam Application
Python + OpenCV + TensorFlow/Keras
B.Tech Minor Project
"""

import os
import sys
import json
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model


# ==================== CONFIGURATION ====================
WINDOW_TITLE = "Face Mask Detection System"
WINDOW_WIDTH = 1080
WINDOW_HEIGHT = 720
FACE_SIZE = (128, 128)
MODEL_PATH = "models/face_mask_model.keras"
CLASS_INDICES_PATH = "models/class_indices.json"
YUNET_MODEL_PATH = "models/face_detection_yunet.onnx"

# Colors - Modern Dashboard Theme
COLOR_PRIMARY = "#1a73e8"
COLOR_SUCCESS = "#0d9488"
COLOR_DANGER = "#dc2626"
COLOR_WARNING = "#f59e0b"
COLOR_BG_DARK = "#0f172a"
COLOR_BG_CARD = "#1e293b"
COLOR_TEXT_WHITE = "#ffffff"
COLOR_TEXT_GRAY = "#94a3b8"
COLOR_TEXT_LIGHT = "#e2e8f0"

# Bounding box colors
COLOR_MASK = (0, 255, 0)      # Green for MASK
COLOR_NO_MASK = (0, 0, 255)   # Red for NO MASK

# Debug mode - prints class_indices and prediction for every detected face
DEBUG_MODE = True

# How close (in pixels) a face position must be to an already-seen face
# to be considered the same face (used to avoid counting the same face
# many times per second).
SAME_FACE_DISTANCE = 80


# ==================== HELPERS ====================
def load_class_mapping(path=CLASS_INDICES_PATH):
    """Load the class mapping saved by train_model.py.

    Returns a dict:
        {
            "mask_label":    "MASK" or "NO MASK"  (the label shown when raw >= 0.5)
            "no_mask_label": "..."                (the label shown when raw <  0.5)
            "class_indices": {...},
            "img_size": [w, h]
        }
    """
    default = {
        "mask_label": "MASK",
        "no_mask_label": "NO MASK",
        "class_indices": {"mask": 0, "no_mask": 1},
        "img_size": [128, 128]
    }

    if not os.path.exists(path):
        print(f"[WARN] {path} not found. Using default mapping: "
              f"output >= 0.5 = NO MASK, output < 0.5 = MASK.")
        return default

    try:
        with open(path, 'r') as f:
            data = json.load(f)
        ci = data.get("class_indices", default["class_indices"])
        inv = {v: k for k, v in ci.items()}

        # The class whose index is 1 is what the sigmoid pushes TOWARDS 1.0.
        high_label = inv.get(1, "no_mask").upper().replace("_", " ")
        low_label = inv.get(0, "mask").upper().replace("_", " ")

        out = default.copy()
        out["class_indices"] = ci
        out["mask_label"] = high_label     # when raw >= 0.5
        out["no_mask_label"] = low_label   # when raw <  0.5
        out["img_size"] = data.get("img_size", [128, 128])
        return out
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}. Using default mapping.")
        return default


# ==================== FACE DETECTOR ====================
class FaceDetector:
    """Handle face detection using YuNet ONNX model (OpenCV 5.x)."""
    def __init__(self, model_path=YUNET_MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Face detector model not found: {model_path}")

        self.detector = cv2.FaceDetectorYN_create(model_path, '', (320, 320))
        self.detector.setScoreThreshold(0.3)

    def detect_faces(self, frame):
        """Return list of (x, y, w, h) tuples for each detected face."""
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)

        rects = []
        if faces is not None:
            for face in faces:
                x, y, bw, bh = face[:4].astype(int)
                rects.append((int(x), int(y), int(bw), int(bh)))
        return rects


# ==================== MASK CLASSIFIER ====================
class MaskClassifier:
    """Classify a face ROI as MASK or NO MASK.

    The class mapping is loaded from models/class_indices.json so we
    never hard-code which sigmoid value corresponds to which class.
    """

    def __init__(self, class_mapping):
        self.model = None
        self.face_detector = None
        self.class_mapping = class_mapping
        # Convenience labels: high_label is shown when raw >= 0.5
        self.high_label = class_mapping["mask_label"]     # e.g. "NO MASK"
        self.low_label = class_mapping["no_mask_label"]   # e.g. "MASK"
        self.class_indices = class_mapping["class_indices"]
        self.face_size = tuple(class_mapping["img_size"])
        self._load_models()

    def _load_models(self):
        # Face detector
        try:
            self.face_detector = FaceDetector()
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            messagebox.showerror(
                "Error",
                f"Face detector model not found.\n\n"
                f"Please ensure '{YUNET_MODEL_PATH}' exists."
            )
            self.face_detector = None
            return

        # Mask classifier
        if os.path.exists(MODEL_PATH):
            self.model = load_model(MODEL_PATH)
            if DEBUG_MODE:
                print("[DEBUG] CLASS INDICES:", self.class_indices)
                print(f"[DEBUG] raw >= 0.5 -> '{self.high_label}'")
                print(f"[DEBUG] raw <  0.5 -> '{self.low_label}'")
                print(f"[DEBUG] face input size: {self.face_size}, /255.0, RGB")
        else:
            print(f"WARNING: Model not found at {MODEL_PATH}")
            print("Run train_model.py first.")
            self.model = None

    def get_face_roi(self, frame, face_coords):
        """Extract a padded face ROI from the frame."""
        x, y, w, h = face_coords
        pad_x = int(w * 0.1)
        pad_y = int(h * 0.1)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame.shape[1], x + w + pad_x)
        y2 = min(frame.shape[0], y + h + pad_y)
        roi = frame[y1:y2, x1:x2]
        return roi

    def preprocess_face(self, face_roi):
        """Preprocess face ROI to match training pipeline.

        Training pipeline (see train_model.py):
            - read with cv2 (BGR)
            - convert BGR -> RGB
            - resize to (128, 128)
            - rescale by /255.0
        """
        if face_roi is None or face_roi.size == 0:
            return None
        if face_roi.shape[0] < 10 or face_roi.shape[1] < 10:
            return None

        rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, self.face_size)
        normalized = resized.astype('float32') / 255.0
        return normalized

    def predict(self, face_roi):
        """Return (label, confidence, raw_output).

        label:     one of the strings from class_mapping
        confidence: probability of the predicted class (0..1)
        raw_output: raw sigmoid value (0..1)
        """
        if self.model is None:
            return "NO MODEL", 0.0, 0.0

        processed = self.preprocess_face(face_roi)
        if processed is None:
            return "INVALID", 0.0, 0.0

        x = np.expand_dims(processed, axis=0)
        raw = float(self.model.predict(x, verbose=0)[0][0])

        # IMPORTANT: the class mapping comes from training.
        # If mask=0 and no_mask=1, then raw>=0.5 means "no_mask" (label = NO MASK).
        if raw >= 0.5:
            label = self.high_label
            confidence = raw
        else:
            label = self.low_label
            confidence = 1.0 - raw

        if DEBUG_MODE:
            print(f"[DEBUG] RAW PREDICTION: {raw:.4f}")
            print(f"[DEBUG] CLASS INDICES:  {self.class_indices}")
            print(f"[DEBUG] PREDICTED CLASS: {label}")
            print(f"[DEBUG] CONFIDENCE:     {confidence:.4f}")

        return label, float(confidence), raw


# ==================== MAIN APPLICATION ====================
class FaceMaskApp:
    """Main application class with dashboard UI."""

    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg=COLOR_BG_DARK)
        self.root.resizable(False, False)

        # State
        self.is_running = False
        self.cap = None
        self.classifier = None
        self.photo = None

        # ---- Session counters (cumulative across the current camera run) ----
        # Reset to 0 when START CAMERA is pressed.
        self.session_total_faces = 0
        self.session_mask_count = 0
        self.session_no_mask_count = 0

        # ---- Current-frame display state ----
        self.current_label = None
        self.current_confidence = 0.0
        self.current_raw = 0.0
        self.current_total_faces = 0
        self.current_mask_count = 0
        self.current_no_mask_count = 0

        # ---- De-duplication of face counts ----
        # Track face centres we have already counted in this session so that
        # the same person isn't counted 30 times per second.
        self.counted_centres = []  # list of (cx, cy)

        self._setup_ui()

        # Build classifier (uses class_indices.json written by train_model.py)
        class_mapping = load_class_mapping(CLASS_INDICES_PATH)
        if DEBUG_MODE:
            print("[DEBUG] Loaded class mapping:", class_mapping)
        self.classifier = MaskClassifier(class_mapping)

        # Start the UI refresh loop
        self._update_stats()

    # ---------------- UI ----------------
    def _setup_ui(self):
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Header
        header_frame = ttk.Frame(main_container)
        header_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(
            header_frame,
            text="FACE MASK DETECTION",
            font=("Segoe UI", 28, "bold"),
            fg=COLOR_TEXT_WHITE,
            bg=COLOR_BG_DARK
        ).pack(side=tk.LEFT)

        self.status_label = tk.Label(
            header_frame,
            text="● OFFLINE",
            font=("Segoe UI", 14, "bold"),
            fg=COLOR_DANGER,
            bg=COLOR_BG_DARK
        )
        self.status_label.pack(side=tk.RIGHT, padx=10)

        # Content
        content_frame = ttk.Frame(main_container)
        content_frame.pack(fill=tk.BOTH, expand=True)

        # Left: video
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            left_frame,
            width=640, height=480,
            bg="#000000",
            highlightthickness=2,
            highlightbackground=COLOR_PRIMARY
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.overlay_text = tk.Label(
            left_frame, text="Press START to begin",
            font=("Segoe UI", 12),
            fg=COLOR_TEXT_GRAY, bg=COLOR_BG_DARK
        )
        self.overlay_text.pack(pady=(5, 0))

        # Right: controls + stats
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))

        btn_frame = ttk.LabelFrame(right_frame, text="Controls", padding=15)
        btn_frame.pack(fill=tk.X, pady=(0, 15))

        self.start_btn = tk.Button(
            btn_frame, text="▶ START CAMERA",
            font=("Segoe UI", 13, "bold"),
            fg="white", bg=COLOR_SUCCESS,
            activebackground="#059669", activeforeground="white",
            relief=tk.FLAT, cursor="hand2",
            command=self._start_camera, width=20, height=2
        )
        self.start_btn.pack(pady=5, fill=tk.X)

        self.stop_btn = tk.Button(
            btn_frame, text="■ STOP CAMERA",
            font=("Segoe UI", 13, "bold"),
            fg="white", bg=COLOR_WARNING,
            activebackground="#d97706", activeforeground="white",
            relief=tk.FLAT, cursor="hand2",
            command=self._stop_camera, width=20, height=2,
            state=tk.DISABLED
        )
        self.stop_btn.pack(pady=5, fill=tk.X)

        self.exit_btn = tk.Button(
            btn_frame, text="✕ EXIT",
            font=("Segoe UI", 13, "bold"),
            fg="white", bg=COLOR_DANGER,
            activebackground="#b91c1c", activeforeground="white",
            relief=tk.FLAT, cursor="hand2",
            command=self._exit_app, width=20, height=2
        )
        self.exit_btn.pack(pady=5, fill=tk.X)

        # Detection result card
        result_frame = ttk.LabelFrame(right_frame, text="Detection Result", padding=15)
        result_frame.pack(fill=tk.X, pady=(0, 15))

        self.result_label = tk.Label(
            result_frame, text="--",
            font=("Segoe UI", 24, "bold"),
            fg=COLOR_TEXT_GRAY, bg=COLOR_BG_CARD
        )
        self.result_label.pack(pady=(5, 2))

        self.confidence_label = tk.Label(
            result_frame, text="Confidence: --%",
            font=("Segoe UI", 14),
            fg=COLOR_TEXT_LIGHT, bg=COLOR_BG_CARD
        )
        self.confidence_label.pack(pady=(0, 10))

        # Session stats
        stats_frame = ttk.LabelFrame(right_frame, text="Session Stats", padding=15)
        stats_frame.pack(fill=tk.X)

        def make_stat_row(parent, caption, fg_caption):
            row = ttk.Frame(parent)
            row.pack(fill=tk.X, pady=3)
            tk.Label(
                row, text=caption,
                font=("Segoe UI", 11),
                fg=fg_caption, bg=COLOR_BG_CARD
            ).pack(side=tk.LEFT)
            value = tk.Label(
                row, text="0",
                font=("Segoe UI", 14, "bold"),
                fg=COLOR_TEXT_WHITE, bg=COLOR_BG_CARD
            )
            value.pack(side=tk.RIGHT)
            return value

        self.total_count_label = make_stat_row(stats_frame, "Total Faces:", COLOR_TEXT_LIGHT)
        self.mask_count_label = make_stat_row(stats_frame, "MASK Detected:", COLOR_SUCCESS)
        self.no_mask_count_label = make_stat_row(stats_frame, "NO MASK Detected:", COLOR_DANGER)

        # Footer
        footer = ttk.Frame(main_container)
        footer.pack(fill=tk.X, pady=(15, 0))
        tk.Label(
            footer,
            text="Face Mask Detection System | B.Tech Minor Project | Python + OpenCV + TensorFlow",
            font=("Segoe UI", 9),
            fg=COLOR_TEXT_GRAY, bg=COLOR_BG_DARK
        ).pack(fill=tk.X)

    # ---------------- Stats refresh ----------------
    def _update_stats(self):
        """Periodically refresh the stats labels from state variables."""
        # Show CURRENT FRAME stats (not cumulative session stats)
        self.total_count_label.config(text=str(self.current_total_faces))
        self.mask_count_label.config(text=str(self.current_mask_count))
        self.no_mask_count_label.config(text=str(self.current_no_mask_count))

        if self.current_label and self.current_label not in ("NO MODEL", "INVALID"):
            self.result_label.config(
                text=self.current_label,
                fg=COLOR_SUCCESS if self.current_label == self.classifier.low_label
                else COLOR_DANGER
            )
            self.confidence_label.config(
                text=f"Confidence: {self.current_confidence * 100:.1f}%",
                fg=COLOR_SUCCESS if self.current_label == self.classifier.low_label
                else COLOR_DANGER
            )
        else:
            self.result_label.config(text="No face detected", fg=COLOR_TEXT_GRAY)
            self.confidence_label.config(text="Confidence: --%", fg=COLOR_TEXT_LIGHT)

        self.root.after(100, self._update_stats)

    # ---------------- Camera control ----------------
    def _start_camera(self):
        if self.is_running:
            return

        if self.classifier is None or self.classifier.face_detector is None or self.classifier.model is None:
            messagebox.showerror(
                "Error",
                "Models not loaded.\n\n"
                f"Make sure both:\n  - {YUNET_MODEL_PATH}\n  - {MODEL_PATH}\n"
                f"  - {CLASS_INDICES_PATH}\nexist."
            )
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Cannot open webcam.")
            return

        # Flush a few frames so the first displayed frame isn't black
        for _ in range(5):
            self.cap.read()

        # Reset session stats for this new run
        self.session_total_faces = 0
        self.session_mask_count = 0
        self.session_no_mask_count = 0
        self.counted_centres = []
        self.current_label = None
        self.current_confidence = 0.0
        self.current_raw = 0.0

        self.is_running = True
        self.status_label.config(text="● LIVE", fg=COLOR_SUCCESS)
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.overlay_text.config(text="Detecting faces...", fg=COLOR_SUCCESS)

        if DEBUG_MODE:
            print("\n[DEBUG] === START CAMERA ===")
            print(f"[DEBUG] CLASS INDICES: {self.classifier.class_indices}")
            print(f"[DEBUG] high_label (raw >= 0.5) = {self.classifier.high_label}")
            print(f"[DEBUG] low_label  (raw <  0.5) = {self.classifier.low_label}")

        self._update_frame()

    def _stop_camera(self):
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.status_label.config(text="● OFFLINE", fg=COLOR_DANGER)
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.overlay_text.config(text="Press START to begin", fg=COLOR_TEXT_GRAY)

        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, 640, 480, fill="#000000")

        # Reset current-frame counters
        self.current_total_faces = 0
        self.current_mask_count = 0
        self.current_no_mask_count = 0

        # Clear current detection state
        self.current_label = None
        self.current_confidence = 0.0
        self.current_raw = 0.0

    # ---------------- Frame loop ----------------
    def _already_counted(self, cx, cy):
        """Return True if a face at (cx,cy) is close to one we already counted."""
        for (px, py) in self.counted_centres:
            if (cx - px) ** 2 + (cy - py) ** 2 < SAME_FACE_DISTANCE ** 2:
                return True
        return False

    def _update_frame(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self._stop_camera()
            messagebox.showerror("Error", "Failed to capture frame from webcam.")
            return

        # Detect faces ONCE per frame - use SAME result for everything
        detected_faces = self.classifier.face_detector.detect_faces(frame)

        # RESET current-frame counters (critical fix!)
        frame_total_faces = 0
        frame_mask_count = 0
        frame_no_mask_count = 0

        latest_label = None
        latest_confidence = 0.0
        latest_raw = 0.0

        for (x, y, w, h) in detected_faces:
            face_roi = self.classifier.get_face_roi(frame, (x, y, w, h))
            if face_roi is None or face_roi.size == 0:
                continue

            label, confidence, raw = self.classifier.predict(face_roi)
            if label in ("NO MODEL", "INVALID"):
                continue

            # Count THIS frame's detections
            frame_total_faces += 1
            if label == self.classifier.low_label:
                frame_mask_count += 1
            else:
                frame_no_mask_count += 1

            # DEBUG: Show current frame stats
            print(
                f"Current Frame -> Faces: {frame_total_faces}, "
                f"Mask: {frame_mask_count}, "
                f"No Mask: {frame_no_mask_count}"
            )

            # Draw bounding box + label
            color = COLOR_MASK if label == self.classifier.low_label else COLOR_NO_MASK
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label_text = f"{label} {confidence * 100:.1f}%"
            (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            bg_y1 = max(y - th - 10, 0)
            cv2.rectangle(frame, (x, bg_y1), (x + tw + 10, y), color, cv2.FILLED)
            cv2.putText(frame, label_text, (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Track latest detection for result display
            latest_label = label
            latest_confidence = confidence
            latest_raw = raw

        # Update current-frame display state (used by Session Stats UI)
        self.current_total_faces = frame_total_faces
        self.current_mask_count = frame_mask_count
        self.current_no_mask_count = frame_no_mask_count

        # Update current detection state
        self.current_label = latest_label
        self.current_confidence = latest_confidence
        self.current_raw = latest_raw

        # Push frame to canvas
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        self.photo = ImageTk.PhotoImage(image=img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

        self.root.after(30, self._update_frame)

    # ---------------- Exit ----------------
    def _exit_app(self):
        self._stop_camera()
        self.root.destroy()
        sys.exit(0)


# ==================== ENTRY POINT ====================
def main():
    try:
        import cv2
        import tensorflow
        import PIL
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install: pip install -r requirements.txt")
        sys.exit(1)

    if not os.path.exists(YUNET_MODEL_PATH):
        print(f"WARNING: Face detector model not found at {YUNET_MODEL_PATH}")

    if not os.path.exists(MODEL_PATH):
        print(f"WARNING: Mask classifier model not found at {MODEL_PATH}")
        print("Run 'python train_model.py' to train the model.")

    if not os.path.exists(CLASS_INDICES_PATH):
        print(f"WARNING: Class mapping not found at {CLASS_INDICES_PATH}")
        print("It is created automatically by train_model.py.")

    root = tk.Tk()
    app = FaceMaskApp(root)
    root.protocol("WM_DELETE_WINDOW", app._exit_app)
    root.mainloop()


if __name__ == "__main__":
    main()
