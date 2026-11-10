import cv2
import sys
import os
import numpy as np
import torch
from collections import deque, Counter
from pathlib import Path

# Allow imports from sign_language/
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from model import SignLanguageModel
from labels import LABELS
from scripts.feature_extractor import extract_frame_features


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "sign_language/models/sign_language_model.pth"

SEQUENCE_LENGTH = 30

CONFIDENCE_THRESHOLD = 0.80

SMOOTHING_WINDOW = 10

# Validated using train + validation data
ORIENTATION_THRESHOLD = 1.0


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


print("=" * 60)
print("SENSEA — REAL-TIME SIGN LANGUAGE RECOGNITION")
print("=" * 60)

print("Device:", DEVICE)
print("Classes:", len(LABELS))
print(
    f"Orientation gate: right-hand detection < "
    f"{ORIENTATION_THRESHOLD}% → FLIP"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = SignLanguageModel(
    input_size=258,
    num_classes=len(LABELS)
).to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )

model.eval()

print("Model loaded successfully.")


# ============================================================
# MEDIAPIPE ORIENTATION DETECTOR
# ============================================================

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


BASE_DIR = Path(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

HAND_MODEL = (
    BASE_DIR /
    "models" /
    "mediapipe" /
    "hand_landmarker.task"
)


def create_orientation_detector():

    options = vision.HandLandmarkerOptions(

        base_options=python.BaseOptions(
            model_asset_path=str(HAND_MODEL)
        ),

        running_mode=vision.RunningMode.VIDEO,

        num_hands=2,

        min_hand_detection_confidence=0.5,

        min_hand_presence_confidence=0.5,

        min_tracking_confidence=0.5,
    )

    return vision.HandLandmarker.create_from_options(
        options
    )


def detect_right_hand(frame, detector, timestamp):

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect_for_video(
        image,
        timestamp
    )

    for handedness in result.handedness:

        if len(handedness) == 0:
            continue

        category = handedness[0]

        if (
            category.category_name
            and
            category.category_name.lower()
            == "right"
        ):
            return True

    return False


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("\nERROR: Could not open camera.")
    sys.exit(1)


print("\nCamera opened successfully.")

print("\nControls:")
print("  Q = quit")
print("  SPACE = clear sequence")

print("\nFirst 30 frames determine orientation.")
print("Then show one sign clearly to the camera.")


# ============================================================
# TEMPORAL BUFFER
# ============================================================

sequence = deque(
    maxlen=SEQUENCE_LENGTH
)

prediction_history = deque(
    maxlen=SMOOTHING_WINDOW
)


# ============================================================
# ORIENTATION BUFFER
# ============================================================

orientation_detector = (
    create_orientation_detector()
)

orientation_frames = 0

right_hand_frames = 0

orientation_locked = False

use_flip = False


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, raw_frame = cap.read()

    if not ret:

        print(
            "Failed to read camera frame."
        )

        break


    # ========================================================
    # ORIENTATION DETECTION
    # ========================================================

    if not orientation_locked:

        orientation_frames += 1

        timestamp = (
            orientation_frames * 33
        )

        if detect_right_hand(
            raw_frame,
            orientation_detector,
            timestamp
        ):

            right_hand_frames += 1


        # ----------------------------------------------------
        # Once 30 frames are collected, lock orientation.
        # ----------------------------------------------------

        if orientation_frames >= SEQUENCE_LENGTH:

            right_percentage = (
                right_hand_frames /
                orientation_frames *
                100.0
            )

            use_flip = (
                right_percentage
                < ORIENTATION_THRESHOLD
            )

            orientation_locked = True

            orientation_detector.close()

            print()
            print("=" * 60)
            print("ORIENTATION LOCKED")
            print("=" * 60)

            print(
                f"Right-hand detection: "
                f"{right_percentage:.2f}%"
            )

            if use_flip:

                print(
                    "Orientation: FLIPPED"
                )

            else:

                print(
                    "Orientation: ORIGINAL"
                )

            print("=" * 60)


    # ========================================================
    # APPLY SELECTED ORIENTATION
    # ========================================================

    if orientation_locked and use_flip:

        frame = cv2.flip(
            raw_frame,
            1
        )

    else:

        frame = raw_frame.copy()


    # ========================================================
    # MIRROR DISPLAY ONLY
    # ========================================================

    display_frame = cv2.flip(
        frame,
        1
    )


    # ========================================================
    # FEATURE EXTRACTION
    # ========================================================

    features = extract_frame_features(
        frame
    )

    if features is not None:

        sequence.append(
            features
        )


    # ========================================================
    # PREDICTION
    # ========================================================

    predicted_label = "Waiting..."

    confidence = 0.0


    if len(sequence) == SEQUENCE_LENGTH:

        input_sequence = np.array(
            sequence,
            dtype=np.float32
        )

        input_tensor = torch.tensor(
            input_sequence,
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)


        with torch.no_grad():

            outputs = model(
                input_tensor
            )

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            confidence_value, predicted_class = (
                torch.max(
                    probabilities,
                    dim=1
                )
            )

            confidence = (
                confidence_value.item()
            )

            predicted_class = (
                predicted_class.item()
            )


        # ====================================================
        # CONFIDENCE FILTER
        # ====================================================

        if confidence >= CONFIDENCE_THRESHOLD:

            current_label = (
                LABELS[predicted_class]
            )

            prediction_history.append(
                current_label
            )


            if len(prediction_history) >= 3:

                predicted_label = Counter(
                    prediction_history
                ).most_common(1)[0][0]

            else:

                predicted_label = (
                    current_label
                )

        else:

            predicted_label = "Uncertain"


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.rectangle(
        display_frame,
        (10, 10),
        (640, 140),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        display_frame,
        f"Sign: {predicted_label}",
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )


    cv2.putText(
        display_frame,
        f"Confidence: "
        f"{confidence * 100:.1f}%",
        (25, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        display_frame,
        f"Frames: "
        f"{len(sequence)}/{SEQUENCE_LENGTH}",
        (25, 108),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )


    if not orientation_locked:

        orientation_text = (
            f"Orientation: detecting "
            f"({orientation_frames}/30)"
        )

    elif use_flip:

        orientation_text = (
            "Orientation: FLIPPED"
        )

    else:

        orientation_text = (
            "Orientation: ORIGINAL"
        )


    cv2.putText(
        display_frame,
        orientation_text,
        (25, 132),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )


    cv2.imshow(
        "SENSEA - ISL Recognition",
        display_frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


    elif key == 32:

        sequence.clear()

        prediction_history.clear()

        # Recalculate orientation
        orientation_detector.close()

        orientation_detector = (
            create_orientation_detector()
        )

        orientation_frames = 0

        right_hand_frames = 0

        orientation_locked = False

        use_flip = False

        print(
            "\nSequence and orientation cleared."
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

if orientation_detector is not None:

    orientation_detector.close()

print(
    "\nReal-time recognition stopped."
)
