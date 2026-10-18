import cv2
import sys
import os
import numpy as np
import torch
from collections import deque, Counter

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

CONFIDENCE_THRESHOLD = 0.70

SMOOTHING_WINDOW = 7


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("=" * 60)
print("SENSEA — REAL-TIME SIGN LANGUAGE RECOGNITION")
print("=" * 60)

print("Device:", DEVICE)
print("Classes:", len(LABELS))


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
    model.load_state_dict(checkpoint)

model.eval()

print("Model loaded successfully.")


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
print("\nShow one sign clearly to the camera.")
print("Collecting 30 frames...")


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
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera frame.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Extract 258 features from current frame
    features = extract_frame_features(frame)

    if features is not None:

        sequence.append(features)

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

            confidence_value, predicted_class = torch.max(
                probabilities,
                dim=1
            )

            confidence = confidence_value.item()

            predicted_class = predicted_class.item()

        # ====================================================
        # CONFIDENCE FILTER
        # ====================================================

        if confidence >= CONFIDENCE_THRESHOLD:

            current_label = LABELS[predicted_class]

            prediction_history.append(
                current_label
            )

            # Majority vote
            if len(prediction_history) >= 3:

                predicted_label = Counter(
                    prediction_history
                ).most_common(1)[0][0]

            else:

                predicted_label = current_label

        else:

            predicted_label = "Uncertain"


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (620, 110),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        f"Sign: {predicted_label}",
        (25, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Confidence: {confidence * 100:.1f}%",
        (25, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Frames: {len(sequence)}/{SEQUENCE_LENGTH}",
        (25, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    cv2.imshow(
        "SENSEA - ISL Recognition",
        frame
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

        print("\nSequence cleared.")


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("\nReal-time recognition stopped.")
