import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# ============================================================
# PATHS
# ============================================================

SIGN_LANGUAGE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SIGN_LANGUAGE_DIR))

MODEL_PATH = str(SIGN_LANGUAGE_DIR / "models" / "sign_language_model.pth")

# ============================================================
# IMPORTS
# ============================================================

from labels import LABELS
from model import SignLanguageModel
from scripts.feature_extractor import extract_frame_features

NUM_FRAMES = 30
FEATURE_SIZE = 258


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}")


# ============================================================
# LOAD MODEL
# ============================================================

model = SignLanguageModel(
    input_size=FEATURE_SIZE,
    num_classes=len(LABELS)
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

# Support both checkpoint formats
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# READ ALL VIDEO FRAMES
# ============================================================

def read_video_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()

    return frames, fps, total_frames


# ============================================================
# SAME 30-FRAME SAMPLING AS TRAINING
# ============================================================

def sample_frames(frames, num_frames=NUM_FRAMES):
    if len(frames) == 0:
        raise RuntimeError("Video contains no readable frames.")

    # --------------------------------------------------------
    # If video has more than 30 frames:
    # uniformly sample across the entire video
    # --------------------------------------------------------

    if len(frames) > num_frames:

        indices = np.linspace(
            0,
            len(frames) - 1,
            num_frames
        ).astype(int)

        sampled = [
            frames[i]
            for i in indices
        ]

    # --------------------------------------------------------
    # If video has fewer than 30 frames:
    # repeat the final frame
    # --------------------------------------------------------

    else:

        sampled = frames.copy()

        while len(sampled) < num_frames:
            sampled.append(frames[-1])

    return sampled


# ============================================================
# EXTRACT 30 × 258 FEATURES
# ============================================================

def extract_video_features(frames):

    sampled_frames = sample_frames(
        frames,
        NUM_FRAMES
    )

    sequence = []

    for frame in sampled_frames:

        features = extract_frame_features(frame)

        if features is None:
            raise RuntimeError(
                "Feature extraction returned None."
            )

        sequence.append(features)

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    if sequence.shape != (NUM_FRAMES, FEATURE_SIZE):
        raise RuntimeError(
            f"Unexpected feature shape: {sequence.shape}"
        )

    return sequence


# ============================================================
# PREDICTION
# ============================================================

def predict(sequence):

    x = torch.tensor(
        sequence,
        dtype=torch.float32
    ).unsqueeze(0)

    x = x.to(DEVICE)

    with torch.no_grad():

        logits = model(x)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

    probabilities = probabilities[0].cpu().numpy()

    top_indices = np.argsort(
        probabilities
    )[::-1][:5]

    return [
        (
            LABELS[i],
            float(probabilities[i] * 100)
        )
        for i in top_indices
    ]


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        required=True,
        help="Path to video"
    )

    args = parser.parse_args()

    video_path = Path(args.video)

    if not video_path.exists():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    print()
    print("=" * 60)
    print("SENSEA — EXACT DATASET PIPELINE TEST")
    print("=" * 60)
    print(f"Video: {video_path}")

    # --------------------------------------------------------
    # Read video
    # --------------------------------------------------------

    frames, fps, total_frames = read_video_frames(
        video_path
    )

    print(f"Frames: {total_frames}")
    print(f"FPS: {fps:.1f}")

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    sequence = extract_video_features(
        frames
    )

    print(f"Features: {FEATURE_SIZE}")
    print(f"Sequence shape: {sequence.shape}")

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    top5 = predict(sequence)

    print()
    print("=" * 60)
    print("TOP 5 PREDICTIONS")
    print("=" * 60)

    for rank, (label, confidence) in enumerate(
        top5,
        start=1
    ):

        print(
            f"{rank}. {label:<15} {confidence:.2f}%"
        )

    print()
    print("=" * 60)
    print("EXPECTED LABEL: hello")
    print("=" * 60)

    predicted_label = top5[0][0]
    predicted_confidence = top5[0][1]

    print(
        f"Predicted: {predicted_label}"
    )

    print(
        f"Confidence: {predicted_confidence:.2f}%"
    )

    if predicted_label == "hello":
        print()
        print("✅ CORRECT — HELLO RECOGNIZED")
    else:
        print()
        print("❌ NOT CORRECT — HELLO WAS NOT TOP-1")