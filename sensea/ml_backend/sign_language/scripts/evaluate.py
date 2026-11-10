import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import torch

# ============================================================
# PATHS
# ============================================================

SIGN_LANGUAGE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = SIGN_LANGUAGE_DIR / "data" / "raw" / "isl_isolated_40words"
MODEL_PATH = SIGN_LANGUAGE_DIR / "models" / "sign_language_model.pth"

sys.path.insert(0, str(SIGN_LANGUAGE_DIR))

from labels import LABELS
from model import SignLanguageModel
from scripts.feature_extractor import extract_frame_features


# ============================================================
# CONFIG
# ============================================================

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

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# READ VIDEO
# ============================================================

def read_video(video_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return None

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()

    return frames


# ============================================================
# SAME SAMPLING AS TRAINING
# ============================================================

def sample_frames(frames):

    if len(frames) == 0:
        return None

    if len(frames) > NUM_FRAMES:

        indices = np.linspace(
            0,
            len(frames) - 1,
            NUM_FRAMES
        ).astype(int)

        return [
            frames[i]
            for i in indices
        ]

    sampled = frames.copy()

    while len(sampled) < NUM_FRAMES:
        sampled.append(frames[-1])

    return sampled


# ============================================================
# EXTRACT FEATURES
# ============================================================

def extract_video_features(frames):

    frames = sample_frames(frames)

    if frames is None:
        return None

    sequence = []

    for frame in frames:

        features = extract_frame_features(frame)

        if features is None:
            return None

        sequence.append(features)

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    if sequence.shape != (
        NUM_FRAMES,
        FEATURE_SIZE
    ):
        return None

    return sequence


# ============================================================
# PREDICT
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

    predicted_id = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_id]
    )

    return LABELS[predicted_id], confidence


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print()
    print("=" * 60)
    print("SENSEA — FULL 20-SIGN DATASET EVALUATION")
    print("=" * 60)

    results = []

    total_videos = 0
    successful = 0
    failed = 0
    correct = 0

    # --------------------------------------------------------
    # Process each class
    # --------------------------------------------------------

    for true_label in LABELS:

        class_dir = DATASET_DIR / true_label

        if not class_dir.exists():
            print(
                f"WARNING: Missing directory: {true_label}"
            )
            continue

        videos = sorted(
            class_dir.glob("*.mp4")
        )

        print(
            f"\n[{true_label}] {len(videos)} videos"
        )

        for video_path in videos:

            total_videos += 1

            frames = read_video(
                video_path
            )

            if frames is None:
                failed += 1
                continue

            sequence = extract_video_features(
                frames
            )

            if sequence is None:
                failed += 1
                continue

            predicted_label, confidence = predict(
                sequence
            )

            successful += 1

            is_correct = (
                predicted_label == true_label
            )

            if is_correct:
                correct += 1

            results.append({
                "video": video_path.name,
                "true": true_label,
                "predicted": predicted_label,
                "confidence": confidence,
                "correct": is_correct
            })

            symbol = "✓" if is_correct else "✗"

            print(
                f"  {symbol} "
                f"{video_path.name[:35]:<35} "
                f"→ {predicted_label:<12} "
                f"{confidence * 100:5.1f}%"
            )

    # ========================================================
    # OVERALL RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)

    print(f"Videos found     : {total_videos}")
    print(f"Successful       : {successful}")
    print(f"Failed           : {failed}")

    if successful > 0:

        accuracy = (
            correct / successful
        ) * 100

        print(
            f"Correct          : {correct}"
        )

        print(
            f"Incorrect        : {successful - correct}"
        )

        print(
            f"Accuracy         : {accuracy:.2f}%"
        )

    # ========================================================
    # PER-CLASS RESULTS
    # ========================================================

    print()
    print("=" * 60)
    print("PER-CLASS RESULTS")
    print("=" * 60)

    class_stats = defaultdict(
        lambda: {
            "total": 0,
            "correct": 0
        }
    )

    for result in results:

        true_label = result["true"]

        class_stats[true_label]["total"] += 1

        if result["correct"]:
            class_stats[true_label]["correct"] += 1

    for label in LABELS:

        total = class_stats[label]["total"]
        class_correct = class_stats[label]["correct"]

        if total == 0:
            print(
                f"{label:<15} NO DATA"
            )
            continue

        class_accuracy = (
            class_correct / total
        ) * 100

        print(
            f"{label:<15} "
            f"{class_correct:>3}/{total:<3} "
            f"({class_accuracy:6.2f}%)"
        )

    # ========================================================
    # CONFUSIONS
    # ========================================================

    print()
    print("=" * 60)
    print("COMMON CONFUSIONS")
    print("=" * 60)

    confusion = defaultdict(int)

    for result in results:

        if not result["correct"]:

            key = (
                result["true"],
                result["predicted"]
            )

            confusion[key] += 1

    sorted_confusions = sorted(
        confusion.items(),
        key=lambda x: x[1],
        reverse=True
    )

    if not sorted_confusions:

        print("No incorrect predictions!")

    else:

        for (true_label, predicted_label), count in sorted_confusions:

            print(
                f"{true_label:<15} → "
                f"{predicted_label:<15} "
                f"{count} video(s)"
            )

    # ========================================================
    # SAVE CSV
    # ========================================================

    output_path = (
        SIGN_LANGUAGE_DIR
        / "data"
        / "processed"
        / "evaluation_results.csv"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "video,true,predicted,confidence,correct\n"
        )

        for result in results:

            f.write(
                f'"{result["video"]}",'
                f'"{result["true"]}",'
                f'"{result["predicted"]}",'
                f'{result["confidence"]:.6f},'
                f'{result["correct"]}\n'
            )

    print()
    print(
        f"Results saved to:\n{output_path}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()