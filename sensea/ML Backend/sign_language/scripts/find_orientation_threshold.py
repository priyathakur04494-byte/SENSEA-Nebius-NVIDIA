import cv2
import numpy as np
import pandas as pd
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SPLIT_FILE = BASE_DIR / "data" / "processed" / "signer_split.csv"
SAMPLES_FILE = BASE_DIR / "data" / "processed" / "samples.csv"

HAND_MODEL = (
    BASE_DIR
    / "models"
    / "mediapipe"
    / "hand_landmarker.task"
)


# ============================================================
# LOAD DATA
# ============================================================

split_df = pd.read_csv(SPLIT_FILE)
samples_df = pd.read_csv(SAMPLES_FILE)

df = samples_df.merge(
    split_df[["index", "split"]],
    on="index",
    how="inner"
)

# ONLY TRAIN + VALIDATION
# The unseen-signer test set is untouched.
df = df[df["split"].isin(["train", "val"])].copy()

print("=" * 70)
print("ORIENTATION THRESHOLD SEARCH")
print("=" * 70)
print(f"Train + validation samples: {len(df)}")
print("Test samples are NOT used.")


# ============================================================
# RIGHT-HAND DETECTION
# ============================================================

def measure_right_hand_percentage(video_path):

    # IMPORTANT:
    # Create a NEW detector for every video.
    # MediaPipe VIDEO mode requires timestamps to keep increasing
    # within one detector instance.
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

    detector = vision.HandLandmarker.create_from_options(
        options
    )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        detector.close()
        raise RuntimeError(
            f"Could not open: {video_path}"
        )

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        detector.close()
        raise RuntimeError(
            f"No frames: {video_path}"
        )

    # Same 30-frame sampling used by the pipeline.
    if len(frames) >= 30:

        indices = np.linspace(
            0,
            len(frames) - 1,
            30
        ).astype(int)

        frames = [
            frames[i]
            for i in indices
        ]

    else:

        last = frames[-1]

        while len(frames) < 30:
            frames.append(last.copy())

    right_frames = 0

    for frame_number, frame in enumerate(frames):

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = frame_number * 33

        result = detector.detect_for_video(
            image,
            timestamp
        )

        has_right = False

        for handedness in result.handedness:

            if len(handedness) == 0:
                continue

            category = handedness[0]

            if (
                category.category_name
                and
                category.category_name.lower() == "right"
            ):
                has_right = True
                break

        if has_right:
            right_frames += 1

    detector.close()

    return (
        right_frames /
        len(frames) *
        100.0
    )


# ============================================================
# PROCESS TRAIN + VALIDATION
# ============================================================

results = []

total = len(df)

for n, row in enumerate(
    df.itertuples(index=False),
    start=1
):

    video_path = BASE_DIR.parent / row.video_path

    try:

        right_pct = measure_right_hand_percentage(
            video_path
        )

        results.append({
            "index": row.index,
            "split": row.split,
            "signer": row.signer,
            "label": row.label,
            "right_hand_pct": right_pct,
        })

        print(
            f"[{n}/{total}] "
            f"{row.split:5s} | "
            f"{row.signer:8s} | "
            f"{row.label:12s} | "
            f"right={right_pct:6.2f}%"
        )

    except Exception as e:

        print(
            f"[{n}/{total}] FAILED | "
            f"{row.video_path} | {e}"
        )


results_df = pd.DataFrame(results)


# ============================================================
# THRESHOLD SEARCH
# ============================================================

thresholds = [
    1.0,
    2.5,
    5.0,
    7.5,
    10.0,
    12.5,
    15.0,
    20.0,
]

print()
print("=" * 70)
print("THRESHOLD SUMMARY")
print("=" * 70)

print(
    f"{'Threshold':>12} | "
    f"{'Train+Val gated':>16}"
)

print("-" * 40)

for threshold in thresholds:

    gated = (
        results_df["right_hand_pct"] < threshold
    )

    total_gated = int(gated.sum())

    print(
        f"{threshold:10.1f}% | "
        f"{total_gated:16d}"
    )


# ============================================================
# DATASET SUMMARY
# ============================================================

print()
print("=" * 70)
print("RIGHT-HAND DETECTION SUMMARY")
print("=" * 70)

summary = (
    results_df
    .groupby("signer")["right_hand_pct"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        minimum="min",
        maximum="max"
    )
)

for signer, row in summary.iterrows():

    print(
        f"{str(signer):8s} | "
        f"mean={row['mean']:6.2f}% | "
        f"median={row['median']:6.2f}% | "
        f"min={row['minimum']:6.2f}% | "
        f"max={row['maximum']:6.2f}%"
    )


# ============================================================
# SAVE
# ============================================================

output_file = (
    BASE_DIR
    / "data"
    / "processed"
    / "orientation_threshold_train_val.csv"
)

results_df.to_csv(
    output_file,
    index=False
)

print()
print("=" * 70)
print("THRESHOLD SEARCH COMPLETE")
print("=" * 70)
print(f"Successful videos: {len(results_df)}/{total}")
