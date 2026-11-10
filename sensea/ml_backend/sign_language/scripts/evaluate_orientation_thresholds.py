import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from sign_language.model import SignLanguageModel


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SPLIT_FILE = BASE_DIR / "data" / "processed" / "signer_split.csv"
SAMPLES_FILE = BASE_DIR / "data" / "processed" / "samples.csv"
MODEL_FILE = BASE_DIR / "models" / "sign_language_model.pth"

HAND_MODEL = (
    BASE_DIR
    / "models"
    / "mediapipe"
    / "hand_landmarker.task"
)

POSE_MODEL = (
    BASE_DIR
    / "models"
    / "mediapipe"
    / "pose_landmarker.task"
)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}")


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

# NEVER use unseen-signer test data here.
df = df[df["split"].isin(["train", "val"])].copy()

print("=" * 70)
print("ORIENTATION THRESHOLD CLASSIFICATION TEST")
print("=" * 70)
print(f"Train + validation samples: {len(df)}")
print("Unseen-signer test set: NOT USED")


# ============================================================
# LOAD MODEL
# ============================================================

model = SignLanguageModel(
    input_size=258,
    num_classes=20
).to(DEVICE)

checkpoint = torch.load(
    MODEL_FILE,
    map_location=DEVICE,
    weights_only=True
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()


# ============================================================
# LABELS
# ============================================================

LABELS = [
    "hello",
    "goodbye",
    "thank_you",
    "sorry",
    "please",
    "yes",
    "no",
    "help",
    "stop",
    "okay",
    "me",
    "you",
    "mother",
    "father",
    "friend",
    "home",
    "hospital",
    "eat",
    "drink",
    "water",
]


# ============================================================
# CREATE MEDIAPIPE DETECTORS
# ============================================================

def create_detectors():

    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(HAND_MODEL)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(POSE_MODEL)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    hand_detector = vision.HandLandmarker.create_from_options(
        hand_options
    )

    pose_detector = vision.PoseLandmarker.create_from_options(
        pose_options
    )

    return hand_detector, pose_detector


# ============================================================
# FEATURE EXTRACTION HELPERS
# ============================================================

def normalize_hand(landmarks):

    points = np.array(
        [[p.x, p.y, p.z] for p in landmarks],
        dtype=np.float32
    )

    wrist = points[0]

    points = points - wrist

    scale = np.linalg.norm(
        points[9]
    )

    if scale > 1e-6:
        points = points / scale

    return points.flatten()


def normalize_pose(landmarks):

    points = np.array(
        [
            [p.x, p.y, p.z, p.visibility]
            for p in landmarks
        ],
        dtype=np.float32
    )

    shoulder_center = (
        points[11, :3] +
        points[12, :3]
    ) / 2.0

    points[:, :3] -= shoulder_center

    shoulder_distance = np.linalg.norm(
        points[11, :3] -
        points[12, :3]
    )

    if shoulder_distance > 1e-6:
        points[:, :3] /= shoulder_distance

    return points.flatten()


def extract_features(
    hand_result,
    pose_result
):

    left_hand = np.zeros(
        63,
        dtype=np.float32
    )

    right_hand = np.zeros(
        63,
        dtype=np.float32
    )

    for hand_landmarks, handedness in zip(
        hand_result.hand_landmarks,
        hand_result.handedness
    ):

        if len(handedness) == 0:
            continue

        name = handedness[0].category_name.lower()

        features = normalize_hand(
            hand_landmarks
        )

        if name == "left":
            left_hand = features

        elif name == "right":
            right_hand = features

    pose = np.zeros(
        132,
        dtype=np.float32
    )

    if pose_result.pose_landmarks:

        pose = normalize_pose(
            pose_result.pose_landmarks[0]
        )

    return np.concatenate(
        [
            left_hand,
            right_hand,
            pose
        ]
    )


# ============================================================
# EXTRACT ONE VIDEO
# ============================================================

def extract_video_features(video_path, flip=False):

    hand_detector, pose_detector = create_detectors()

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        hand_detector.close()
        pose_detector.close()
        raise RuntimeError(
            f"Could not open {video_path}"
        )

    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if flip:
            frame = cv2.flip(
                frame,
                1
            )

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        hand_detector.close()
        pose_detector.close()
        raise RuntimeError(
            f"No frames: {video_path}"
        )

    # Same 30-frame sampling.
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
            frames.append(
                last.copy()
            )

    sequence = []

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

        hand_result = hand_detector.detect_for_video(
            image,
            timestamp
        )

        pose_result = pose_detector.detect_for_video(
            image,
            timestamp
        )

        features = extract_features(
            hand_result,
            pose_result
        )

        sequence.append(features)

    hand_detector.close()
    pose_detector.close()

    return np.array(
        sequence,
        dtype=np.float32
    )


# ============================================================
# RIGHT HAND DETECTION
# ============================================================

def right_hand_percentage(video_path):

    detector_options = vision.HandLandmarkerOptions(
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
        detector_options
    )

    cap = cv2.VideoCapture(
        str(video_path)
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
            frames.append(
                last.copy()
            )

    right_count = 0

    for frame_number, frame in enumerate(frames):

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
            frame_number * 33
        )

        found_right = False

        for handedness in result.handedness:

            if len(handedness) == 0:
                continue

            if (
                handedness[0]
                .category_name
                .lower()
                == "right"
            ):
                found_right = True
                break

        if found_right:
            right_count += 1

    detector.close()

    return (
        right_count /
        len(frames) *
        100.0
    )


# ============================================================
# MODEL PREDICTION
# ============================================================

def predict(sequence):

    tensor = torch.from_numpy(
        sequence
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        logits = model(
            tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

    confidence, prediction = torch.max(
        probabilities,
        dim=1
    )

    return (
        int(prediction.item()),
        float(confidence.item())
    )


# ============================================================
# PROCESS ALL TRAIN + VAL
# ============================================================

records = []

total = len(df)

for n, row in enumerate(
    df.itertuples(index=False),
    start=1
):

    video_path = BASE_DIR.parent / row.video_path

    try:

        original_sequence = extract_video_features(
            video_path,
            flip=False
        )

        flipped_sequence = extract_video_features(
            video_path,
            flip=True
        )

        original_prediction, original_conf = predict(
            original_sequence
        )

        flipped_prediction, flipped_conf = predict(
            flipped_sequence
        )

        right_pct = right_hand_percentage(
            video_path
        )

        records.append({
            "index": row.index,
            "split": row.split,
            "signer": row.signer,
            "label": row.label,
            "label_id": row.label_id,
            "right_hand_pct": right_pct,
            "original_prediction": original_prediction,
            "original_label": LABELS[original_prediction],
            "original_confidence": original_conf,
            "flipped_prediction": flipped_prediction,
            "flipped_label": LABELS[flipped_prediction],
            "flipped_confidence": flipped_conf,
        })

        print(
            f"[{n}/{total}] "
            f"{row.split:5s} | "
            f"{row.label:12s} | "
            f"right={right_pct:6.2f}% | "
            f"orig={LABELS[original_prediction]:12s} | "
            f"flip={LABELS[flipped_prediction]:12s}"
        )

    except Exception as e:

        print(
            f"[{n}/{total}] FAILED | "
            f"{row.video_path} | {e}"
        )


results = pd.DataFrame(records)


# ============================================================
# THRESHOLD EVALUATION
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
print("THRESHOLD CLASSIFICATION RESULTS")
print("=" * 70)

print(
    f"{'Threshold':>10} | "
    f"{'Accuracy':>10} | "
    f"{'Correct':>10} | "
    f"{'Flipped':>10}"
)

print("-" * 55)

summary = []

for threshold in thresholds:

    predictions = np.where(
        results["right_hand_pct"].values < threshold,
        results["flipped_prediction"].values,
        results["original_prediction"].values
    )

    correct = (
        predictions ==
        results["label_id"].values
    )

    accuracy = (
        correct.mean() * 100.0
    )

    flipped_count = int(
        (
            results["right_hand_pct"]
            < threshold
        ).sum()
    )

    summary.append({
        "threshold": threshold,
        "accuracy": accuracy,
        "correct": int(correct.sum()),
        "total": len(correct),
        "flipped": flipped_count,
    })

    print(
        f"{threshold:8.1f}% | "
        f"{accuracy:9.2f}% | "
        f"{int(correct.sum()):10d} | "
        f"{flipped_count:10d}"
    )


# ============================================================
# BEST THRESHOLD
# ============================================================

summary_df = pd.DataFrame(summary)

best = summary_df.loc[
    summary_df["accuracy"].idxmax()
]

print()
print("=" * 70)
print("BEST TRAIN+VALIDATION THRESHOLD")
print("=" * 70)

print(
    f"Threshold : {best['threshold']:.1f}%"
)

print(
    f"Accuracy  : {best['accuracy']:.2f}%"
)

print(
    f"Correct   : {int(best['correct'])}/{int(best['total'])}"
)

print(
    f"Flipped   : {int(best['flipped'])}"
)


# ============================================================
# SAVE
# ============================================================

output_predictions = (
    BASE_DIR
    / "data"
    / "processed"
    / "orientation_threshold_predictions.csv"
)

output_summary = (
    BASE_DIR
    / "data"
    / "processed"
    / "orientation_threshold_results.csv"
)

results.to_csv(
    output_predictions,
    index=False
)

summary_df.to_csv(
    output_summary,
    index=False
)

print()
print(f"Predictions saved to: {output_predictions}")
print(f"Summary saved to:     {output_summary}")
print()
print("THRESHOLD TEST COMPLETE")
