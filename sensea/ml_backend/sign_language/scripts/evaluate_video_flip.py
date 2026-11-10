import cv2
import numpy as np
import pandas as pd
import torch
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from sign_language.model import SignLanguageModel
from sign_language.labels import LABELS


# ============================================================
# PATHS
# ============================================================

SAMPLES_PATH = "sign_language/data/processed/samples.csv"
SPLIT_PATH = "sign_language/data/processed/signer_split.csv"
MODEL_PATH = "sign_language/models/sign_language_model.pth"

HAND_MODEL = "sign_language/models/mediapipe/hand_landmarker.task"
POSE_MODEL = "sign_language/models/mediapipe/pose_landmarker.task"

NUM_FRAMES = 30
FEATURE_SIZE = 258


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_hand(landmarks):

    points = np.array(
        [[p.x, p.y, p.z] for p in landmarks],
        dtype=np.float32
    )

    wrist = points[0]
    points = points - wrist

    scale = np.linalg.norm(points[9])

    if scale > 1e-6:
        points = points / scale

    return points.flatten()


def normalize_pose(landmarks):

    points = np.array(
        [[p.x, p.y, p.z, p.visibility] for p in landmarks],
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


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_video_features(video_path, flip=False):

    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=HAND_MODEL
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=POSE_MODEL
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR opening: {video_path}")
        return None

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    # Read all frames first
    frames = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if flip:
            frame = cv2.flip(frame, 1)

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return None

    # ========================================================
    # Uniformly sample / pad to 30 frames
    # ========================================================

    if len(frames) >= NUM_FRAMES:

        indices = np.linspace(
            0,
            len(frames) - 1,
            NUM_FRAMES
        ).astype(int)

        frames = [
            frames[i]
            for i in indices
        ]

    else:

        last_frame = frames[-1]

        while len(frames) < NUM_FRAMES:
            frames.append(last_frame.copy())

    features = []

    with vision.HandLandmarker.create_from_options(
        hand_options
    ) as hand_detector, vision.PoseLandmarker.create_from_options(
        pose_options
    ) as pose_detector:

        for frame_index, frame in enumerate(frames):

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = int(
                frame_index * 1000 / fps
            )

            hand_result = hand_detector.detect_for_video(
                image,
                timestamp_ms
            )

            pose_result = pose_detector.detect_for_video(
                image,
                timestamp_ms
            )

            left_hand = np.zeros(
                63,
                dtype=np.float32
            )

            right_hand = np.zeros(
                63,
                dtype=np.float32
            )

            for i, hand_landmarks in enumerate(
                hand_result.hand_landmarks
            ):

                if i >= len(hand_result.handedness):
                    continue

                handedness = (
                    hand_result.handedness[i][0]
                    .category_name
                    .lower()
                )

                hand_features = normalize_hand(
                    hand_landmarks
                )

                if handedness == "left":

                    left_hand = hand_features

                elif handedness == "right":

                    right_hand = hand_features

            pose_features = np.zeros(
                132,
                dtype=np.float32
            )

            if pose_result.pose_landmarks:

                pose_features = normalize_pose(
                    pose_result.pose_landmarks[0]
                )

            frame_features = np.concatenate(
                [
                    left_hand,
                    right_hand,
                    pose_features
                ]
            )

            features.append(frame_features)

    features = np.asarray(
        features,
        dtype=np.float32
    )

    return features


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("=" * 70)
print("VIDEO FLIP EXPERIMENT")
print("=" * 70)

print()
print(f"Device: {DEVICE}")

model = SignLanguageModel(
    input_size=FEATURE_SIZE,
    num_classes=len(LABELS)
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if "model_state_dict" in checkpoint:
    checkpoint = checkpoint["model_state_dict"]

model.load_state_dict(checkpoint)
model.to(DEVICE)
model.eval()


# ============================================================
# LOAD TEST SET
# ============================================================

samples = pd.read_csv(SAMPLES_PATH)
split_df = pd.read_csv(SPLIT_PATH)

test_split = split_df[
    split_df["split"] == "test"
]

test_indices = test_split["index"].astype(int).tolist()

test_samples = samples[
    samples["index"].isin(test_indices)
].copy()

test_samples = test_samples.sort_values("index")


print()
print(f"Held-out test videos: {len(test_samples)}")


# ============================================================
# PREDICT
# ============================================================

def predict(features):

    tensor = torch.tensor(
        features,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        logits = model(tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        prediction = torch.argmax(
            probabilities,
            dim=1
        ).item()

        confidence = probabilities[
            0,
            prediction
        ].item()

    return prediction, confidence


# ============================================================
# RUN BOTH CONDITIONS
# ============================================================

results = []

for row_number, row in enumerate(
    test_samples.itertuples(index=False),
    start=1
):

    video_path = row.video_path
    true_label = row.label
    signer = row.signer

    true_id = int(row.label_id)

    print()
    print(
        f"[{row_number}/{len(test_samples)}] "
        f"{signer} | {true_label}"
    )

    # --------------------------------------------------------
    # ORIGINAL
    # --------------------------------------------------------

    original_features = extract_video_features(
        video_path,
        flip=False
    )

    if original_features is None:
        print("  ERROR: original extraction failed")
        continue

    original_pred, original_conf = predict(
        original_features
    )

    # --------------------------------------------------------
    # FLIPPED
    # --------------------------------------------------------

    flipped_features = extract_video_features(
        video_path,
        flip=True
    )

    if flipped_features is None:
        print("  ERROR: flipped extraction failed")
        continue

    flipped_pred, flipped_conf = predict(
        flipped_features
    )

    original_label = LABELS[original_pred]
    flipped_label = LABELS[flipped_pred]

    original_correct = (
        original_pred == true_id
    )

    flipped_correct = (
        flipped_pred == true_id
    )

    print(
        f"  Original: "
        f"{original_label} "
        f"({original_conf * 100:.1f}%)"
        f"{' ✓' if original_correct else ''}"
    )

    print(
        f"  Flipped:  "
        f"{flipped_label} "
        f"({flipped_conf * 100:.1f}%)"
        f"{' ✓' if flipped_correct else ''}"
    )

    results.append({
        "index": row.index,
        "signer": signer,
        "true_label": true_label,
        "original_prediction": original_label,
        "original_confidence": original_conf,
        "original_correct": original_correct,
        "flipped_prediction": flipped_label,
        "flipped_confidence": flipped_conf,
        "flipped_correct": flipped_correct,
    })


# ============================================================
# SUMMARY
# ============================================================

results_df = pd.DataFrame(results)

print()
print("=" * 70)
print("FINAL RESULTS")
print("=" * 70)

original_accuracy = (
    results_df["original_correct"].mean()
)

flipped_accuracy = (
    results_df["flipped_correct"].mean()
)

print()
print(
    f"Original video accuracy : "
    f"{original_accuracy * 100:.2f}% "
    f"({results_df['original_correct'].sum()}/"
    f"{len(results_df)})"
)

print(
    f"Flipped video accuracy  : "
    f"{flipped_accuracy * 100:.2f}% "
    f"({results_df['flipped_correct'].sum()}/"
    f"{len(results_df)})"
)

print(
    f"Difference              : "
    f"{(flipped_accuracy - original_accuracy) * 100:+.2f} "
    f"percentage points"
)


# ============================================================
# SIGNER RESULTS
# ============================================================

print()
print("=" * 70)
print("BY SIGNER")
print("=" * 70)

for signer in [
    "User013",
    "User014",
    "User015"
]:

    signer_df = results_df[
        results_df["signer"] == signer
    ]

    if len(signer_df) == 0:
        continue

    original_acc = (
        signer_df["original_correct"].mean()
    )

    flipped_acc = (
        signer_df["flipped_correct"].mean()
    )

    print()
    print(signer)

    print(
        f"  Original: "
        f"{original_acc * 100:.2f}% "
        f"({signer_df['original_correct'].sum()}/"
        f"{len(signer_df)})"
    )

    print(
        f"  Flipped:  "
        f"{flipped_acc * 100:.2f}% "
        f"({signer_df['flipped_correct'].sum()}/"
        f"{len(signer_df)})"
    )


# ============================================================
# CASES WHERE FLIPPING HELPED
# ============================================================

print()
print("=" * 70)
print("CASES WHERE FLIPPING CHANGED THE RESULT")
print("=" * 70)

changed = results_df[
    results_df["original_prediction"] !=
    results_df["flipped_prediction"]
]

if len(changed) == 0:

    print()
    print("No predictions changed.")

else:

    for _, row in changed.iterrows():

        print()
        print(
            f"{row['signer']} | "
            f"True: {row['true_label']}"
        )

        print(
            f"  Original → "
            f"{row['original_prediction']}"
        )

        print(
            f"  Flipped  → "
            f"{row['flipped_prediction']}"
        )


# ============================================================
# SAVE
# ============================================================

OUTPUT = (
    "sign_language/data/processed/"
    "video_flip_results.csv"
)

results_df.to_csv(
    OUTPUT,
    index=False
)

print()
print("=" * 70)
print(f"Saved: {OUTPUT}")
print("=" * 70)
