import os
import sys
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

BASE_DIR = "sign_language"

METADATA_PATH = (
    f"{BASE_DIR}/data/raw/"
    "isl_isolated_40words/metadata.csv"
)

OUTPUT_DIR = f"{BASE_DIR}/data/processed"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

NUM_FRAMES = 30
FEATURE_SIZE = 258

# MediaPipe VIDEO mode requires timestamps to be
# strictly increasing for each detector instance.
# Since we reuse the detectors across all videos,
# keep one global timestamp counter.
GLOBAL_TIMESTAMP_MS = 0


# ============================================================
# MEDIAPIPE MODEL PATHS
# ============================================================

HAND_MODEL = (
    f"{BASE_DIR}/models/mediapipe/"
    "hand_landmarker.task"
)

POSE_MODEL = (
    f"{BASE_DIR}/models/mediapipe/"
    "pose_landmarker.task"
)


# ============================================================
# CHECK MODEL FILES
# ============================================================

if not os.path.exists(HAND_MODEL):
    print("ERROR: Hand model not found:")
    print(HAND_MODEL)
    sys.exit(1)

if not os.path.exists(POSE_MODEL):
    print("ERROR: Pose model not found:")
    print(POSE_MODEL)
    sys.exit(1)


# ============================================================
# MEDIAPIPE SETUP
# ============================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions

VisionRunningMode = mp.tasks.vision.RunningMode


hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=HAND_MODEL
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
)

pose_options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=POSE_MODEL
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
)


# ============================================================
# NORMALIZE HAND
# ============================================================

def normalize_hand(landmarks):

    points = np.array(
        [
            [lm.x, lm.y, lm.z]
            for lm in landmarks
        ],
        dtype=np.float32
    )

    # Wrist = landmark 0
    wrist = points[0].copy()

    # Move wrist to origin
    points -= wrist

    # Scale using wrist -> middle finger base
    scale = np.linalg.norm(points[9])

    if scale > 1e-6:
        points /= scale

    return points.flatten()


# ============================================================
# NORMALIZE POSE
# ============================================================

def normalize_pose(landmarks):

    points = np.array(
        [
            [
                lm.x,
                lm.y,
                lm.z,
                lm.visibility
            ]
            for lm in landmarks
        ],
        dtype=np.float32
    )

    # Shoulder center
    shoulder_center = (
        points[11, :3] +
        points[12, :3]
    ) / 2.0

    # Normalize x/y/z around shoulder center
    points[:, :3] -= shoulder_center

    # Scale using shoulder distance
    shoulder_distance = np.linalg.norm(
        points[11, :3] -
        points[12, :3]
    )

    if shoulder_distance > 1e-6:
        points[:, :3] /= shoulder_distance

    return points.flatten()


# ============================================================
# EXTRACT 258 FEATURES FROM ONE FRAME
# ============================================================

def extract_features(hand_result, pose_result):

    # --------------------------------------------------------
    # HAND FEATURES
    # --------------------------------------------------------

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
        )

        features = normalize_hand(
            hand_landmarks
        )

        if handedness.lower() == "left":
            left_hand = features

        elif handedness.lower() == "right":
            right_hand = features

    # --------------------------------------------------------
    # POSE FEATURES
    # --------------------------------------------------------

    pose_features = np.zeros(
        132,
        dtype=np.float32
    )

    if len(pose_result.pose_landmarks) > 0:

        pose_features = normalize_pose(
            pose_result.pose_landmarks[0]
        )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    features = np.concatenate(
        [
            left_hand,
            right_hand,
            pose_features
        ]
    )

    return features


# ============================================================
# EXTRACT ONE VIDEO
# ============================================================

def extract_video_features(
    video_path,
    hand_landmarker,
    pose_landmarker
):

    global GLOBAL_TIMESTAMP_MS

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames <= 0:
        cap.release()
        return None

    # --------------------------------------------------------
    # Select 30 frame positions uniformly
    # --------------------------------------------------------

    frame_indices = np.linspace(
        0,
        total_frames - 1,
        NUM_FRAMES
    ).astype(int)

    sequence = []

    current_index = 0

    target_set = set(
        frame_indices.tolist()
    )

    # --------------------------------------------------------
    # Read video
    # --------------------------------------------------------

    while True:

        success, frame = cap.read()

        if not success:
            break

        if current_index in target_set:

            # ----------------------------------------------
            # Convert BGR -> RGB
            # ----------------------------------------------

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # ----------------------------------------------
            # GLOBAL MONOTONIC TIMESTAMP
            # ----------------------------------------------

            GLOBAL_TIMESTAMP_MS += 33

            timestamp_ms = GLOBAL_TIMESTAMP_MS

            # ----------------------------------------------
            # HAND DETECTION
            # ----------------------------------------------

            hand_result = (
                hand_landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )
            )

            # ----------------------------------------------
            # POSE DETECTION
            # ----------------------------------------------

            pose_result = (
                pose_landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )
            )

            # ----------------------------------------------
            # EXTRACT FEATURES
            # ----------------------------------------------

            features = extract_features(
                hand_result,
                pose_result
            )

            # ----------------------------------------------
            # CHECK FEATURE SIZE
            # ----------------------------------------------

            if len(features) != FEATURE_SIZE:
                cap.release()
                return None

            sequence.append(features)

            # ----------------------------------------------
            # We have enough frames
            # ----------------------------------------------

            if len(sequence) >= NUM_FRAMES:
                break

        current_index += 1

    cap.release()

    # ========================================================
    # CHECK SEQUENCE
    # ========================================================

    if len(sequence) != NUM_FRAMES:
        return None

    sequence = np.array(
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
# MAIN
# ============================================================

print()
print("=" * 70)
print("SENSEA — ISL DATASET FEATURE EXTRACTION")
print("=" * 70)
print()


# ============================================================
# LOAD METADATA
# ============================================================

df = pd.read_csv(
    METADATA_PATH
)


# ============================================================
# DATASET LABELS -> MODEL LABELS
# ============================================================

DATASET_TO_MODEL_LABEL = {

    "hello": "hello",

    "goodbye": "goodbye",

    "thank you": "thank_you",

    "sorry": "sorry",

    "please": "please",

    "yes": "yes",

    "no": "no",

    "help": "help",

    "stop": "stop",

    "okay": "okay",

    "me": "me",

    "you": "you",

    "mother": "mother",

    "father": "father",

    "friend": "friend",

    "home": "home",

    "hospital": "hospital",

    "eat": "eat",

    "drink": "drink",

    "water": "water",
}


TARGET_LABELS = list(
    DATASET_TO_MODEL_LABEL.values()
)


# ============================================================
# KEEP ONLY 20 TARGET SIGNS
# ============================================================

df = df[
    df["normalized_word"].isin(
        DATASET_TO_MODEL_LABEL.keys()
    )
].copy()

df = df.reset_index(
    drop=True
)

print(
    f"Target videos: {len(df)}"
)

print()


# ============================================================
# CREATE LABEL MAPPING
# ============================================================

label_to_id = {
    label: i
    for i, label in enumerate(
        TARGET_LABELS
    )
}

print("Classes:")

for label, idx in label_to_id.items():

    print(
        f"  {idx:2} -> {label}"
    )

print()


# ============================================================
# OUTPUT ARRAYS
# ============================================================

X = []
y = []

successful = []
failed = []


# ============================================================
# RUN MEDIAPIPE
# ============================================================

with HandLandmarker.create_from_options(
    hand_options
) as hand_landmarker:

    with PoseLandmarker.create_from_options(
        pose_options
    ) as pose_landmarker:

        print("Starting extraction...")
        print()

        for index, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc="Videos"
        ):

            relative_video_path = row[
                "video_path"
            ]

            # ------------------------------------------------
            # Dataset video paths are relative to:
            #
            # sign_language/data/raw/
            # isl_isolated_40words/
            # ------------------------------------------------

            video_path = os.path.join(
                BASE_DIR,
                "data",
                "raw",
                "isl_isolated_40words",
                relative_video_path
            )

            # ------------------------------------------------
            # Check file
            # ------------------------------------------------

            if not os.path.exists(
                video_path
            ):

                failed.append(
                    {
                        "index": index,
                        "path": video_path,
                        "reason": "file_missing"
                    }
                )

                continue

            # ------------------------------------------------
            # Extract
            # ------------------------------------------------

            try:

                features = extract_video_features(
                    video_path,
                    hand_landmarker,
                    pose_landmarker
                )

                if features is None:

                    failed.append(
                        {
                            "index": index,
                            "path": video_path,
                            "reason": "extraction_failed"
                        }
                    )

                    continue

                # ------------------------------------------------
                # Convert dataset label -> model label
                # ------------------------------------------------

                dataset_label = row[
                    "normalized_word"
                ]

                label = DATASET_TO_MODEL_LABEL[
                    dataset_label
                ]

                label_id = label_to_id[
                    label
                ]

                # ------------------------------------------------
                # Store
                # ------------------------------------------------

                X.append(
                    features
                )

                y.append(
                    label_id
                )

                successful.append(
                    {
                        "index": index,
                        "label": label,
                        "label_id": label_id,
                        "video_path": video_path,
                        "signer": row["signer"],
                        "dataset": row["dataset"]
                    }
                )

            except Exception as e:

                failed.append(
                    {
                        "index": index,
                        "path": video_path,
                        "reason": str(e)
                    }
                )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

X = np.array(
    X,
    dtype=np.float32
)

y = np.array(
    y,
    dtype=np.int64
)


# ============================================================
# SAVE
# ============================================================

np.save(
    f"{OUTPUT_DIR}/X.npy",
    X
)

np.save(
    f"{OUTPUT_DIR}/y.npy",
    y
)

pd.DataFrame(
    successful
).to_csv(
    f"{OUTPUT_DIR}/samples.csv",
    index=False
)

pd.DataFrame(
    failed
).to_csv(
    f"{OUTPUT_DIR}/failed.csv",
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 70)
print("EXTRACTION COMPLETE")
print("=" * 70)
print()

print(
    "Successful:",
    len(successful)
)

print(
    "Failed:",
    len(failed)
)

print()

print(
    "X shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)

print()

print("Expected X:")

print(
    f"({len(successful)}, 30, 258)"
)

print()

print("Class distribution:")
print("-" * 40)

for label, label_id in label_to_id.items():

    count = int(
        np.sum(y == label_id)
    )

    print(
        f"{label:15} {count:4}"
    )

print()


# ============================================================
# FAILED FILE REPORT
# ============================================================

if len(failed) > 0:

    print(
        "Failed videos were saved to:"
    )

    print(
        f"{OUTPUT_DIR}/failed.csv"
    )

print()


# ============================================================
# SAVED FILES
# ============================================================

print("Saved:")

print(
    f"  {OUTPUT_DIR}/X.npy"
)

print(
    f"  {OUTPUT_DIR}/y.npy"
)

print(
    f"  {OUTPUT_DIR}/samples.csv"
)

print(
    f"  {OUTPUT_DIR}/failed.csv"
)

print()

print("=" * 70)