import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score

from sign_language.labels import ID_TO_LABEL
from sign_language.model import SignLanguageModel
from sign_language.scripts.feature_extractor import extract_frame_features


# ------------------------------------------------------------
# PATHS / DEVICE
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

X_PATH = BASE_DIR / "data" / "processed" / "X.npy"
Y_PATH = BASE_DIR / "data" / "processed" / "y.npy"
SPLIT_PATH = BASE_DIR / "data" / "processed" / "signer_split.csv"
MODEL_PATH = BASE_DIR / "models" / "sign_language_model.pth"

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

X = np.load(X_PATH)
y = np.load(Y_PATH)

split_df = pd.read_csv(SPLIT_PATH)

test_df = split_df[
    split_df["split"] == "test"
].copy()

test_indices = test_df["index"].astype(int).tolist()

y_test = y[test_indices]


# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

model = SignLanguageModel(
    input_size=258,
    num_classes=20
).to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


print("=" * 70)
print("AUTOMATIC ORIENTATION GATE TEST")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Total test samples: {len(test_indices)}")
print()
print("Gate rule:")
print("  Original right-hand detection < 1%  -> FLIP")
print("  Otherwise                         -> ORIGINAL")
print()


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

def predict(features):

    x = torch.tensor(
        features,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        logits = model(x)

        probs = torch.softmax(
            logits,
            dim=1
        )

    pred = int(
        torch.argmax(
            probs,
            dim=1
        ).item()
    )

    confidence = float(
        torch.max(probs).item()
    )

    return pred, confidence


# ------------------------------------------------------------
# EXTRACT VIDEO
# ------------------------------------------------------------

def extract_video(video_path, flip=False):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open: {video_path}"
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
        raise RuntimeError(
            f"No frames: {video_path}"
        )

    # --------------------------------------------------------
    # Uniform sample / pad to 30 frames
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Reset MediaPipe timestamps
    # --------------------------------------------------------

    import sign_language.scripts.feature_extractor as fe

    fe.timestamp_ms = 0

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    features = []

    for frame in frames:

        f = extract_frame_features(
            frame
        )

        features.append(f)

    return np.asarray(
        features,
        dtype=np.float32
    )


# ------------------------------------------------------------
# MEASURE ORIENTATION
# ------------------------------------------------------------

def measure_right_hand_percentage(video_path):

    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    # --------------------------------------------------------
    # Create a completely separate HandLandmarker instance.
    # This avoids timestamp conflicts with feature_extractor.py.
    # --------------------------------------------------------

    hand_model = (
        BASE_DIR /
        "models" /
        "mediapipe" /
        "hand_landmarker.task"
    )

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(hand_model)
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

    cap = cv2.VideoCapture(
        str(video_path)
    )

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

    # --------------------------------------------------------
    # Same 30-frame sampling as the classifier
    # --------------------------------------------------------

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

    right_frames = 0

    # --------------------------------------------------------
    # Detect hands using the independent detector
    # --------------------------------------------------------

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


# ------------------------------------------------------------
# IMPORT MEDIAPIPE
# ------------------------------------------------------------

import mediapipe as mp


# ------------------------------------------------------------
# RUN TEST
# ------------------------------------------------------------

pred_original = []
pred_flipped = []
pred_gate = []

conf_original = []
conf_flipped = []
conf_gate = []

gate_flip = []

signers = []
paths = []
right_percentages = []


for n, idx in enumerate(test_indices):

    row = test_df[
        test_df["index"] == idx
    ].iloc[0]

    video_path = row["video_path"]

    signer = row["signer"]

    if not Path(video_path).is_absolute():

        video_path = str(
            BASE_DIR.parent /
            video_path
        )

    true_label = int(
        y[idx]
    )

    true_name = ID_TO_LABEL[
        true_label
    ]

    print()
    print(
        f"[{n+1:02d}/{len(test_indices)}] "
        f"{signer} | "
        f"True={true_name}"
    )

    # --------------------------------------------------------
    # Measure orientation
    # --------------------------------------------------------

    right_pct = measure_right_hand_percentage(
        video_path
    )

    should_flip = (
        right_pct < 1.0
    )

    # --------------------------------------------------------
    # Original
    # --------------------------------------------------------

    original_features = extract_video(
        video_path,
        flip=False
    )

    original_pred, original_conf = predict(
        original_features
    )

    # --------------------------------------------------------
    # Flipped
    # --------------------------------------------------------

    flipped_features = extract_video(
        video_path,
        flip=True
    )

    flipped_pred, flipped_conf = predict(
        flipped_features
    )

    # --------------------------------------------------------
    # Gate
    # --------------------------------------------------------

    if should_flip:

        gate_pred = flipped_pred
        gate_conf = flipped_conf

        decision = "FLIP"

    else:

        gate_pred = original_pred
        gate_conf = original_conf

        decision = "ORIGINAL"

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    pred_original.append(
        original_pred
    )

    pred_flipped.append(
        flipped_pred
    )

    pred_gate.append(
        gate_pred
    )

    conf_original.append(
        original_conf
    )

    conf_flipped.append(
        flipped_conf
    )

    conf_gate.append(
        gate_conf
    )

    gate_flip.append(
        should_flip
    )

    signers.append(
        signer
    )

    paths.append(
        video_path
    )

    right_percentages.append(
        right_pct
    )

    print(
        f"  Right-hand detection: "
        f"{right_pct:.2f}%"
    )

    print(
        f"  Gate decision: {decision}"
    )

    print(
        f"  Original: "
        f"{ID_TO_LABEL[original_pred]:12s} "
        f"({original_conf:.2f})"
    )

    print(
        f"  Flipped : "
        f"{ID_TO_LABEL[flipped_pred]:12s} "
        f"({flipped_conf:.2f})"
    )

    print(
        f"  Gate    : "
        f"{ID_TO_LABEL[gate_pred]:12s} "
        f"({gate_conf:.2f})"
    )


# ------------------------------------------------------------
# ARRAYS
# ------------------------------------------------------------

pred_original = np.asarray(
    pred_original
)

pred_flipped = np.asarray(
    pred_flipped
)

pred_gate = np.asarray(
    pred_gate
)

gate_flip = np.asarray(
    gate_flip
)

signers = np.asarray(
    signers
)

right_percentages = np.asarray(
    right_percentages
)


# ------------------------------------------------------------
# ACCURACY
# ------------------------------------------------------------

original_acc = accuracy_score(
    y_test,
    pred_original
)

flipped_acc = accuracy_score(
    y_test,
    pred_flipped
)

gate_acc = accuracy_score(
    y_test,
    pred_gate
)


# ------------------------------------------------------------
# OVERALL
# ------------------------------------------------------------

print()
print("=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(
    f"Original : "
    f"{np.sum(pred_original == y_test)}/{len(y_test)} "
    f"= {original_acc * 100:.2f}%"
)

print(
    f"Flipped  : "
    f"{np.sum(pred_flipped == y_test)}/{len(y_test)} "
    f"= {flipped_acc * 100:.2f}%"
)

print(
    f"Gate     : "
    f"{np.sum(pred_gate == y_test)}/{len(y_test)} "
    f"= {gate_acc * 100:.2f}%"
)

print(
    f"\nGate flipped "
    f"{np.sum(gate_flip)}/{len(gate_flip)} samples"
)


# ------------------------------------------------------------
# PER-SIGNER
# ------------------------------------------------------------

print()
print("=" * 70)
print("PER-SIGNER RESULTS")
print("=" * 70)

for signer in sorted(
    np.unique(signers)
):

    mask = (
        signers == signer
    )

    total = np.sum(mask)

    orig_correct = np.sum(
        pred_original[mask] ==
        y_test[mask]
    )

    flip_correct = np.sum(
        pred_flipped[mask] ==
        y_test[mask]
    )

    gate_correct = np.sum(
        pred_gate[mask] ==
        y_test[mask]
    )

    gate_count = np.sum(
        gate_flip[mask]
    )

    print(
        f"{signer}: "
        f"Original {orig_correct}/{total} "
        f"({orig_correct/total*100:.2f}%) | "
        f"Flipped {flip_correct}/{total} "
        f"({flip_correct/total*100:.2f}%) | "
        f"Gate {gate_correct}/{total} "
        f"({gate_correct/total*100:.2f}%) | "
        f"Gate flips {gate_count}/{total}"
    )


# ------------------------------------------------------------
# SAVE RESULTS
# ------------------------------------------------------------

results = pd.DataFrame({

    "index": test_indices,

    "signer": signers,

    "true_label": [
        ID_TO_LABEL[int(v)]
        for v in y_test
    ],

    "right_hand_pct":
        right_percentages,

    "gate_flip":
        gate_flip,

    "original_prediction": [
        ID_TO_LABEL[int(v)]
        for v in pred_original
    ],

    "original_confidence":
        conf_original,

    "flipped_prediction": [
        ID_TO_LABEL[int(v)]
        for v in pred_flipped
    ],

    "flipped_confidence":
        conf_flipped,

    "gate_prediction": [
        ID_TO_LABEL[int(v)]
        for v in pred_gate
    ],

    "gate_confidence":
        conf_gate,

})


OUTPUT_PATH = (
    BASE_DIR /
    "data" /
    "processed" /
    "orientation_gate_results.csv"
)

results.to_csv(
    OUTPUT_PATH,
    index=False
)

print()
print(
    f"Results saved to:\n"
    f"{OUTPUT_PATH}"
)

print()
print("=" * 70)
print("ORIENTATION GATE TEST COMPLETE")
print("=" * 70)
