import sys
from pathlib import Path

import cv2
import numpy as np
import torch

SIGN_LANGUAGE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SIGN_LANGUAGE_DIR))

from scripts.feature_extractor import extract_frame_features
from model import SignLanguageModel
from labels import ID_TO_LABEL

NUM_FRAMES = 30

MODEL_PATH = SIGN_LANGUAGE_DIR / "models" / "sign_language_model.pth"


def read_video(video_path):
    cap = cv2.VideoCapture(str(video_path))

    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()

    return frames


def sample_frames(frames):

    if len(frames) == 0:
        return []

    if len(frames) > NUM_FRAMES:

        indices = np.linspace(
            0,
            len(frames) - 1,
            NUM_FRAMES
        ).astype(int)

        return [frames[i] for i in indices]

    while len(frames) < NUM_FRAMES:
        frames.append(frames[-1])

    return frames


def extract_video(video_path):

    frames = read_video(video_path)
    frames = sample_frames(frames)

    if len(frames) != NUM_FRAMES:
        return None

    features = []

    for frame in frames:

        f = extract_frame_features(frame)

        if f is None:
            return None

        features.append(f)

    return np.asarray(features, dtype=np.float32)


def predict(model, X):

    tensor = torch.tensor(X, dtype=torch.float32)

    device = next(model.parameters()).device
    tensor = tensor.to(device)

    with torch.no_grad():

        logits = model(tensor.unsqueeze(0))

        probabilities = torch.softmax(logits, dim=1)[0]

        confidence, predicted = torch.max(probabilities, dim=0)

    return (
        ID_TO_LABEL[predicted.item()],
        confidence.item()
    )


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)

model = SignLanguageModel(
    input_size=258,
    num_classes=20
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()


# --------------------------------------------------
# TEST VIDEOS
# --------------------------------------------------

father_dir = (
    SIGN_LANGUAGE_DIR
    / "data"
    / "raw"
    / "isl_isolated_40words"
    / "father"
)

mother_dir = (
    SIGN_LANGUAGE_DIR
    / "data"
    / "raw"
    / "isl_isolated_40words"
    / "mother"
)


videos = [
    ("father", sorted(father_dir.glob("*.mp4"))),
    ("mother", sorted(mother_dir.glob("*.mp4"))),
]


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

results = {
    "full": [],
    "hands_only": [],
    "pose_only": []
}


for true_label, video_list in videos:

    print()
    print("=" * 75)
    print(true_label.upper())
    print("=" * 75)

    for video in video_list:

        X = extract_video(video)

        if X is None:
            continue

        # -----------------------------
        # FULL FEATURES
        # -----------------------------

        pred_full, conf_full = predict(
            model,
            X
        )

        # -----------------------------
        # HANDS ONLY
        # -----------------------------

        X_hands = X.copy()

        # Zero pose features
        X_hands[:, 126:] = 0

        pred_hands, conf_hands = predict(
            model,
            X_hands
        )

        # -----------------------------
        # POSE ONLY
        # -----------------------------

        X_pose = X.copy()

        # Zero hand features
        X_pose[:, :126] = 0

        pred_pose, conf_pose = predict(
            model,
            X_pose
        )

        results["full"].append(
            (true_label, pred_full)
        )

        results["hands_only"].append(
            (true_label, pred_hands)
        )

        results["pose_only"].append(
            (true_label, pred_pose)
        )

        print(
            f"{video.name[:38]:38s} "
            f"FULL={pred_full:10s} "
            f"HANDS={pred_hands:10s} "
            f"POSE={pred_pose:10s}"
        )


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print()
print("=" * 75)
print("ABLATION SUMMARY")
print("=" * 75)


for mode, predictions in results.items():

    correct = sum(
        true == pred
        for true, pred in predictions
    )

    total = len(predictions)

    accuracy = (
        correct / total * 100
        if total
        else 0
    )

    print(
        f"{mode:15s}: "
        f"{correct}/{total} "
        f"({accuracy:.2f}%)"
    )


print()
print("Diagnostic complete.")
