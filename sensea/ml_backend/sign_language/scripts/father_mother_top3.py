from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd
import torch

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from model import SignLanguageModel
from labels import LABELS
from feature_extractor import extract_frame_features

MODEL_PATH = BASE_DIR / "models" / "sign_language_model.pth"
SAMPLES_PATH = BASE_DIR / "data" / "processed" / "samples.csv"

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

model = SignLanguageModel(
    input_size=checkpoint.get("input_size", 258),
    num_classes=checkpoint.get("num_classes", len(LABELS))
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(SAMPLES_PATH)

df = df[
    (df["dataset"] == "ISL500") &
    (df["label"].isin(["father", "mother"]))
].copy()

# ------------------------------------------------------------
# VIDEO FEATURES
# ------------------------------------------------------------

def extract_video_features(video_path):

    cap = cv2.VideoCapture(str(video_path))

    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()

    if not frames:
        return None

    if len(frames) >= 30:
        indices = np.linspace(
            0,
            len(frames) - 1,
            30
        ).astype(int)

        frames = [frames[i] for i in indices]

    else:
        frames += [frames[-1]] * (30 - len(frames))

    features = []

    for frame in frames:

        feature = extract_frame_features(frame)

        if feature is None:
            return None

        features.append(feature)

    return np.asarray(
        features,
        dtype=np.float32
    )


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

print("=" * 75)
print("FATHER vs MOTHER — TOP-3 MODEL PREDICTIONS")
print("=" * 75)

for _, row in df.iterrows():

    features = extract_video_features(row["video_path"])

    if features is None:
        print("FAILED:", row["video_path"])
        continue

    x = torch.tensor(
        features,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        logits = model(x)

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

    top_probs, top_indices = torch.topk(
        probabilities,
        3
    )

    print()
    print(
        f"{row['signer']} | "
        f"ACTUAL: {row['label']}"
    )

    for rank, (prob, idx) in enumerate(
        zip(top_probs, top_indices),
        1
    ):

        label = LABELS[idx.item()]

        print(
            f"  {rank}. "
            f"{label:10s} "
            f"{prob.item() * 100:6.2f}%"
        )

print()
print("=" * 75)
print("COMPLETE")
print("=" * 75)
