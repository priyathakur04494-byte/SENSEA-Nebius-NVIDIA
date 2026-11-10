from pathlib import Path
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

SIGN_LANGUAGE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SIGN_LANGUAGE_DIR))

from model import SignLanguageModel
from labels import LABELS
from feature_extractor import extract_frame_features


MODEL_PATH = SIGN_LANGUAGE_DIR / "models" / "sign_language_model.pth"
SAMPLES_PATH = SIGN_LANGUAGE_DIR / "data" / "processed" / "samples.csv"

DEVICE = (
    torch.device("mps")
    if torch.backends.mps.is_available()
    else torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("cpu")
)

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print("=" * 70)
print("FATHER vs MOTHER — SIGNER DIAGNOSTIC")
print("=" * 70)

print(f"Device: {DEVICE}")

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

model = SignLanguageModel(
    input_size=checkpoint.get("input_size", 258),
    num_classes=checkpoint.get("num_classes", len(LABELS)),
).to(DEVICE)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(SAMPLES_PATH)

df = df[
    (df["dataset"] == "ISL500")
    & (df["label"].isin(["father", "mother"]))
].copy()

print(f"Father samples: {(df['label'] == 'father').sum()}")
print(f"Mother samples: {(df['label'] == 'mother').sum()}")
print(f"Total samples : {len(df)}")

# ------------------------------------------------------------
# VIDEO -> 30 x 258 FEATURES
# ------------------------------------------------------------

def extract_video_features(video_path):

    import cv2

    cap = cv2.VideoCapture(str(video_path))

    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return None

    # Uniformly sample exactly 30 frames
    if len(frames) >= 30:
        indices = np.linspace(0, len(frames) - 1, 30).astype(int)
        frames = [frames[i] for i in indices]
    else:
        last_frame = frames[-1]
        frames = frames + [last_frame] * (30 - len(frames))

    features = []

    for frame in frames:
        feat = extract_frame_features(frame)

        if feat is None:
            return None

        features.append(feat)

    return np.asarray(features, dtype=np.float32)


# ------------------------------------------------------------
# PREDICT
# ------------------------------------------------------------

results = []

for _, row in df.iterrows():

    video_path = Path(row["video_path"])

    if not video_path.is_absolute():
        video_path = Path.cwd() / video_path

    features = extract_video_features(video_path)

    if features is None:
        print(f"FAILED: {video_path.name}")
        continue

    x = torch.tensor(
        features,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

        confidence, prediction = torch.max(probs, dim=1)

    predicted_label = LABELS[prediction.item()]
    confidence_value = confidence.item()

    results.append({
        "signer": row["signer"],
        "actual": row["label"],
        "predicted": predicted_label,
        "confidence": confidence_value,
    })


# ------------------------------------------------------------
# DISPLAY RESULTS BY SIGNER
# ------------------------------------------------------------

print()
print("=" * 70)
print("RESULTS BY SIGNER")
print("=" * 70)

results_by_signer = defaultdict(dict)

for r in results:
    results_by_signer[r["signer"]][r["actual"]] = r


for signer in sorted(results_by_signer.keys()):

    signer_results = results_by_signer[signer]

    father = signer_results.get("father")
    mother = signer_results.get("mother")

    print()

    if father:
        mark = "✓" if father["predicted"] == "father" else "✗"

        print(
            f"{signer:8s}  "
            f"Father → {father['predicted']:10s} "
            f"{father['confidence'] * 100:6.2f}% {mark}"
        )

    if mother:
        mark = "✓" if mother["predicted"] == "mother" else "✗"

        print(
            f"{'':8s}  "
            f"Mother → {mother['predicted']:10s} "
            f"{mother['confidence'] * 100:6.2f}% {mark}"
        )


# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

correct_father = sum(
    r["actual"] == "father" and r["predicted"] == "father"
    for r in results
)

correct_mother = sum(
    r["actual"] == "mother" and r["predicted"] == "mother"
    for r in results
)

total_father = sum(r["actual"] == "father" for r in results)
total_mother = sum(r["actual"] == "mother" for r in results)

print(
    f"Father: {correct_father}/{total_father} "
    f"({correct_father / total_father * 100:.2f}%)"
)

print(
    f"Mother: {correct_mother}/{total_mother} "
    f"({correct_mother / total_mother * 100:.2f}%)"
)

print()

print("Confusions:")
for r in results:

    if r["actual"] != r["predicted"]:

        print(
            f"  {r['signer']:8s} "
            f"{r['actual']:7s} → "
            f"{r['predicted']:10s} "
            f"{r['confidence'] * 100:.2f}%"
        )

print()
print("=" * 70)
print("Diagnostic complete.")
print("=" * 70)
