import cv2
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score

from sign_language.model import SignLanguageModel
from sign_language.labels import ID_TO_LABEL
from sign_language.scripts.feature_extractor import extract_frame_features


BASE_DIR = Path(__file__).resolve().parents[1]

X_PATH = BASE_DIR / "data" / "processed" / "X.npy"
Y_PATH = BASE_DIR / "data" / "processed" / "y.npy"
SPLIT_PATH = BASE_DIR / "data" / "processed" / "signer_split.csv"
MODEL_PATH = BASE_DIR / "models" / "sign_language_model.pth"


# ------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("=" * 70)
print("RAW VIDEO FLIP DIAGNOSTIC")
print("=" * 70)
print(f"Device: {DEVICE}")


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

X = np.load(X_PATH)
y = np.load(Y_PATH)

split_df = pd.read_csv(SPLIT_PATH)

test_df = split_df[split_df["split"] == "test"].copy()

test_indices = test_df["index"].to_numpy()

y_test = y[test_indices]

print(f"Total test samples: {len(test_indices)}")


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

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# ------------------------------------------------------------
# MODEL PREDICTION
# ------------------------------------------------------------

def predict(features):
    x = torch.tensor(
        features,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)

    pred = int(torch.argmax(probs, dim=1).item())
    confidence = float(torch.max(probs).item())

    return pred, confidence


# ------------------------------------------------------------
# EXTRACT FEATURES FROM VIDEO
# ------------------------------------------------------------

def extract_video(video_path, flip=False):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {video_path}")

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
        raise RuntimeError(f"No frames: {video_path}")

    # Uniformly sample / pad to 30 frames
    if len(frames) >= 30:
        indices = np.linspace(
            0,
            len(frames) - 1,
            30
        ).astype(int)

        frames = [frames[i] for i in indices]

    else:
        last = frames[-1]

        while len(frames) < 30:
            frames.append(last.copy())

    # New extractor instance so timestamps start cleanly
    import sign_language.scripts.feature_extractor as fe

    fe.timestamp_ms = 0

    features = []

    for frame in frames:
        f = extract_frame_features(frame)
        features.append(f)

    return np.asarray(features, dtype=np.float32)


# ------------------------------------------------------------
# RUN TEST
# ------------------------------------------------------------

pred_original = []
pred_flipped = []

conf_original = []
conf_flipped = []

signers = []
paths = []


for n, idx in enumerate(test_indices):

    row = test_df[test_df["index"] == idx].iloc[0]

    video_path = row["video_path"]
    signer = row["signer"]

    if not Path(video_path).is_absolute():
        video_path = str(BASE_DIR.parent / video_path)

    true_label = int(y[idx])

    print(
        f"\n[{n+1:02d}/39] "
        f"{signer} | "
        f"True={ID_TO_LABEL[true_label]}"
    )

    # Original
    original_features = extract_video(
        video_path,
        flip=False
    )

    original_pred, original_conf = predict(
        original_features
    )

    # Raw flipped video
    flipped_features = extract_video(
        video_path,
        flip=True
    )

    flipped_pred, flipped_conf = predict(
        flipped_features
    )

    pred_original.append(original_pred)
    pred_flipped.append(flipped_pred)

    conf_original.append(original_conf)
    conf_flipped.append(flipped_conf)

    signers.append(signer)
    paths.append(video_path)

    print(
        f"  Original: {ID_TO_LABEL[original_pred]:12s} "
        f"({original_conf:.2f})"
    )

    print(
        f"  Flipped : {ID_TO_LABEL[flipped_pred]:12s} "
        f"({flipped_conf:.2f})"
    )


# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

pred_original = np.asarray(pred_original)
pred_flipped = np.asarray(pred_flipped)

conf_original = np.asarray(conf_original)
conf_flipped = np.asarray(conf_flipped)


original_acc = accuracy_score(
    y_test,
    pred_original
)

flipped_acc = accuracy_score(
    y_test,
    pred_flipped
)


print("\n" + "=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(
    f"Original raw video : "
    f"{np.sum(pred_original == y_test)}/39 = "
    f"{original_acc * 100:.2f}%"
)

print(
    f"Flipped raw video  : "
    f"{np.sum(pred_flipped == y_test)}/39 = "
    f"{flipped_acc * 100:.2f}%"
)


# ------------------------------------------------------------
# PER-SIGNER
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PER-SIGNER RESULTS")
print("=" * 70)

signers = np.asarray(signers)

for signer in sorted(np.unique(signers)):

    mask = signers == signer

    orig_correct = np.sum(
        pred_original[mask] == y_test[mask]
    )

    flip_correct = np.sum(
        pred_flipped[mask] == y_test[mask]
    )

    total = np.sum(mask)

    print(
        f"{signer}: "
        f"Original {orig_correct}/{total} "
        f"({orig_correct / total * 100:.2f}%) | "
        f"Flipped {flip_correct}/{total} "
        f"({flip_correct / total * 100:.2f}%)"
    )


# ------------------------------------------------------------
# USER014 DETAIL
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("USER014 DETAIL")
print("=" * 70)

mask = signers == "User014"

for i in np.where(mask)[0]:

    true_name = ID_TO_LABEL[int(y_test[i])]
    orig_name = ID_TO_LABEL[int(pred_original[i])]
    flip_name = ID_TO_LABEL[int(pred_flipped[i])]

    print(
        f"{true_name:12s} | "
        f"Original={orig_name:12s} "
        f"({conf_original[i]:.2f}) | "
        f"Flipped={flip_name:12s} "
        f"({conf_flipped[i]:.2f})"
    )


print("\n" + "=" * 70)
print("RAW FLIP DIAGNOSTIC COMPLETE")
print("=" * 70)
