import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report

from sign_language.model import SignLanguageModel


BASE_DIR = Path(__file__).resolve().parent.parent

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
print("ORIENTATION ENSEMBLE DIAGNOSTIC")
print("=" * 70)
print(f"Device: {DEVICE}")


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

X = np.load(X_PATH)
y = np.load(Y_PATH)

split_df = pd.read_csv(SPLIT_PATH)

test_rows = split_df[split_df["split"] == "test"]

test_indices = test_rows["index"].astype(int).values
test_signers = test_rows["signer"].values

X_test = X[test_indices]
y_test = y[test_indices]

print(f"Total test samples: {len(X_test)}")


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

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()


# ------------------------------------------------------------
# HORIZONTAL FLIP
# ------------------------------------------------------------

def flip_features(x):
    flipped = x.copy()

    # Left hand: 63 features = 21 landmarks × 3
    left = flipped[:, :, 0:63].reshape(
        flipped.shape[0],
        flipped.shape[1],
        21,
        3
    )

    left[:, :, :, 0] *= -1

    flipped[:, :, 0:63] = left.reshape(
        flipped.shape[0],
        flipped.shape[1],
        63
    )

    # Right hand: 63 features
    right = flipped[:, :, 63:126].reshape(
        flipped.shape[0],
        flipped.shape[1],
        21,
        3
    )

    right[:, :, :, 0] *= -1

    flipped[:, :, 63:126] = right.reshape(
        flipped.shape[0],
        flipped.shape[1],
        63
    )

    # Pose: 33 landmarks × 4
    pose = flipped[:, :, 126:258].reshape(
        flipped.shape[0],
        flipped.shape[1],
        33,
        4
    )

    pose[:, :, :, 0] *= -1

    flipped[:, :, 126:258] = pose.reshape(
        flipped.shape[0],
        flipped.shape[1],
        132
    )

    return flipped


# ------------------------------------------------------------
# RUN MODEL
# ------------------------------------------------------------

X_original = torch.tensor(
    X_test,
    dtype=torch.float32,
    device=DEVICE
)

X_flipped_np = flip_features(X_test)

X_flipped = torch.tensor(
    X_flipped_np,
    dtype=torch.float32,
    device=DEVICE
)


with torch.no_grad():

    logits_original = model(X_original)
    logits_flipped = model(X_flipped)

    probs_original = torch.softmax(
        logits_original,
        dim=1
    )

    probs_flipped = torch.softmax(
        logits_flipped,
        dim=1
    )

    pred_original = torch.argmax(
        probs_original,
        dim=1
    ).cpu().numpy()

    pred_flipped = torch.argmax(
        probs_flipped,
        dim=1
    ).cpu().numpy()

    confidence_original = torch.max(
        probs_original,
        dim=1
    ).values.cpu().numpy()

    confidence_flipped = torch.max(
        probs_flipped,
        dim=1
    ).values.cpu().numpy()


# ------------------------------------------------------------
# CONFIDENCE ENSEMBLE
# ------------------------------------------------------------

use_flipped = confidence_flipped > confidence_original

pred_ensemble = np.where(
    use_flipped,
    pred_flipped,
    pred_original
)


# ------------------------------------------------------------
# OVERALL RESULTS
# ------------------------------------------------------------

original_correct = np.sum(
    pred_original == y_test
)

flipped_correct = np.sum(
    pred_flipped == y_test
)

ensemble_correct = np.sum(
    pred_ensemble == y_test
)

print("\n" + "=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(
    f"Original orientation : "
    f"{original_correct}/39 = "
    f"{original_correct / len(y_test) * 100:.2f}%"
)

print(
    f"Flipped orientation  : "
    f"{flipped_correct}/39 = "
    f"{flipped_correct / len(y_test) * 100:.2f}%"
)

print(
    f"Confidence ensemble  : "
    f"{ensemble_correct}/39 = "
    f"{ensemble_correct / len(y_test) * 100:.2f}%"
)

print(
    f"\nSamples choosing flipped: "
    f"{np.sum(use_flipped)}/39"
)


# ------------------------------------------------------------
# PER-SIGNER RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PER-SIGNER RESULTS")
print("=" * 70)

for signer in sorted(np.unique(test_signers)):

    mask = test_signers == signer

    total = np.sum(mask)

    original = np.sum(
        pred_original[mask] == y_test[mask]
    )

    flipped = np.sum(
        pred_flipped[mask] == y_test[mask]
    )

    ensemble = np.sum(
        pred_ensemble[mask] == y_test[mask]
    )

    flipped_used = np.sum(
        use_flipped[mask]
    )

    print(
        f"{signer}: "
        f"Original {original}/{total} "
        f"({original/total*100:.2f}%) | "
        f"Flipped {flipped}/{total} "
        f"({flipped/total*100:.2f}%) | "
        f"Ensemble {ensemble}/{total} "
        f"({ensemble/total*100:.2f}%) | "
        f"Flipped chosen {flipped_used}/{total}"
    )


# ------------------------------------------------------------
# SAMPLE-BY-SAMPLE
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SAMPLE-BY-SAMPLE RESULTS")
print("=" * 70)

for i in range(len(y_test)):

    print(
        f"{i+1:02d}. "
        f"{test_signers[i]:8s} | "
        f"True={y_test[i]:2d} | "
        f"Orig={pred_original[i]:2d} "
        f"({confidence_original[i]:.2f}) | "
        f"Flip={pred_flipped[i]:2d} "
        f"({confidence_flipped[i]:.2f}) | "
        f"Final={pred_ensemble[i]:2d} | "
        f"{'FLIP' if use_flipped[i] else 'ORIG'}"
    )


# ------------------------------------------------------------
# CLASSIFICATION REPORT
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("ENSEMBLE CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        pred_ensemble,
        zero_division=0
    )
)

print("=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)
