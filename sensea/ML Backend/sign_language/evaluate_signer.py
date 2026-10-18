import os
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report

from model import SignLanguageModel
from labels import LABELS


# ============================================================
# PATHS
# ============================================================

DATA_DIR = "sign_language/data/processed"
MODEL_PATH = "sign_language/models/sign_language_model.pth"
CSV_PATH = "sign_language/data/processed/samples.csv"


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("=" * 60)
print("SIGNER-AWARE EVALUATION")
print("=" * 60)
print("DEVICE:", DEVICE)


# ============================================================
# LOAD DATA
# ============================================================

X = np.load(
    os.path.join(DATA_DIR, "X.npy")
)

y = np.load(
    os.path.join(DATA_DIR, "y.npy")
)

print("\nDataset:")
print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# LOAD CSV
# ============================================================

import pandas as pd

df = pd.read_csv(CSV_PATH)

print("\nCSV columns:")
print(df.columns.tolist())


# Make sure row count matches
if len(df) != len(X):
    raise ValueError(
        f"CSV rows ({len(df)}) do not match X samples ({len(X)})"
    )


# ============================================================
# SIGNER INFORMATION
# ============================================================

signers = df["signer"].astype(str).values

unique_signers = sorted(
    np.unique(signers)
)

print("\nUnique signers:", len(unique_signers))


# ============================================================
# CREATE SIGNER-AWARE SPLIT
# ============================================================
#
# We select approximately 20% of signers for validation.
#
# IMPORTANT:
# Very rare signers such as "unknown", "greeting",
# "islrtc_dict" etc. may represent source groups rather
# than actual individual people.
#
# Therefore, this is a DEVELOPMENT signer-group test,
# not a perfect human-independent benchmark.
# ============================================================

rng = np.random.default_rng(42)

shuffled_signers = unique_signers.copy()

rng.shuffle(shuffled_signers)

num_val_signers = max(
    1,
    int(len(shuffled_signers) * 0.20)
)

val_signers = set(
    shuffled_signers[:num_val_signers]
)

val_mask = np.array(
    [s in val_signers for s in signers]
)

val_indices = np.where(val_mask)[0]

print("\nValidation signer groups:")
for signer in sorted(val_signers):
    count = np.sum(signers == signer)
    print(f"{signer:20s} {count} samples")

print("\nValidation samples:", len(val_indices))


# ============================================================
# CHECK CLASS COVERAGE
# ============================================================

print("\nValidation class distribution:")

for class_id, label in enumerate(LABELS):

    count = np.sum(
        y[val_indices] == class_id
    )

    print(
        f"{label:15s}: {count}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

model = SignLanguageModel(
    input_size=258,
    num_classes=len(LABELS)
).to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )

model.eval()

print("\nModel loaded successfully.")


# ============================================================
# PREDICTION
# ============================================================

X_val = torch.tensor(
    X[val_indices],
    dtype=torch.float32
).to(DEVICE)

y_val = y[val_indices]


with torch.no_grad():

    outputs = model(X_val)

    predictions = torch.argmax(
        outputs,
        dim=1
    ).cpu().numpy()


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_val,
    predictions
)

print("\n")
print("=" * 60)
print("SIGNER-AWARE VALIDATION ACCURACY")
print("=" * 60)

print(
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 60)
print("PER-CLASS RESULTS")
print("=" * 60)

print(
    classification_report(
        y_val,
        predictions,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        zero_division=0
    )
)


# ============================================================
# WRONG PREDICTIONS
# ============================================================

print("\n")
print("=" * 60)
print("WRONG PREDICTIONS")
print("=" * 60)

for i in range(len(y_val)):

    if y_val[i] != predictions[i]:

        actual = LABELS[y_val[i]]
        predicted = LABELS[predictions[i]]

        signer = signers[val_indices[i]]

        print(
            f"Signer: {signer:15s} "
            f"Actual: {actual:15s} "
            f"Predicted: {predicted}"
        )


print("\n" + "=" * 60)
print("SIGNER-AWARE EVALUATION COMPLETE")
print("=" * 60)
