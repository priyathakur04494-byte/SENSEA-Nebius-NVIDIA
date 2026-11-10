import os
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from model import SignLanguageModel
from labels import LABELS


DATA_DIR = "sign_language/data/processed"
MODEL_PATH = "sign_language/models/sign_language_model.pth"


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("=" * 60)
print("DEVICE:", DEVICE)
print("=" * 60)


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
# RECREATE VALIDATION SPLIT
# ============================================================

val_indices = []

rng = np.random.default_rng(42)

for class_id in np.unique(y):

    indices = np.where(y == class_id)[0]
    indices = rng.permutation(indices)

    if len(indices) <= 4:

        val_indices.extend(
            indices[-1:]
        )

    else:

        val_count = max(
            1,
            int(round(len(indices) * 0.20))
        )

        val_indices.extend(
            indices[-val_count:]
        )


val_indices = np.array(val_indices)

X_val = X[val_indices]
y_val = y[val_indices]

print(
    "\nValidation samples:",
    len(X_val)
)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model = SignLanguageModel(
    input_size=258,
    num_classes=len(LABELS)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print("\nModel loaded successfully.")


# ============================================================
# MAKE PREDICTIONS
# ============================================================

X_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
).to(DEVICE)

with torch.no_grad():

    outputs = model(X_tensor)

    predictions = (
        outputs
        .argmax(dim=1)
        .cpu()
        .numpy()
    )


# ============================================================
# OVERALL ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_val,
    predictions
)

print("\n" + "=" * 60)
print("OVERALL VALIDATION ACCURACY")
print("=" * 60)

print(
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# PER-CLASS RESULTS
# ============================================================

print("\n" + "=" * 60)
print("PER-CLASS RESULTS")
print("=" * 60)

print(
    classification_report(
        y_val,
        predictions,
        labels=np.arange(len(LABELS)),
        target_names=LABELS,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_val,
    predictions,
    labels=np.arange(len(LABELS))
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("Rows = Actual")
print("Columns = Predicted")
print()

print(cm)


# ============================================================
# WRONG PREDICTIONS
# ============================================================

print("\n" + "=" * 60)
print("WRONG PREDICTIONS")
print("=" * 60)

wrong = np.where(
    y_val != predictions
)[0]

if len(wrong) == 0:

    print("No incorrect predictions!")

else:

    for i in wrong:

        actual = LABELS[
            y_val[i]
        ]

        predicted = LABELS[
            predictions[i]
        ]

        print(
            f"Actual: {actual:15s} "
            f"Predicted: {predicted}"
        )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)
