from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sign_language.model import SignLanguageModel
from sign_language.labels import LABELS


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

X_PATH = BASE_DIR / "data" / "processed" / "X.npy"
Y_PATH = BASE_DIR / "data" / "processed" / "y.npy"
SPLIT_PATH = BASE_DIR / "data" / "processed" / "signer_split.csv"
MODEL_PATH = BASE_DIR / "models" / "sign_language_model.pth"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# LOAD DATA
# ============================================================

X = np.load(X_PATH)
y = np.load(Y_PATH)

split_df = pd.read_csv(SPLIT_PATH)

test_df = split_df[
    (split_df["dataset"] == "ISL500") &
    (split_df["split"] == "test")
].copy()

test_indices = test_df["index"].astype(int).to_numpy()

X_test = X[test_indices]
y_test = y[test_indices]


# ============================================================
# VERIFY TEST SET
# ============================================================

expected_users = [
    "User013",
    "User014",
    "User015"
]

actual_users = sorted(
    test_df["signer"].unique().tolist()
)

assert actual_users == expected_users

assert len(test_df) == 39

print("=" * 70)
print("UNSEEN-SIGNER ISL500 TEST")
print("=" * 70)

print(f"\nDevice: {DEVICE}")

print("\nHeld-out signers:")
for user in expected_users:
    count = (test_df["signer"] == user).sum()
    print(f"  {user}: {count} videos")

print(f"\nTotal test samples: {len(test_df)}")


# ============================================================
# LOAD MODEL
# ============================================================

model = SignLanguageModel(
    input_size=258,
    num_classes=len(LABELS)
).to(DEVICE)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if "model_state_dict" in state_dict:
    state_dict = state_dict["model_state_dict"]

model.load_state_dict(state_dict)

model.eval()


# ============================================================
# INFERENCE
# ============================================================

X_tensor = torch.tensor(
    X_test,
    dtype=torch.float32
).to(DEVICE)

with torch.no_grad():

    outputs = model(X_tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    predictions = outputs.argmax(
        dim=1
    ).cpu().numpy()

    probabilities = probabilities.cpu().numpy()


# ============================================================
# OVERALL ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

correct = int(
    np.sum(predictions == y_test)
)

incorrect = len(y_test) - correct


print("\n" + "=" * 70)
print("OVERALL RESULT")
print("=" * 70)

print(
    f"\nCorrect: {correct}/{len(y_test)}"
)

print(
    f"Incorrect: {incorrect}/{len(y_test)}"
)

print(
    f"UNSEEN-SIGNER ACCURACY: {accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        predictions,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

present_labels = sorted(
    np.unique(
        np.concatenate(
            [y_test, predictions]
        )
    ).tolist()
)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=present_labels
)

present_names = [
    LABELS[i]
    for i in present_labels
]

cm_df = pd.DataFrame(
    cm,
    index=present_names,
    columns=present_names
)

print(cm_df.to_string())


# ============================================================
# PER-SIGNER ACCURACY
# ============================================================

print("\n" + "=" * 70)
print("PER-SIGNER ACCURACY")
print("=" * 70)

for signer in expected_users:

    mask = (
        test_df["signer"].to_numpy()
        == signer
    )

    signer_true = y_test[mask]
    signer_pred = predictions[mask]

    signer_accuracy = accuracy_score(
        signer_true,
        signer_pred
    )

    print(
        f"{signer}: "
        f"{np.sum(signer_true == signer_pred)}/"
        f"{len(signer_true)} "
        f"({signer_accuracy * 100:.2f}%)"
    )


# ============================================================
# WRONG PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("WRONG PREDICTIONS")
print("=" * 70)

wrong_indices = np.where(
    predictions != y_test
)[0]

if len(wrong_indices) == 0:

    print("\nNo incorrect predictions! 🎉")

else:

    for i in wrong_indices:

        true_label = LABELS[y_test[i]]
        predicted_label = LABELS[predictions[i]]

        confidence = probabilities[i][predictions[i]]

        row = test_df.iloc[i]

        print(
            f"\n{row['signer']} | "
            f"True: {true_label:12s} | "
            f"Predicted: {predicted_label:12s} | "
            f"Confidence: {confidence * 100:.2f}%"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

results = test_df.copy()

results["true_label"] = [
    LABELS[i]
    for i in y_test
]

results["predicted_label"] = [
    LABELS[i]
    for i in predictions
]

results["confidence"] = [
    float(probabilities[i][predictions[i]])
    for i in range(len(predictions))
]

results["correct"] = (
    results["true_label"]
    == results["predicted_label"]
)

output_path = (
    BASE_DIR /
    "data" /
    "processed" /
    "unseen_signer_results.csv"
)

results.to_csv(
    output_path,
    index=False
)

print("\nResults saved to:")
print(output_path)

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)
