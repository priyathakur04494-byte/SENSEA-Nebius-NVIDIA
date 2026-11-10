import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report

from sign_language.labels import LABELS
from sign_language.model import SignLanguageModel


# ============================================================
# PATHS
# ============================================================

X_PATH = "sign_language/data/processed/X.npy"
Y_PATH = "sign_language/data/processed/y.npy"
SPLIT_PATH = "sign_language/data/processed/signer_split.csv"
MODEL_PATH = "sign_language/models/sign_language_model.pth"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("HAND-BLOCK SWAP EXPERIMENT")
print("=" * 70)

X = np.load(X_PATH)
y = np.load(Y_PATH)

split_df = pd.read_csv(SPLIT_PATH)

print()
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


# ============================================================
# GET HELD-OUT TEST INDICES
# ============================================================

test_indices = split_df[
    split_df["split"] == "test"
]["index"].values

test_indices = test_indices.astype(int)

X_test = X[test_indices]
y_test = y[test_indices]

print()
print(f"Held-out test samples: {len(X_test)}")


# ============================================================
# LOAD MODEL
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print(f"Device: {DEVICE}")

model = SignLanguageModel(
    input_size=258,
    num_classes=20
)

state_dict = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

# Handle checkpoint format
if "model_state_dict" in state_dict:
    state_dict = state_dict["model_state_dict"]

model.load_state_dict(state_dict)

model.to(DEVICE)
model.eval()


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(X_input):

    tensor = torch.tensor(
        X_input,
        dtype=torch.float32
    ).to(DEVICE)

    with torch.no_grad():

        logits = model(tensor)

        predictions = torch.argmax(
            logits,
            dim=1
        )

    return predictions.cpu().numpy()


# ============================================================
# ORIGINAL PREDICTIONS
# ============================================================

print()
print("Running ORIGINAL features...")

original_predictions = predict(X_test)

original_accuracy = accuracy_score(
    y_test,
    original_predictions
)


# ============================================================
# SWAP LEFT/RIGHT HAND BLOCKS
# ============================================================

print()
print("Running SWAPPED hand features...")

X_swapped = X_test.copy()

left_hand = X_test[:, :, 0:63].copy()
right_hand = X_test[:, :, 63:126].copy()

X_swapped[:, :, 0:63] = right_hand
X_swapped[:, :, 63:126] = left_hand

swapped_predictions = predict(X_swapped)

swapped_accuracy = accuracy_score(
    y_test,
    swapped_predictions
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("ACCURACY COMPARISON")
print("=" * 70)

print()
print(
    f"Original accuracy : "
    f"{original_accuracy * 100:.2f}% "
    f"({np.sum(original_predictions == y_test)}/{len(y_test)})"
)

print(
    f"Swapped accuracy  : "
    f"{swapped_accuracy * 100:.2f}% "
    f"({np.sum(swapped_predictions == y_test)}/{len(y_test)})"
)

print()

difference = (
    swapped_accuracy -
    original_accuracy
) * 100

print(
    f"Difference        : "
    f"{difference:+.2f} percentage points"
)


# ============================================================
# SIGNER-BY-SIGNER RESULTS
# ============================================================

print()
print("=" * 70)
print("SIGNER-BY-SIGNER")
print("=" * 70)

test_signers = split_df[
    split_df["split"] == "test"
].copy()

test_signers = test_signers[
    test_signers["index"].isin(test_indices)
].sort_values("index")

for signer in ["User013", "User014", "User015"]:

    mask = (
        test_signers["signer"].values ==
        signer
    )

    signer_y = y_test[mask]

    signer_original = original_predictions[mask]

    signer_swapped = swapped_predictions[mask]

    original_acc = accuracy_score(
        signer_y,
        signer_original
    )

    swapped_acc = accuracy_score(
        signer_y,
        signer_swapped
    )

    print()
    print(signer)

    print(
        f"  Original: "
        f"{original_acc * 100:.2f}% "
        f"({np.sum(signer_original == signer_y)}/{len(signer_y)})"
    )

    print(
        f"  Swapped:  "
        f"{swapped_acc * 100:.2f}% "
        f"({np.sum(signer_swapped == signer_y)}/{len(signer_y)})"
    )


# ============================================================
# SWAPPED CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("SWAPPED FEATURE CLASSIFICATION REPORT")
print("=" * 70)

present_labels = sorted(
    np.unique(
        np.concatenate(
            [y_test, swapped_predictions]
        )
    )
)

target_names = [
    LABELS[i]
    for i in present_labels
]

print(
    classification_report(
        y_test,
        swapped_predictions,
        labels=present_labels,
        target_names=target_names,
        zero_division=0
    )
)

print("=" * 70)
print("DONE")
print("=" * 70)
