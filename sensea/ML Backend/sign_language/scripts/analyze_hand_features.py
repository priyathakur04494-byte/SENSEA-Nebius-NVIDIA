import numpy as np
from pathlib import Path

SIGN_LANGUAGE_DIR = Path(__file__).resolve().parent.parent
X_PATH = SIGN_LANGUAGE_DIR / "data" / "processed" / "X.npy"
Y_PATH = SIGN_LANGUAGE_DIR / "data" / "processed" / "y.npy"

# Load extracted features
X = np.load(X_PATH)
y = np.load(Y_PATH)

# Your label IDs
LABELS = [
    "hello", "goodbye", "thank_you", "sorry", "please",
    "yes", "no", "help", "stop", "okay",
    "me", "you", "mother", "father", "friend",
    "home", "hospital", "eat", "drink", "water"
]

FATHER_ID = LABELS.index("father")
MOTHER_ID = LABELS.index("mother")

father = X[y == FATHER_ID]
mother = X[y == MOTHER_ID]

print("=" * 70)
print("FATHER vs MOTHER — HAND FEATURE ANALYSIS")
print("=" * 70)

print(f"Father samples : {len(father)}")
print(f"Mother samples : {len(mother)}")

# ---------------------------------------------------------
# HAND FEATURES = first 126 dimensions
# ---------------------------------------------------------

father_hands = father[:, :, :126]
mother_hands = mother[:, :, :126]

# Average across time and samples
father_mean = father_hands.mean(axis=(0, 1))
mother_mean = mother_hands.mean(axis=(0, 1))

# Standard deviation
father_std = father_hands.std(axis=(0, 1))
mother_std = mother_hands.std(axis=(0, 1))

# Absolute difference between class means
difference = np.abs(father_mean - mother_mean)

# ---------------------------------------------------------
# Rank features by difference
# ---------------------------------------------------------

top_indices = np.argsort(difference)[::-1][:20]

print("\nTOP 20 HAND FEATURES THAT DIFFER MOST")
print("-" * 70)

for rank, idx in enumerate(top_indices, 1):

    # Each hand = 21 landmarks × 3 coordinates
    hand_id = idx // 63
    landmark_local = (idx % 63) // 3
    coordinate = ["x", "y", "z"][idx % 3]

    hand_name = "LEFT" if hand_id == 0 else "RIGHT"

    print(
        f"{rank:2d}. feature={idx:3d}  "
        f"{hand_name:5s}  "
        f"landmark={landmark_local:2d}  "
        f"{coordinate}   "
        f"diff={difference[idx]:.5f}   "
        f"father={father_mean[idx]:+.4f}   "
        f"mother={mother_mean[idx]:+.4f}"
    )

# ---------------------------------------------------------
# Compare left vs right hand
# ---------------------------------------------------------

left_diff = difference[:63].mean()
right_diff = difference[63:126].mean()

print("\nAVERAGE LEFT/RIGHT HAND DIFFERENCE")
print("-" * 70)
print(f"Left hand  : {left_diff:.6f}")
print(f"Right hand : {right_diff:.6f}")

if left_diff > right_diff:
    print("→ Left hand contains more Father/Mother separation.")
else:
    print("→ Right hand contains more Father/Mother separation.")

# ---------------------------------------------------------
# Compare each landmark
# ---------------------------------------------------------

landmark_diffs = []

for landmark in range(21):

    start = landmark * 3
    end = start + 3

    left = difference[start:end].mean()
    right = difference[63 + start:63 + end].mean()

    landmark_diffs.append((landmark, left, right))

print("\nLANDMARK DIFFERENCES")
print("-" * 70)

for landmark, left, right in sorted(
    landmark_diffs,
    key=lambda x: x[1] + x[2],
    reverse=True
)[:10]:

    print(
        f"Landmark {landmark:2d}   "
        f"left={left:.6f}   "
        f"right={right:.6f}   "
        f"combined={left + right:.6f}"
    )

print("\n" + "=" * 70)
print("Analysis complete.")
print("=" * 70)
