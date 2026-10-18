from pathlib import Path
import numpy as np
import pandas as pd

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

X_PATH = BASE_DIR / "data" / "processed" / "X.npy"
Y_PATH = BASE_DIR / "data" / "processed" / "y.npy"
CSV_PATH = BASE_DIR / "data" / "processed" / "samples.csv"

# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------

X = np.load(X_PATH)
y = np.load(Y_PATH)
df = pd.read_csv(CSV_PATH)

print("=" * 70)
print("FATHER vs MOTHER FEATURE ANALYSIS")
print("=" * 70)

print(f"X shape: {X.shape}")

father_id = df[df["label"] == "father"]["label_id"].iloc[0]
mother_id = df[df["label"] == "mother"]["label_id"].iloc[0]

father_mask = y == father_id
mother_mask = y == mother_id

father = X[father_mask]
mother = X[mother_mask]

print(f"Father samples: {len(father)}")
print(f"Mother samples: {len(mother)}")

# ------------------------------------------------------------
# FLATTEN TEMPORAL FEATURES
# ------------------------------------------------------------

father_flat = father.reshape(len(father), -1)
mother_flat = mother.reshape(len(mother), -1)

father_mean = father_flat.mean(axis=0)
mother_mean = mother_flat.mean(axis=0)

difference = np.abs(father_mean - mother_mean)

# ------------------------------------------------------------
# TOP DIFFERENT FEATURES
# ------------------------------------------------------------

print()
print("=" * 70)
print("TOP 30 MOST DIFFERENT FEATURES")
print("=" * 70)

top_indices = np.argsort(difference)[::-1][:30]

for rank, idx in enumerate(top_indices, 1):

    frame = idx // 258
    feature = idx % 258

    print(
        f"{rank:2d}. "
        f"frame={frame:2d} "
        f"feature={feature:3d} "
        f"father={father_mean[idx]: .5f} "
        f"mother={mother_mean[idx]: .5f} "
        f"diff={difference[idx]: .5f}"
    )

# ------------------------------------------------------------
# HAND vs POSE
# ------------------------------------------------------------

print()
print("=" * 70)
print("HAND vs POSE DIFFERENCE")
print("=" * 70)

# 126 hand features
hand_diff = difference.reshape(30, 258)[:, :126]

# 132 pose features
pose_diff = difference.reshape(30, 258)[:, 126:]

print(
    f"Average HAND difference: "
    f"{hand_diff.mean():.6f}"
)

print(
    f"Average POSE difference: "
    f"{pose_diff.mean():.6f}"
)

print(
    f"Maximum HAND difference: "
    f"{hand_diff.max():.6f}"
)

print(
    f"Maximum POSE difference: "
    f"{pose_diff.max():.6f}"
)

# ------------------------------------------------------------
# LEFT vs RIGHT HAND
# ------------------------------------------------------------

print()
print("=" * 70)
print("LEFT vs RIGHT HAND")
print("=" * 70)

# Each hand = 63 features
left_diff = difference.reshape(30, 258)[:, :63]
right_diff = difference.reshape(30, 258)[:, 63:126]

print(
    f"Left hand average difference : "
    f"{left_diff.mean():.6f}"
)

print(
    f"Right hand average difference: "
    f"{right_diff.mean():.6f}"
)

# ------------------------------------------------------------
# TEMPORAL DIFFERENCE
# ------------------------------------------------------------

print()
print("=" * 70)
print("TEMPORAL DIFFERENCE")
print("=" * 70)

frame_diff = difference.reshape(30, 258).mean(axis=1)

for frame, value in enumerate(frame_diff):
    print(
        f"Frame {frame:02d}: {value:.6f}"
    )

# ------------------------------------------------------------
# DATASET BREAKDOWN
# ------------------------------------------------------------

print()
print("=" * 70)
print("FATHER / MOTHER BY DATASET")
print("=" * 70)

subset = df[df["label"].isin(["father", "mother"])]

print(
    pd.crosstab(
        subset["dataset"],
        subset["label"]
    )
)

print()
print("=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
