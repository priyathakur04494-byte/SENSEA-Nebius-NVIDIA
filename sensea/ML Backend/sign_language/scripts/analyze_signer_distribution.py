from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sign_language.model import SignLanguageModel
import torch

X_PATH = ROOT / "sign_language/data/processed/X.npy"
Y_PATH = ROOT / "sign_language/data/processed/y.npy"
SPLIT_PATH = ROOT / "sign_language/data/processed/signer_split.csv"

X = np.load(X_PATH)
y = np.load(Y_PATH)

df = pd.read_csv(SPLIT_PATH)

print("=" * 70)
print("SIGNER FEATURE DISTRIBUTION ANALYSIS")
print("=" * 70)

print(f"\nX shape: {X.shape}")

# ------------------------------------------------------------------
# Join feature arrays with metadata
# ------------------------------------------------------------------

df = df.sort_values("index").reset_index(drop=True)

assert len(df) == len(X), (
    f"Mismatch: metadata={len(df)}, X={len(X)}"
)

# Overall feature statistics
flat = X.reshape(len(X), -1)

df["feature_mean"] = flat.mean(axis=1)
df["feature_std"] = flat.std(axis=1)
df["feature_norm"] = np.linalg.norm(flat, axis=1)

# ------------------------------------------------------------------
# Signer statistics
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("SIGNER STATISTICS")
print("=" * 70)

isl500 = df[df["dataset"] == "ISL500"].copy()

signer_stats = (
    isl500
    .groupby(["signer", "split"])
    .agg(
        samples=("index", "count"),
        mean_feature_mean=("feature_mean", "mean"),
        mean_feature_std=("feature_std", "mean"),
        mean_feature_norm=("feature_norm", "mean"),
    )
    .reset_index()
)

print(signer_stats.to_string(index=False))

# ------------------------------------------------------------------
# Compare User014 with each other signer
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("USER014 DISTANCE FROM OTHER SIGNERS")
print("=" * 70)

user014 = flat[
    (df["dataset"].values == "ISL500") &
    (df["signer"].values == "User014")
]

for signer in [
    "User001", "User002", "User003", "User004", "User005",
    "User006", "User007", "User008", "User009", "User010",
    "User011", "User012", "User013", "User015"
]:

    other = flat[
        (df["dataset"].values == "ISL500") &
        (df["signer"].values == signer)
    ]

    # Compare mean feature vector
    mean_014 = user014.mean(axis=0)
    mean_other = other.mean(axis=0)

    distance = np.linalg.norm(mean_014 - mean_other)

    print(f"{signer:8s} | mean-vector distance = {distance:.4f}")

# ------------------------------------------------------------------
# Compare feature statistics by label
# ------------------------------------------------------------------

print("\n" + "=" * 70)
print("USER014 VS TRAINING — PER SIGN")
print("=" * 70)

train_df = df[
    (df["dataset"] == "ISL500") &
    (df["split"] == "train")
]

for label in sorted(isl500["label"].unique()):

    u = flat[
        (df["dataset"].values == "ISL500") &
        (df["signer"].values == "User014") &
        (df["label"].values == label)
    ]

    t = flat[
        (df["dataset"].values == "ISL500") &
        (df["split"].values == "train") &
        (df["label"].values == label)
    ]

    if len(u) == 0 or len(t) == 0:
        continue

    distance = np.linalg.norm(
        u.mean(axis=0) - t.mean(axis=0)
    )

    print(
        f"{label:12s} | "
        f"User014={len(u):2d} | "
        f"Train={len(t):2d} | "
        f"distance={distance:.4f}"
    )

# ------------------------------------------------------------------
# Save statistics
# ------------------------------------------------------------------

out = ROOT / "sign_language/data/processed/signer_distribution_analysis.csv"
signer_stats.to_csv(out, index=False)

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"Saved to: {out}")
