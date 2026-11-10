from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

X = np.load(ROOT / "sign_language/data/processed/X.npy")
y = np.load(ROOT / "sign_language/data/processed/y.npy")

samples = pd.read_csv(
    ROOT / "sign_language/data/processed/samples.csv"
)

split = pd.read_csv(
    ROOT / "sign_language/data/processed/signer_split.csv"
)

df = samples.merge(
    split[["index", "split"]],
    on="index",
    how="left"
)

# ------------------------------------------------------------
# Select User014 test samples
# ------------------------------------------------------------

user014 = df[
    (df["signer"] == "User014") &
    (df["split"] == "test")
].copy()

train_df = df[df["split"] == "train"].copy()

print("=" * 70)
print("USER014 VS TRAINING FEATURE COMPARISON")
print("=" * 70)

print(f"\nUser014 samples: {len(user014)}")
print(f"Training samples: {len(train_df)}")

# ------------------------------------------------------------
# Feature groups
# ------------------------------------------------------------

# 258 features:
# left hand  = 63
# right hand = 63
# pose      = 132

GROUPS = {
    "left_hand": (0, 63),
    "right_hand": (63, 126),
    "pose": (126, 258),
}

# ------------------------------------------------------------
# Compare each User014 sample to every training sample
# ------------------------------------------------------------

results = []

for _, test_row in user014.iterrows():

    test_idx = int(test_row["index"])
    true_label = test_row["label"]

    test_x = X[test_idx]

    # Mean feature vector across time
    test_mean = test_x.mean(axis=0)

    distances = []

    for _, train_row in train_df.iterrows():

        train_idx = int(train_row["index"])
        train_x = X[train_idx]

        train_mean = train_x.mean(axis=0)

        # Overall distance
        overall = np.linalg.norm(test_mean - train_mean)

        # Group distances
        group_dist = {}

        for name, (start, end) in GROUPS.items():
            group_dist[name] = np.linalg.norm(
                test_mean[start:end] -
                train_mean[start:end]
            )

        distances.append({
            "train_index": train_idx,
            "train_label": train_row["label"],
            "overall": overall,
            **group_dist
        })

    dist_df = pd.DataFrame(distances)

    # --------------------------------------------------------
    # Nearest examples
    # --------------------------------------------------------

    nearest = dist_df.sort_values("overall").head(5)

    print("\n" + "-" * 70)
    print(f"User014 TRUE SIGN: {true_label}")
    print(f"Video: {test_row['video_path']}")

    print("\nNearest training examples:")

    print(
        nearest[
            [
                "train_label",
                "overall",
                "left_hand",
                "right_hand",
                "pose"
            ]
        ].to_string(index=False)
    )

    # --------------------------------------------------------
    # Nearest class statistics
    # --------------------------------------------------------

    class_stats = (
        dist_df
        .groupby("train_label")
        .agg(
            mean_distance=("overall", "mean"),
            min_distance=("overall", "min"),
            left_hand=("left_hand", "mean"),
            right_hand=("right_hand", "mean"),
            pose=("pose", "mean"),
        )
        .sort_values("mean_distance")
    )

    print("\nClosest classes by mean distance:")

    print(class_stats.head(5).to_string())

    # Save top class
    closest_class = class_stats.index[0]

    results.append({
        "user": "User014",
        "true_label": true_label,
        "closest_class": closest_class,
        "true_class_rank": (
            list(class_stats.index).index(true_label) + 1
            if true_label in class_stats.index
            else -1
        ),
        "true_class_mean_distance": (
            class_stats.loc[true_label, "mean_distance"]
            if true_label in class_stats.index
            else np.nan
        ),
        "closest_class_distance":
            class_stats.iloc[0]["mean_distance"],
    })


# ------------------------------------------------------------
# Save results
# ------------------------------------------------------------

out = ROOT / "sign_language/data/processed/user014_comparison.csv"

pd.DataFrame(results).to_csv(out, index=False)

print("\n" + "=" * 70)
print("COMPARISON COMPLETE")
print("=" * 70)

print(f"Saved to: {out}")
