import numpy as np
import pandas as pd

X = np.load("sign_language/data/processed/X.npy")
split_df = pd.read_csv("sign_language/data/processed/signer_split.csv")

print("=" * 75)
print("USER014 HAND VISIBILITY BY SIGN")
print("=" * 75)

rows = split_df[split_df["signer"] == "User014"]

for _, row in rows.iterrows():

    idx = int(row["index"])
    label = row["label"]

    sample = X[idx]

    left = sample[:, 0:63]
    right = sample[:, 63:126]

    left_frames = np.any(np.abs(left) > 1e-8, axis=1)
    right_frames = np.any(np.abs(right) > 1e-8, axis=1)

    left_count = int(left_frames.sum())
    right_count = int(right_frames.sum())

    print(
        f"{label:12s} | "
        f"Left hand: {left_count:2d}/30 frames "
        f"({left_count / 30 * 100:5.1f}%) | "
        f"Right hand: {right_count:2d}/30 frames "
        f"({right_count / 30 * 100:5.1f}%)"
    )

print("\n" + "=" * 75)
print("DONE")
print("=" * 75)
