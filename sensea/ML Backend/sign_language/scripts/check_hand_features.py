import numpy as np
import pandas as pd

X = np.load("sign_language/data/processed/X.npy")
split_df = pd.read_csv("sign_language/data/processed/signer_split.csv")

print("=" * 70)
print("HAND FEATURE NON-ZERO CHECK")
print("=" * 70)

for split in ["train", "val", "test"]:

    print(f"\n--- {split.upper()} ---")

    rows = split_df[split_df["split"] == split]

    for signer in sorted(rows["signer"].unique()):

        indices = rows[rows["signer"] == signer]["index"].astype(int).values

        samples = X[indices]

        left = samples[:, :, 0:63]
        right = samples[:, :, 63:126]

        left_nonzero = np.sum(np.abs(left) > 1e-8)
        right_nonzero = np.sum(np.abs(right) > 1e-8)

        total_left = left.size
        total_right = right.size

        print(
            f"{signer:8s} | "
            f"Left non-zero: {left_nonzero/total_left*100:6.2f}% | "
            f"Right non-zero: {right_nonzero/total_right*100:6.2f}%"
        )

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
