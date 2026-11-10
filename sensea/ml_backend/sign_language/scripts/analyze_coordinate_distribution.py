import numpy as np
import pandas as pd

X = np.load("sign_language/data/processed/X.npy")
split_df = pd.read_csv("sign_language/data/processed/signer_split.csv")

# Feature layout:
# Left hand  = 0:63   -> 21 landmarks × (x,y,z)
# Right hand = 63:126 -> 21 landmarks × (x,y,z)
# Pose       = 126:258 -> 33 landmarks × (x,y,z,visibility)

print("=" * 75)
print("COORDINATE DISTRIBUTION ANALYSIS")
print("=" * 75)

signers = ["User013", "User014", "User015"]

parts = {
    "Left hand": (0, 63, 3),
    "Right hand": (63, 126, 3),
    "Pose": (126, 258, 4),
}

for signer in signers:

    indices = split_df[
        split_df["signer"] == signer
    ]["index"].astype(int).values

    data = X[indices]

    print("\n" + "=" * 75)
    print(f"SIGNER: {signer}")
    print(f"Samples: {len(indices)}")
    print("=" * 75)

    for part_name, (start, end, stride) in parts.items():

        part = data[:, :, start:end]

        # Only coordinate channels, excluding pose visibility
        coords = part[:, :, :stride]

        print(f"\n--- {part_name} ---")

        print(
            f"Mean: {coords.mean():.4f} | "
            f"Std: {coords.std():.4f} | "
            f"Min: {coords.min():.4f} | "
            f"Max: {coords.max():.4f}"
        )

        print(
            f"X mean: {coords[:, :, 0].mean():.4f} | "
            f"X std:  {coords[:, :, 0].std():.4f}"
        )

        print(
            f"Y mean: {coords[:, :, 1].mean():.4f} | "
            f"Y std:  {coords[:, :, 1].std():.4f}"
        )

        print(
            f"Z mean: {coords[:, :, 2].mean():.4f} | "
            f"Z std:  {coords[:, :, 2].std():.4f}"
        )

    # Overall feature statistics
    print("\n--- FULL FEATURE VECTOR ---")

    print(
        f"Mean: {data.mean():.4f} | "
        f"Std: {data.std():.4f} | "
        f"Min: {data.min():.4f} | "
        f"Max: {data.max():.4f}"
    )

print("\n" + "=" * 75)
print("DONE")
print("=" * 75)
