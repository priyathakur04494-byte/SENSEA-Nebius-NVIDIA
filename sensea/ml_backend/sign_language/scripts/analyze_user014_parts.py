import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

X = np.load("sign_language/data/processed/X.npy")
split_df = pd.read_csv("sign_language/data/processed/signer_split.csv")

train_rows = split_df[split_df["split"] == "train"]
train_indices = train_rows["index"].astype(int).values

# Feature blocks
LEFT_HAND = slice(0, 63)
RIGHT_HAND = slice(63, 126)
POSE = slice(126, 258)

PARTS = {
    "left_hand": LEFT_HAND,
    "right_hand": RIGHT_HAND,
    "pose": POSE,
    "hands": slice(0, 126),
    "full": slice(0, 258),
}

for signer in ["User013", "User014", "User015"]:

    print("\n" + "=" * 70)
    print(f"SIGNER: {signer}")
    print("=" * 70)

    signer_rows = split_df[split_df["signer"] == signer]

    for part_name, part_slice in PARTS.items():

        correct = 0
        total = len(signer_rows)

        print(f"\n--- {part_name.upper()} ---")

        for _, row in signer_rows.iterrows():

            idx = int(row["index"])
            true_label = row["label"]

            query = X[idx:idx+1, :, part_slice].reshape(1, -1)
            train_data = X[train_indices, :, part_slice].reshape(
                len(train_indices), -1
            )

            distances = pairwise_distances(
                query,
                train_data,
                metric="euclidean"
            )[0]

            nearest = np.argmin(distances)

            nearest_label = train_rows.iloc[int(nearest)]["label"]

            if nearest_label == true_label:
                correct += 1

        accuracy = 100 * correct / total

        print(
            f"Nearest-neighbor accuracy: "
            f"{correct}/{total} ({accuracy:.2f}%)"
        )

    print()
