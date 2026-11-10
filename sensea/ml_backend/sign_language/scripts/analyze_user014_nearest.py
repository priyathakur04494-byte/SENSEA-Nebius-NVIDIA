import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

X = np.load("sign_language/data/processed/X.npy")
y = np.load("sign_language/data/processed/y.npy")
split_df = pd.read_csv("sign_language/data/processed/signer_split.csv")

print("=" * 70)
print("USER014 NEAREST-NEIGHBOR FEATURE ANALYSIS")
print("=" * 70)

# Flatten each 30-frame sequence into one vector
X_flat = X.reshape(len(X), -1)

for signer in ["User013", "User014", "User015"]:

    print(f"\n{'-' * 70}")
    print(f"SIGNER: {signer}")
    print(f"{'-' * 70}")

    signer_rows = split_df[split_df["signer"] == signer]

    correct_nearest = 0

    for _, row in signer_rows.iterrows():

        idx = int(row["index"])
        true_label = row["label"]

        # Compare against training signers only
        train_rows = split_df[split_df["split"] == "train"]
        train_indices = train_rows["index"].astype(int).values

        distances = pairwise_distances(
            X_flat[idx:idx+1],
            X_flat[train_indices],
            metric="euclidean"
        )[0]

        nearest_order = np.argsort(distances)[:5]

        nearest_labels = [
            train_rows.iloc[int(i)]["label"]
            for i in nearest_order
        ]

        nearest_distances = [
            float(distances[int(i)])
            for i in nearest_order
        ]

        nearest_correct = nearest_labels[0] == true_label

        if nearest_correct:
            correct_nearest += 1

        print(
            f"{true_label:12s} | "
            f"Nearest: {nearest_labels[0]:12s} "
            f"({nearest_distances[0]:.2f}) | "
            f"Top5: {', '.join(nearest_labels)}"
        )

    print(
        f"\nNearest-neighbor accuracy: "
        f"{correct_nearest}/{len(signer_rows)} "
        f"({100 * correct_nearest / len(signer_rows):.2f}%)"
    )
