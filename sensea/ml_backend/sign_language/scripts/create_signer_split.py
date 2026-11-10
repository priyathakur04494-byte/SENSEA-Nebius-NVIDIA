import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "data" / "processed" / "samples.csv"
OUT_PATH = BASE_DIR / "data" / "processed" / "signer_split.csv"

df = pd.read_csv(CSV_PATH)

train_idx = []
val_idx = []
test_idx = []

# --------------------------------------------------
# ISL500: TRUE SIGNER-AWARE SPLIT
# --------------------------------------------------
isl500 = df[df["dataset"] == "ISL500"]

train_users = [f"User{i:03d}" for i in range(1, 11)]
val_users = ["User011", "User012"]
test_users = ["User013", "User014", "User015"]

for idx, row in isl500.iterrows():
    signer = row["signer"]

    if signer in train_users:
        train_idx.append(idx)
    elif signer in val_users:
        val_idx.append(idx)
    elif signer in test_users:
        test_idx.append(idx)

# --------------------------------------------------
# OTHER DATASETS
#
# They don't have reliable person-level signer IDs,
# so split them stratified by class into train/val.
# They are NOT used in the final unseen-signer test.
# --------------------------------------------------
other = df[df["dataset"] != "ISL500"].copy()

for label, group in other.groupby("label"):
    indices = group.index.tolist()

    # deterministic ordering
    indices = sorted(indices)

    n = len(indices)

    if n == 1:
        train_idx.extend(indices)

    elif n == 2:
        train_idx.append(indices[0])
        val_idx.append(indices[1])

    else:
        # roughly 80/20 train/validation
        split = max(1, int(round(n * 0.8)))

        train_idx.extend(indices[:split])
        val_idx.extend(indices[split:])

# --------------------------------------------------
# CREATE SPLIT COLUMN
# --------------------------------------------------
df["split"] = "unused"

df.loc[train_idx, "split"] = "train"
df.loc[val_idx, "split"] = "val"
df.loc[test_idx, "split"] = "test"

# Save
df.to_csv(OUT_PATH, index=False)

print("\nSIGNER-AWARE SPLIT CREATED")
print("=" * 60)

print("\nOverall:")
print(df["split"].value_counts())

print("\nISL500:")
print(
    df[df["dataset"] == "ISL500"]
    .groupby(["split", "signer"])
    .size()
)

print("\nISL500 test users:")
print(
    df[
        (df["dataset"] == "ISL500") &
        (df["split"] == "test")
    ][["label", "signer"]]
    .to_string(index=False)
)

print("\nSaved to:")
print(OUT_PATH)
