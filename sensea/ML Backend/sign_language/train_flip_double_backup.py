from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

from sign_language.model import SignLanguageModel
from sign_language.labels import ID_TO_LABEL


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

X_PATH = ROOT / "sign_language" / "data" / "processed" / "X.npy"
Y_PATH = ROOT / "sign_language" / "data" / "processed" / "y.npy"
SPLIT_PATH = ROOT / "sign_language" / "data" / "processed" / "signer_split.csv"

MODEL_PATH = ROOT / "sign_language" / "models" / "sign_language_model.pth"


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("=" * 70)
print("SIGN LANGUAGE TRAINING — SIGNER ROBUST VERSION")
print("=" * 70)
print(f"DEVICE: {DEVICE}")


# ============================================================
# LOAD DATA
# ============================================================

X = np.load(X_PATH)
y = np.load(Y_PATH)

split_df = pd.read_csv(SPLIT_PATH)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


# ============================================================
# GET INDICES FROM SIGNER SPLIT
# ============================================================

train_indices = split_df.loc[
    split_df["split"] == "train", "index"
].astype(int).to_numpy()

val_indices = split_df.loc[
    split_df["split"] == "val", "index"
].astype(int).to_numpy()

test_indices = split_df.loc[
    split_df["split"] == "test", "index"
].astype(int).to_numpy()


print(f"Train samples: {len(train_indices)}")
print(f"Validation samples: {len(val_indices)}")
print(f"Held-out test samples: {len(test_indices)}")


# ============================================================
# SAFETY CHECK — NO OVERLAP
# ============================================================

assert len(set(train_indices) & set(val_indices)) == 0
assert len(set(train_indices) & set(test_indices)) == 0
assert len(set(val_indices) & set(test_indices)) == 0

print("Split overlap check: PASSED")


# ============================================================
# DATASET
# ============================================================

class SignLanguageDataset(Dataset):

    def __init__(self, X, y, indices, augment=False):
        self.X = X
        self.y = y
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def spatial_augmentation(self, x):
        """
        Make landmark features more robust to different signers.

        Feature layout per frame:
            0:63     left hand
            63:126   right hand
            126:258  pose

        We apply small geometric perturbations rather than
        completely changing the sign.
        """

        x = x.copy()

        # ----------------------------------------------------
        # 1. Small coordinate scaling
        # ----------------------------------------------------
        if np.random.rand() < 0.8:
            scale = np.random.uniform(0.90, 1.10)

            # x/y coordinates are affected.
            # Keep z and visibility untouched.
            for start, end in [(0, 63), (63, 126), (126, 258)]:

                block = x[:, start:end]

                # Hand: 3 values per landmark
                if end <= 126:
                    block[:, 0::3] *= scale
                    block[:, 1::3] *= scale

                # Pose: 4 values per landmark
                else:
                    block[:, 0::4] *= scale
                    block[:, 1::4] *= scale

                x[:, start:end] = block

        # ----------------------------------------------------
        # 2. Small x/y translation
        # ----------------------------------------------------
        if np.random.rand() < 0.7:

            shift_x = np.random.normal(0.0, 0.025)
            shift_y = np.random.normal(0.0, 0.025)

            # Left hand
            x[:, 0:63:3] += shift_x
            x[:, 1:63:3] += shift_y

            # Right hand
            x[:, 63:126:3] += shift_x
            x[:, 64:126:3] += shift_y

            # Pose
            x[:, 126:258:4] += shift_x
            x[:, 127:258:4] += shift_y

        # ----------------------------------------------------
        # 3. Small independent hand scaling
        # ----------------------------------------------------
        if np.random.rand() < 0.5:

            left_scale = np.random.uniform(0.95, 1.05)
            right_scale = np.random.uniform(0.95, 1.05)

            x[:, 0:63:3] *= left_scale
            x[:, 1:63:3] *= left_scale

            x[:, 63:126:3] *= right_scale
            x[:, 64:126:3] *= right_scale

        # ----------------------------------------------------
        # 4. Small Gaussian landmark noise
        # ----------------------------------------------------
        if np.random.rand() < 0.7:

            noise = np.random.normal(
                0.0,
                0.008,
                size=x.shape
            ).astype(np.float32)

            x += noise

        return x

    def temporal_augmentation(self, x):

        # ----------------------------------------------------
        # Small temporal shift
        # ----------------------------------------------------
        if np.random.rand() < 0.5:

            shift = np.random.randint(-2, 3)

            if shift > 0:
                x = np.concatenate(
                    [
                        np.repeat(x[:1], shift, axis=0),
                        x[:-shift]
                    ],
                    axis=0
                )

            elif shift < 0:

                shift = abs(shift)

                x = np.concatenate(
                    [
                        x[shift:],
                        np.repeat(x[-1:], shift, axis=0)
                    ],
                    axis=0
                )

        # ----------------------------------------------------
        # Slight temporal speed variation
        # ----------------------------------------------------
        if np.random.rand() < 0.3:

            # Randomly duplicate/remove one nearby frame
            # while preserving sequence length.
            pos = np.random.randint(1, len(x) - 1)

            if np.random.rand() < 0.5:

                x = np.insert(
                    x,
                    pos,
                    x[pos],
                    axis=0
                )[:30]

            else:

                x = np.delete(
                    x,
                    pos,
                    axis=0
                )

                x = np.concatenate(
                    [x, x[-1:]],
                    axis=0
                )

        return x

    def __getitem__(self, idx):

        real_idx = self.indices[idx]

        sample = self.X[real_idx].astype(np.float32)
        label = int(self.y[real_idx])

        if self.augment:

            sample = self.spatial_augmentation(sample)
            sample = self.temporal_augmentation(sample)

        return (
            torch.tensor(sample, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long)
        )


# ============================================================
# DATASETS
# ============================================================

train_dataset = SignLanguageDataset(
    X,
    y,
    train_indices,
    augment=True
)

val_dataset = SignLanguageDataset(
    X,
    y,
    val_indices,
    augment=False
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

train_labels = y[train_indices]

class_counts = np.bincount(
    train_labels,
    minlength=20
)

class_weights = len(train_labels) / (
    20 * np.maximum(class_counts, 1)
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)


print("\nClass counts:")
for class_id in range(20):
    print(
        f"{ID_TO_LABEL[class_id]:12s}: "
        f"{class_counts[class_id]}"
    )


# ============================================================
# MODEL
# ============================================================

model = SignLanguageModel(
    input_size=258,
    num_classes=20
).to(DEVICE)


# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights,
    label_smoothing=0.05
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0005,
    weight_decay=0.01
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=5
)


# ============================================================
# TRAINING SETTINGS
# ============================================================

NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15

best_val_accuracy = 0.0
epochs_without_improvement = 0


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(1, NUM_EPOCHS + 1):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(batch_x)

        loss = criterion(
            logits,
            batch_y
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        train_loss += (
            loss.item() * batch_x.size(0)
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        train_correct += (
            predictions == batch_y
        ).sum().item()

        train_total += batch_y.size(0)

    train_loss /= train_total
    train_accuracy = train_correct / train_total


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    val_true = []
    val_pred = []

    with torch.no_grad():

        for batch_x, batch_y in val_loader:

            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            logits = model(batch_x)

            loss = criterion(
                logits,
                batch_y
            )

            val_loss += (
                loss.item() * batch_x.size(0)
            )

            predictions = torch.argmax(
                logits,
                dim=1
            )

            val_correct += (
                predictions == batch_y
            ).sum().item()

            val_total += batch_y.size(0)

            val_true.extend(
                batch_y.cpu().numpy()
            )

            val_pred.extend(
                predictions.cpu().numpy()
            )

    val_loss /= val_total
    val_accuracy = val_correct / val_total

    scheduler.step(val_accuracy)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss {train_loss:.4f} | "
        f"Train Acc {train_accuracy * 100:.2f}% | "
        f"Val Loss {val_loss:.4f} | "
        f"Val Acc {val_accuracy * 100:.2f}% | "
        f"LR {current_lr:.6f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL BY VALIDATION ACCURACY
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy
        epochs_without_improvement = 0

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "best_val_accuracy": best_val_accuracy,
                "epoch": epoch
            },
            MODEL_PATH
        )

        print(
            f"  ✓ New best validation accuracy: "
            f"{best_val_accuracy * 100:.2f}%"
        )

    else:

        epochs_without_improvement += 1


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:

        print(
            f"\nEarly stopping at epoch {epoch}"
        )

        break


# ============================================================
# LOAD BEST CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()


# ============================================================
# VALIDATION REPORT
# ============================================================

val_true = []
val_pred = []

with torch.no_grad():

    for batch_x, batch_y in val_loader:

        batch_x = batch_x.to(DEVICE)

        logits = model(batch_x)

        predictions = torch.argmax(
            logits,
            dim=1
        )

        val_true.extend(
            batch_y.numpy()
        )

        val_pred.extend(
            predictions.cpu().numpy()
        )


print("\n" + "=" * 70)
print("BEST VALIDATION RESULTS")
print("=" * 70)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print("\nClassification report:")

target_names = [
    ID_TO_LABEL[i]
    for i in range(20)
]

print(
    classification_report(
        val_true,
        val_pred,
        labels=list(range(20)),
        target_names=target_names,
        zero_division=0
    )
)


# ============================================================
# FINAL TEST REMAINS UNTOUCHED
# ============================================================

print("=" * 70)
print("FINAL TEST")
print("=" * 70)

print(
    f"Held-out test samples: {len(test_indices)}"
)

print(
    "The final unseen-signer test was NOT used "
    "during training or model selection."
)

print(
    "Run evaluate_signer_test.py separately "
    "after training."
)

print("\nModel saved to:")
print(MODEL_PATH)
