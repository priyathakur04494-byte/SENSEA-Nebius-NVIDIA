import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report

from sign_language.model import SignLanguageModel
from sign_language.labels import LABELS


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

X_PATH = BASE_DIR / "data" / "processed" / "X.npy"
Y_PATH = BASE_DIR / "data" / "processed" / "y.npy"
SPLIT_PATH = BASE_DIR / "data" / "processed" / "signer_split.csv"

MODEL_PATH = BASE_DIR / "models" / "sign_language_model.pth"

SEED = 42
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.01

PATIENCE = 15

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# AUGMENTATION
# ============================================================

def augment_sequence(x):
    """
    Apply light temporal/feature augmentation.

    x shape:
        (30, 258)
    """

    x = x.copy()

    # Gaussian noise
    if random.random() < 0.5:
        noise = np.random.normal(
            0,
            0.015,
            size=x.shape
        ).astype(np.float32)

        x += noise

    # Temporal shift
    if random.random() < 0.5:
        shift = random.randint(-2, 2)

        if shift > 0:
            x = np.concatenate(
                [x[shift:], np.repeat(x[-1:], shift, axis=0)],
                axis=0
            )

        elif shift < 0:
            shift = abs(shift)

            x = np.concatenate(
                [np.repeat(x[:1], shift, axis=0), x[:-shift]],
                axis=0
            )

    # Small feature scaling
    if random.random() < 0.5:
        scale = random.uniform(0.95, 1.05)
        x *= scale

    return x.astype(np.float32)


# ============================================================
# DATASET
# ============================================================

class SignLanguageDataset(Dataset):

    def __init__(self, X, y, augment=False):

        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.augment = augment

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        x = self.X[idx]

        if self.augment:
            x = augment_sequence(x)

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long)
        )


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("SIGNER-AWARE ISL TRAINING")
print("=" * 70)

print(f"\nDEVICE: {DEVICE}")

X = np.load(X_PATH)
y = np.load(Y_PATH)

split_df = pd.read_csv(SPLIT_PATH)

print("\nDataset:")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# ------------------------------------------------------------
# IMPORTANT:
# samples.csv and split.csv preserve the original index.
#
# Therefore we use the 'index' column to map the split
# directly to X.npy / y.npy.
# ------------------------------------------------------------

train_indices = split_df[
    split_df["split"] == "train"
]["index"].astype(int).to_numpy()

val_indices = split_df[
    split_df["split"] == "val"
]["index"].astype(int).to_numpy()

test_indices = split_df[
    split_df["split"] == "test"
]["index"].astype(int).to_numpy()


X_train = X[train_indices]
y_train = y[train_indices]

X_val = X[val_indices]
y_val = y[val_indices]

X_test = X[test_indices]
y_test = y[test_indices]


print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Held-out test samples:", len(X_test))


# ============================================================
# VERIFY NO LEAKAGE
# ============================================================

assert len(set(train_indices) & set(val_indices)) == 0
assert len(set(train_indices) & set(test_indices)) == 0
assert len(set(val_indices) & set(test_indices)) == 0

print("\n✓ No train/validation/test index overlap")


# ============================================================
# VERIFY ISL500 SIGNER SPLIT
# ============================================================

isl500_test = split_df[
    (split_df["dataset"] == "ISL500") &
    (split_df["split"] == "test")
]

print("\nFinal unseen-signer test:")
print(
    isl500_test.groupby("signer").size()
)

print("\nTest labels:")
print(
    isl500_test["label"].value_counts().sort_index()
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(
    y_train,
    minlength=len(LABELS)
)

total = len(y_train)

class_weights = np.zeros(len(LABELS), dtype=np.float32)

for i in range(len(LABELS)):

    if class_counts[i] > 0:
        class_weights[i] = total / (
            len(LABELS) * class_counts[i]
        )

    else:
        class_weights[i] = 0.0


print("\nClass weights:")

for i, label in enumerate(LABELS):
    print(
        f"{label:12s} "
        f"count={class_counts[i]:3d} "
        f"weight={class_weights[i]:.3f}"
    )


# ============================================================
# DATALOADERS
# ============================================================

train_dataset = SignLanguageDataset(
    X_train,
    y_train,
    augment=True
)

val_dataset = SignLanguageDataset(
    X_val,
    y_val,
    augment=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

model = SignLanguageModel(
    input_size=258,
    num_classes=len(LABELS)
).to(DEVICE)


# ============================================================
# LOSS
# ============================================================

weights_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE
)

criterion = nn.CrossEntropyLoss(
    weight=weights_tensor
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# LR SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5
)


# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = 0.0
best_model_state = None
epochs_without_improvement = 0


for epoch in range(1, EPOCHS + 1):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0
    train_predictions = []
    train_targets = []

    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(batch_x)

        loss = criterion(outputs, batch_y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        train_loss += loss.item() * len(batch_x)

        predictions = outputs.argmax(dim=1)

        train_predictions.extend(
            predictions.detach().cpu().numpy()
        )

        train_targets.extend(
            batch_y.detach().cpu().numpy()
        )

    train_loss /= len(train_dataset)

    train_accuracy = accuracy_score(
        train_targets,
        train_predictions
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0
    val_predictions = []
    val_targets = []

    with torch.no_grad():

        for batch_x, batch_y in val_loader:

            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            outputs = model(batch_x)

            loss = criterion(
                outputs,
                batch_y
            )

            val_loss += loss.item() * len(batch_x)

            predictions = outputs.argmax(dim=1)

            val_predictions.extend(
                predictions.cpu().numpy()
            )

            val_targets.extend(
                batch_y.cpu().numpy()
            )

    val_loss /= len(val_dataset)

    val_accuracy = accuracy_score(
        val_targets,
        val_predictions
    )


    # --------------------------------------------------------
    # LR SCHEDULER
    # --------------------------------------------------------

    scheduler.step(val_loss)

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
    # BEST MODEL
    #
    # IMPORTANT:
    # We select the checkpoint using validation accuracy,
    # not training accuracy and not test accuracy.
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_model_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

        epochs_without_improvement = 0

        print(
            f"  ✓ New best validation accuracy: "
            f"{best_val_accuracy * 100:.2f}%"
        )

    else:

        epochs_without_improvement += 1


    # --------------------------------------------------------
    # EARLY STOPPING
    # --------------------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")

        break


# ============================================================
# SAVE BEST MODEL
# ============================================================

if best_model_state is None:

    raise RuntimeError(
        "No best model was produced."
    )


model.load_state_dict(
    best_model_state
)

MODEL_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

torch.save(
    model.state_dict(),
    MODEL_PATH
)


print("\n" + "=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(f"Model saved to: {MODEL_PATH}")


# ============================================================
# VALIDATION REPORT
# ============================================================

model.eval()

val_predictions = []
val_targets = []

with torch.no_grad():

    for batch_x, batch_y in val_loader:

        batch_x = batch_x.to(DEVICE)

        outputs = model(batch_x)

        predictions = outputs.argmax(dim=1)

        val_predictions.extend(
            predictions.cpu().numpy()
        )

        val_targets.extend(
            batch_y.numpy()
        )


print("\nValidation classification report:")

print(
    classification_report(
        val_targets,
        val_predictions,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        zero_division=0
    )
)


# ============================================================
# IMPORTANT:
# DO NOT EVALUATE THE FINAL TEST SET HERE.
#
# The test set is reserved for the separate evaluation script.
# ============================================================

print("=" * 70)
print("FINAL TEST SET REMAINS UNTOUCHED")
print("=" * 70)

print(
    "Held-out ISL500 signers: "
    "User013, User014, User015"
)

print(
    "Run the dedicated signer-aware evaluation "
    "after training."
)
