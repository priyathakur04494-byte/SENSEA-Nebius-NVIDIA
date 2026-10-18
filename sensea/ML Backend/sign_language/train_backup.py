import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from model import SignLanguageModel
from labels import LABELS


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "sign_language/data/processed"
MODEL_PATH = "sign_language/models/sign_language_model.pth"

BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.0005

PATIENCE = 15
AUGMENT_PROBABILITY = 0.7

SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

print("=" * 60)
print("DEVICE:", DEVICE)
print("=" * 60)


# ============================================================
# LOAD DATA
# ============================================================

X = np.load(os.path.join(DATA_DIR, "X.npy"))
y = np.load(os.path.join(DATA_DIR, "y.npy"))

print("\nDataset:")
print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

indices = np.arange(len(X))

train_idx, val_idx = train_test_split(
    indices,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

X_train = X[train_idx]
y_train = y[train_idx]

X_val = X[val_idx]
y_val = y[val_idx]

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))


# ============================================================
# DATA AUGMENTATION
# ============================================================

def augment_sequence(sequence):
    """
    Apply small realistic variations to landmark sequences.
    """

    sequence = sequence.copy()

    # Small Gaussian noise
    if np.random.rand() < 0.5:
        noise = np.random.normal(
            0,
            0.015,
            sequence.shape
        ).astype(np.float32)

        sequence += noise

    # Small temporal shift
    if np.random.rand() < 0.3:
        shift = np.random.randint(-2, 3)

        if shift > 0:
            sequence = np.concatenate(
                [
                    sequence[shift:],
                    np.repeat(
                        sequence[-1:],
                        shift,
                        axis=0
                    )
                ],
                axis=0
            )

        elif shift < 0:
            shift = abs(shift)

            sequence = np.concatenate(
                [
                    np.repeat(
                        sequence[:1],
                        shift,
                        axis=0
                    ),
                    sequence[:-shift]
                ],
                axis=0
            )

    # Small global scaling
    if np.random.rand() < 0.3:
        scale = np.random.uniform(0.95, 1.05)
        sequence *= scale

    return sequence.astype(np.float32)


# Create augmented training data
augmented_X = []

for sequence in X_train:

    if np.random.rand() < AUGMENT_PROBABILITY:
        augmented_X.append(augment_sequence(sequence))
    else:
        augmented_X.append(sequence.astype(np.float32))


X_train_aug = np.array(augmented_X, dtype=np.float32)

print("\nAugmentation applied.")
print("Augmented training shape:", X_train_aug.shape)


# ============================================================
# CONVERT TO TENSORS
# ============================================================

X_train_tensor = torch.tensor(
    X_train_aug,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.long
)

X_val_tensor = torch.tensor(
    X_val,
    dtype=torch.float32
)

y_val_tensor = torch.tensor(
    y_val,
    dtype=torch.long
)


train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

val_dataset = TensorDataset(
    X_val_tensor,
    y_val_tensor
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

classes = np.unique(y_train)

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

weights = np.ones(len(LABELS), dtype=np.float32)

for cls, weight in zip(classes, class_weights):
    weights[cls] = weight

class_weights_tensor = torch.tensor(
    weights,
    dtype=torch.float32
).to(DEVICE)

print("\nClass weights:")
for i, label in enumerate(LABELS):
    print(f"{label:15s}: {weights[i]:.3f}")


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

criterion = nn.CrossEntropyLoss(
    weight=class_weights_tensor
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)


# ============================================================
# LEARNING RATE SCHEDULER
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

best_val_loss = float("inf")
best_val_accuracy = 0.0

best_model_state = None

epochs_without_improvement = 0


print("\n")
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    total_train_loss = 0
    correct_train = 0
    total_train = 0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(batch_X)

        loss = criterion(
            outputs,
            batch_y
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        total_train_loss += loss.item()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct_train += (
            predictions == batch_y
        ).sum().item()

        total_train += batch_y.size(0)

    train_loss = (
        total_train_loss /
        len(train_loader)
    )

    train_accuracy = (
        correct_train /
        total_train
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    total_val_loss = 0
    correct_val = 0
    total_val = 0

    with torch.no_grad():

        for batch_X, batch_y in val_loader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            outputs = model(batch_X)

            loss = criterion(
                outputs,
                batch_y
            )

            total_val_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct_val += (
                predictions == batch_y
            ).sum().item()

            total_val += batch_y.size(0)

    val_loss = (
        total_val_loss /
        len(val_loader)
    )

    val_accuracy = (
        correct_val /
        total_val
    )


    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print(
        f"Epoch {epoch + 1:03d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy * 100:.2f}% | "
        f"LR: {current_lr:.6f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        best_val_accuracy = val_accuracy

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        epochs_without_improvement = 0

        print(
            f"  ✓ Best model updated "
            f"(Val Acc: {val_accuracy * 100:.2f}%)"
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
# RESTORE BEST MODEL
# ============================================================

if best_model_state is not None:

    model.load_state_dict(
        best_model_state
    )


# ============================================================
# SAVE MODEL
# ============================================================

checkpoint = {
    "model_state_dict": model.state_dict(),
    "labels": LABELS,
    "input_size": 258,
    "sequence_length": 30,
    "num_classes": len(LABELS),
    "best_val_accuracy": best_val_accuracy
}

torch.save(
    checkpoint,
    MODEL_PATH
)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    "Model saved to:"
)

print(MODEL_PATH)

print("=" * 60)
