import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight

from model import SignLanguageModel
from labels import LABELS


# ============================================================
# SETTINGS
# ============================================================

DATA_DIR = "sign_language/data/processed"
MODEL_DIR = "sign_language/models"

BATCH_SIZE = 16
EPOCHS = 80
LEARNING_RATE = 0.001

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
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
print("Classes:", len(np.unique(y)))


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

train_indices = []
val_indices = []

rng = np.random.default_rng(42)

for class_id in np.unique(y):

    indices = np.where(y == class_id)[0]
    indices = rng.permutation(indices)

    # Very small classes:
    # keep at least one sample for validation
    if len(indices) <= 4:
        train_indices.extend(indices[:-1])
        val_indices.extend(indices[-1:])

    else:
        val_count = max(1, int(round(len(indices) * 0.20)))

        train_indices.extend(indices[:-val_count])
        val_indices.extend(indices[-val_count:])


train_indices = np.array(train_indices)
val_indices = np.array(val_indices)

rng.shuffle(train_indices)
rng.shuffle(val_indices)

X_train = X[train_indices]
y_train = y[train_indices]

X_val = X[val_indices]
y_val = y[val_indices]

print("\nSplit:")
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))


# ============================================================
# PYTORCH DATA
# ============================================================

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)

X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.long)

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

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

classes = np.unique(y_train.numpy())

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train.numpy()
)

class_weights = np.ones(
    len(LABELS),
    dtype=np.float32
)

for class_id, weight in zip(classes, weights):
    class_weights[class_id] = weight

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE
)

print("\nClass weights:")

for i, label in enumerate(LABELS):
    print(
        f"{label:15s} "
        f"{class_weights[i].item():.3f}"
    )


# ============================================================
# MODEL
# ============================================================

model = SignLanguageModel(
    input_size=258,
    num_classes=len(LABELS)
)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

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

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0
    train_correct = 0
    train_total = 0

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

        optimizer.step()

        train_loss += (
            loss.item() * batch_X.size(0)
        )

        predictions = outputs.argmax(dim=1)

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

    with torch.no_grad():

        for batch_X, batch_y in val_loader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            outputs = model(batch_X)

            loss = criterion(
                outputs,
                batch_y
            )

            val_loss += (
                loss.item() * batch_X.size(0)
            )

            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == batch_y
            ).sum().item()

            val_total += batch_y.size(0)

    val_loss /= val_total
    val_accuracy = val_correct / val_total

    scheduler.step(val_loss)


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "labels": LABELS,
                "input_size": 258,
                "sequence_length": 30,
                "num_classes": len(LABELS)
            },
            os.path.join(
                MODEL_DIR,
                "sign_language_model.pth"
            )
        )

        saved = "  ✓ SAVED"

    else:
        saved = ""


    print(
        f"Epoch {epoch + 1:03d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy * 100:.2f}%"
        f"{saved}"
    )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    "\nBest model saved to:"
)

print(
    "sign_language/models/"
    "sign_language_model.pth"
)