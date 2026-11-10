import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from model import SignLanguageModel
from labels import LABELS


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = "sign_language/data/processed"
MODEL_DIR = "sign_language/models"

BATCH_SIZE = 16
EPOCHS = 100

LEARNING_RATE = 0.0005
WEIGHT_DECAY = 0.01

PATIENCE = 15
GRADIENT_CLIP = 1.0

SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


print("=" * 70)
print("SIGN LANGUAGE TRAINING — ORIENTATION ROBUST VERSION")
print("=" * 70)

print(f"DEVICE: {DEVICE}")


# ============================================================
# LOAD DATA
# ============================================================

X = np.load(
    os.path.join(DATA_DIR, "X.npy")
)

y = np.load(
    os.path.join(DATA_DIR, "y.npy")
)

samples = pd.read_csv(
    os.path.join(DATA_DIR, "samples.csv")
)


print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


# ============================================================
# CREATE SIGNER-AWARE SPLIT
#
# ISL500:
#
# Train:
#   User001 - User010
#
# Validation:
#   User011 - User012
#
# Test:
#   User013 - User015
#
# The held-out test signers are NEVER used for training.
# ============================================================

train_indices = []
val_indices = []
test_indices = []


for idx, row in samples.iterrows():

    dataset = str(row["dataset"])
    signer = str(row["signer"])

    if dataset == "ISL500":

        if signer in [
            "User001",
            "User002",
            "User003",
            "User004",
            "User005",
            "User006",
            "User007",
            "User008",
            "User009",
            "User010"
        ]:
            train_indices.append(idx)

        elif signer in [
            "User011",
            "User012"
        ]:
            val_indices.append(idx)

        elif signer in [
            "User013",
            "User014",
            "User015"
        ]:
            test_indices.append(idx)

    else:

        # ----------------------------------------------------
        # Other datasets
        #
        # They are NOT used for the final unseen-signer test.
        # We split them only for additional training/validation.
        # ----------------------------------------------------

        label = row["label"]

        same_class = samples[
            (samples["dataset"] == dataset) &
            (samples["label"] == label)
        ].index.tolist()

        n = len(same_class)

        position = same_class.index(idx)

        if n == 1:

            train_indices.append(idx)

        elif n == 2:

            train_indices.append(idx)

        elif n == 3:

            if position < 2:
                train_indices.append(idx)
            else:
                val_indices.append(idx)

        elif n == 4:

            if position < 2:
                train_indices.append(idx)
            elif position == 2:
                val_indices.append(idx)
            else:
                test_indices.append(idx)

        else:

            rng = np.random.RandomState(
                SEED + int(idx)
            )

            r = rng.random()

            if r < 0.70:
                train_indices.append(idx)

            elif r < 0.85:
                val_indices.append(idx)

            else:
                test_indices.append(idx)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

train_indices = sorted(set(train_indices))
val_indices = sorted(set(val_indices))
test_indices = sorted(set(test_indices))


# ============================================================
# CHECK SPLIT OVERLAP
# ============================================================

assert set(train_indices).isdisjoint(val_indices)
assert set(train_indices).isdisjoint(test_indices)
assert set(val_indices).isdisjoint(test_indices)


print(f"Train samples: {len(train_indices)}")
print(f"Validation samples: {len(val_indices)}")
print(f"Held-out test samples: {len(test_indices)}")

print("Split overlap check: PASSED")


# ============================================================
# EXTRACT SPLITS
# ============================================================

X_train = X[train_indices]
y_train = y[train_indices]

X_val = X[val_indices]
y_val = y[val_indices]

X_test = X[test_indices]
y_test = y[test_indices]


# ============================================================
# CLASS COUNTS
# ============================================================

print("\nClass counts:")

for class_id, label in enumerate(LABELS):

    count = int(
        np.sum(y_train == class_id)
    )

    print(
        f"{label:<12}: {count}"
    )


# ============================================================
# HORIZONTAL FLIP
#
# Feature layout:
#
# 0:63      = left hand
# 63:126    = right hand
# 126:258   = pose
#
# Horizontal mirror:
#
# 1. Swap left/right hands
# 2. Negate X coordinates
# 3. Keep Y/Z/visibility unchanged
#
# IMPORTANT:
# This function is used ONLY for augmentation.
# The feature extractor itself is NOT changed.
# ============================================================

def horizontal_flip_sequence(sequence):

    sequence = sequence.copy()

    # --------------------------------------------------------
    # Separate feature blocks
    # --------------------------------------------------------

    left_hand = sequence[:, 0:63].copy()

    right_hand = sequence[:, 63:126].copy()

    pose = sequence[:, 126:258].copy()


    # --------------------------------------------------------
    # Swap hands
    # --------------------------------------------------------

    flipped_left_hand = right_hand.copy()

    flipped_right_hand = left_hand.copy()


    # --------------------------------------------------------
    # Flip X coordinate of hand landmarks
    #
    # Every landmark is:
    # X, Y, Z
    #
    # Therefore X is every third value.
    # --------------------------------------------------------

    flipped_left_hand[:, 0::3] *= -1

    flipped_right_hand[:, 0::3] *= -1


    # --------------------------------------------------------
    # Flip X coordinate of pose landmarks
    #
    # Every pose landmark is:
    # X, Y, Z, visibility
    #
    # Therefore X is every fourth value.
    # --------------------------------------------------------

    pose[:, 0::4] *= -1


    # --------------------------------------------------------
    # Recombine
    # --------------------------------------------------------

    flipped = np.concatenate(
        [
            flipped_left_hand,
            flipped_right_hand,
            pose
        ],
        axis=1
    )

    return flipped.astype(np.float32)


# ============================================================
# STANDARD TEMPORAL / NOISE AUGMENTATION
# ============================================================

def augment_sequence(sequence):

    sequence = sequence.copy()


    # --------------------------------------------------------
    # Small landmark noise
    # --------------------------------------------------------

    if np.random.random() < 0.5:

        noise = np.random.normal(
            0,
            0.015,
            sequence.shape
        ).astype(np.float32)

        sequence += noise


    # --------------------------------------------------------
    # Small temporal shift
    # --------------------------------------------------------

    if np.random.random() < 0.5:

        shift = np.random.randint(
            -2,
            3
        )

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


    # --------------------------------------------------------
    # Small global scale variation
    # --------------------------------------------------------

    if np.random.random() < 0.5:

        scale = np.random.uniform(
            0.95,
            1.05
        )

        sequence *= scale


    return sequence.astype(np.float32)


# ============================================================
# ORIENTATION-AWARE TRAINING AUGMENTATION
#
# IMPORTANT DIFFERENCE FROM PREVIOUS EXPERIMENT:
#
# We DO NOT double the dataset.
#
# Every training sample gets exactly ONE augmented version.
#
# 50%:
#     normal orientation
#
# 50%:
#     horizontal mirror
#
# This forces the model to learn both orientations without
# making the flipped dataset dominate training.
# ============================================================

X_train_augmented = []

flip_count = 0
normal_count = 0


for sequence in X_train:

    sequence = sequence.copy()


    # --------------------------------------------------------
    # 50% chance of horizontal flip
    # --------------------------------------------------------

    if np.random.random() < 0.50:

        sequence = horizontal_flip_sequence(
            sequence
        )

        flip_count += 1

    else:

        normal_count += 1


    # --------------------------------------------------------
    # Apply normal augmentation after orientation transform
    # --------------------------------------------------------

    sequence = augment_sequence(
        sequence
    )

    X_train_augmented.append(
        sequence
    )


X_train_aug = np.stack(
    X_train_augmented
).astype(np.float32)


print("\n" + "=" * 70)
print("ORIENTATION-AWARE AUGMENTATION")
print("=" * 70)

print(
    f"Original training samples : {len(X_train)}"
)

print(
    f"Normal orientation        : {normal_count}"
)

print(
    f"Flipped orientation       : {flip_count}"
)

print(
    f"Final training samples    : {len(X_train_aug)}"
)

print(
    "Validation data: UNCHANGED"
)

print(
    "Held-out test data: UNCHANGED"
)


# ============================================================
# CREATE DATASETS
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


# ============================================================
# DATALOADERS
# ============================================================

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
#
# Calculate using ORIGINAL training labels.
# ============================================================

classes = np.unique(
    y_train
)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)


class_weights = torch.ones(
    len(LABELS),
    dtype=torch.float32
)

for class_id, weight in zip(
    classes,
    weights
):

    class_weights[class_id] = float(
        weight
    )


class_weights = class_weights.to(
    DEVICE
)


# ============================================================
# MODEL
# ============================================================

model = SignLanguageModel(
    input_size=258,
    num_classes=len(LABELS)
)

model = model.to(
    DEVICE
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
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
# TRAINING TRACKING
# ============================================================

best_val_accuracy = 0.0
best_val_loss = float("inf")

epochs_without_improvement = 0


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "sign_language_model.pth"
)


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(
    1,
    EPOCHS + 1
):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    train_losses = []
    train_predictions = []
    train_targets = []


    for batch_x, batch_y in train_loader:

        batch_x = batch_x.to(
            DEVICE
        )

        batch_y = batch_y.to(
            DEVICE
        )


        optimizer.zero_grad()


        logits = model(
            batch_x
        )


        loss = criterion(
            logits,
            batch_y
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRADIENT_CLIP
        )


        optimizer.step()


        train_losses.append(
            loss.item()
        )


        predictions = torch.argmax(
            logits,
            dim=1
        )


        train_predictions.extend(
            predictions.detach()
            .cpu()
            .numpy()
        )

        train_targets.extend(
            batch_y.detach()
            .cpu()
            .numpy()
        )


    train_loss = np.mean(
        train_losses
    )

    train_accuracy = accuracy_score(
        train_targets,
        train_predictions
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_losses = []
    val_predictions = []
    val_targets = []


    with torch.no_grad():

        for batch_x, batch_y in val_loader:

            batch_x = batch_x.to(
                DEVICE
            )

            batch_y = batch_y.to(
                DEVICE
            )


            logits = model(
                batch_x
            )


            loss = criterion(
                logits,
                batch_y
            )


            val_losses.append(
                loss.item()
            )


            predictions = torch.argmax(
                logits,
                dim=1
            )


            val_predictions.extend(
                predictions.cpu().numpy()
            )

            val_targets.extend(
                batch_y.cpu().numpy()
            )


    val_loss = np.mean(
        val_losses
    )

    val_accuracy = accuracy_score(
        val_targets,
        val_predictions
    )


    # ========================================================
    # LR SCHEDULER
    # ========================================================

    scheduler.step(
        val_loss
    )


    current_lr = optimizer.param_groups[0]["lr"]


    print(
        f"Epoch {epoch:03d} | "
        f"Train Loss {train_loss:.4f} | "
        f"Train Acc {train_accuracy * 100:.2f}% | "
        f"Val Loss {val_loss:.4f} | "
        f"Val Acc {val_accuracy * 100:.2f}% | "
        f"LR {current_lr:.6f}"
    )


    # ========================================================
    # SAVE BEST MODEL BY VALIDATION ACCURACY
    #
    # This is intentionally accuracy-based.
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        best_val_loss = val_loss

        epochs_without_improvement = 0


        torch.save(
            model.state_dict(),
            MODEL_PATH
        )


        print(
            f"  ✓ New best validation accuracy: "
            f"{best_val_accuracy * 100:.2f}%"
        )

    else:

        epochs_without_improvement += 1


    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if epochs_without_improvement >= PATIENCE:

        print(
            f"\nEarly stopping at epoch {epoch}"
        )

        break


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()


# ============================================================
# FINAL VALIDATION REPORT
# ============================================================

val_predictions = []
val_targets = []


with torch.no_grad():

    for batch_x, batch_y in val_loader:

        batch_x = batch_x.to(
            DEVICE
        )

        logits = model(
            batch_x
        )


        predictions = torch.argmax(
            logits,
            dim=1
        )


        val_predictions.extend(
            predictions.cpu().numpy()
        )

        val_targets.extend(
            batch_y.numpy()
        )


print("\n" + "=" * 70)
print("BEST VALIDATION RESULTS")
print("=" * 70)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)


print("\nClassification report:")

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
# CONFUSION MATRIX
# ============================================================

print("\nConfusion matrix:")

cm = confusion_matrix(
    val_targets,
    val_predictions,
    labels=list(range(len(LABELS)))
)

print(cm)


# ============================================================
# HELD-OUT TEST INFORMATION
#
# IMPORTANT:
# This test is NOT evaluated here.
#
# evaluate_signer_test.py is the authoritative test because
# it evaluates User013, User014 and User015 separately.
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST")
print("=" * 70)

print(
    f"Held-out test samples: {len(test_indices)}"
)

print(
    "The final unseen-signer test was NOT used during "
    "training or model selection."
)

print(
    "Run evaluate_signer_test.py separately after training."
)


print("\nModel saved to:")

print(
    os.path.abspath(
        MODEL_PATH
    )
)