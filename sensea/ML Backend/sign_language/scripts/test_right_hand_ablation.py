import sys
import numpy as np
import torch
from pathlib import Path

SIGN_LANGUAGE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SIGN_LANGUAGE_DIR))

from model import SignLanguageModel
from scripts.feature_extractor import extract_frame_features

MODEL_PATH = SIGN_LANGUAGE_DIR / "models" / "sign_language_model.pth"
DATA_DIR = SIGN_LANGUAGE_DIR / "data" / "raw" / "isl_isolated_40words"

LABELS = [
    "hello", "goodbye", "thank_you", "sorry", "please",
    "yes", "no", "help", "stop", "okay",
    "me", "you", "mother", "father", "friend",
    "home", "hospital", "eat", "drink", "water"
]

ID_TO_LABEL = {i: label for i, label in enumerate(LABELS)}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

model = SignLanguageModel(
    input_size=258,
    num_classes=20
).to(device)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def prepare_video(video_path):
    import cv2

    cap = cv2.VideoCapture(str(video_path))

    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        return None

    # Exact same temporal preprocessing used during training
    if len(frames) >= 30:
        indices = np.linspace(0, len(frames) - 1, 30).astype(int)
        frames = [frames[i] for i in indices]
    else:
        last = frames[-1]
        frames.extend([last] * (30 - len(frames)))

    features = []

    for frame in frames:
        f = extract_frame_features(frame)

        if f is None:
            return None

        features.append(f)

    return np.array(features, dtype=np.float32)


def predict(x):
    tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        prediction = torch.argmax(output, dim=1).item()

    return ID_TO_LABEL[prediction]


def test_variant(name, x):
    predictions = []

    for video_x in x:
        predictions.append(predict(video_x))

    return predictions


# ---------------------------------------------------------
# Collect Father + Mother videos
# ---------------------------------------------------------

father_dir = DATA_DIR / "father"
mother_dir = DATA_DIR / "mother"

father_files = sorted(father_dir.glob("*.mp4"))
mother_files = sorted(mother_dir.glob("*.mp4"))

all_files = [
    (f, "father") for f in father_files
] + [
    (f, "mother") for f in mother_files
]

print()
print("=" * 75)
print("RIGHT-HAND ABLATION TEST")
print("=" * 75)
print(f"Father videos : {len(father_files)}")
print(f"Mother videos : {len(mother_files)}")
print(f"Total         : {len(all_files)}")


# ---------------------------------------------------------
# Extract all videos
# ---------------------------------------------------------

samples = []

for path, expected in all_files:

    x = prepare_video(path)

    if x is None:
        print(f"FAILED: {path.name}")
        continue

    samples.append({
        "path": path,
        "expected": expected,
        "x": x
    })


# ---------------------------------------------------------
# Test variants
# ---------------------------------------------------------

variants = {
    "full": lambda x: x.copy(),

    # Keep ONLY right hand, zero left hand + pose
    "right_hand_only": lambda x: np.concatenate(
        [
            np.zeros_like(x[:, :63]),
            x[:, 63:126],
            np.zeros_like(x[:, 126:])
        ],
        axis=1
    ),

    # Keep BOTH hands, remove pose
    "hands_only": lambda x: np.concatenate(
        [
            x[:, :126],
            np.zeros_like(x[:, 126:])
        ],
        axis=1
    ),

    # Keep right hand + pose, remove left hand
    "right_hand_pose": lambda x: np.concatenate(
        [
            np.zeros_like(x[:, :63]),
            x[:, 63:126],
            x[:, 126:]
        ],
        axis=1
    ),
}


results = {}

for variant_name, transform in variants.items():

    correct = 0
    total = 0

    print()
    print("-" * 75)
    print(variant_name.upper())
    print("-" * 75)

    for sample in samples:

        x = transform(sample["x"])
        prediction = predict(x)

        expected = sample["expected"]

        if prediction == expected:
            correct += 1

        total += 1

        print(
            f"{sample['path'].name[:45]:45s} "
            f"expected={expected:7s} "
            f"predicted={prediction:10s}"
        )

    accuracy = 100 * correct / total

    results[variant_name] = accuracy

    print(
        f"\n{variant_name:20s}: "
        f"{correct}/{total} "
        f"({accuracy:.2f}%)"
    )


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print()
print("=" * 75)
print("RIGHT-HAND ABLATION SUMMARY")
print("=" * 75)

for name, accuracy in results.items():
    print(f"{name:20s}: {accuracy:.2f}%")

print("=" * 75)
print("Diagnostic complete.")
print("=" * 75)
