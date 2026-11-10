import sys
from pathlib import Path

import cv2
import numpy as np

SIGN_LANGUAGE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SIGN_LANGUAGE_DIR))

from scripts.feature_extractor import extract_frame_features

NUM_FRAMES = 30


def read_video(video_path):
    cap = cv2.VideoCapture(str(video_path))

    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()

    return frames


def sample_frames(frames):
    if len(frames) == 0:
        return []

    if len(frames) > NUM_FRAMES:
        indices = np.linspace(
            0,
            len(frames) - 1,
            NUM_FRAMES
        ).astype(int)

        return [frames[i] for i in indices]

    while len(frames) < NUM_FRAMES:
        frames.append(frames[-1])

    return frames


def extract_video(video_path):
    frames = read_video(video_path)
    frames = sample_frames(frames)

    if len(frames) != NUM_FRAMES:
        return None

    features = []

    for frame in frames:
        f = extract_frame_features(frame)

        if f is None:
            return None

        features.append(f)

    return np.asarray(features, dtype=np.float32)


def video_stats(video_path):
    X = extract_video(video_path)

    if X is None:
        return None

    # X shape = (30, 258)
    hands = X[:, :126]
    pose = X[:, 126:]

    # Overall variation inside the sequence
    hand_std = float(np.std(hands))
    pose_std = float(np.std(pose))

    # How much landmarks move from frame to frame
    hand_motion = float(np.mean(np.abs(np.diff(hands, axis=0))))
    pose_motion = float(np.mean(np.abs(np.diff(pose, axis=0))))

    # Mean absolute feature magnitude
    hand_energy = float(np.mean(np.abs(hands)))
    pose_energy = float(np.mean(np.abs(pose)))

    return {
        "hand_std": hand_std,
        "pose_std": pose_std,
        "hand_motion": hand_motion,
        "pose_motion": pose_motion,
        "hand_energy": hand_energy,
        "pose_energy": pose_energy,
    }


def analyze_class(class_name):
    folder = (
        SIGN_LANGUAGE_DIR
        / "data"
        / "raw"
        / "isl_isolated_40words"
        / class_name
    )

    results = []

    videos = sorted(folder.glob("*.mp4"))

    print()
    print("=" * 70)
    print(class_name.upper())
    print("=" * 70)

    for video in videos:

        stats = video_stats(video)

        if stats is None:
            print(f"FAILED: {video.name}")
            continue

        results.append(stats)

        print(
            f"{video.name[:42]:42s} "
            f"hand_motion={stats['hand_motion']:.4f} "
            f"pose_motion={stats['pose_motion']:.4f}"
        )

    if not results:
        return None

    keys = results[0].keys()

    averages = {
        key: float(np.mean([r[key] for r in results]))
        for key in keys
    }

    return averages


father = analyze_class("father")
mother = analyze_class("mother")


print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print()
print("FATHER")

for key, value in father.items():
    print(f"{key:15s}: {value:.6f}")

print()
print("MOTHER")

for key, value in mother.items():
    print(f"{key:15s}: {value:.6f}")

print()
print("=" * 70)
print("FATHER - MOTHER DIFFERENCE")
print("=" * 70)

for key in father.keys():
    difference = father[key] - mother[key]
    print(f"{key:15s}: {difference:+.6f}")

print()
print("Diagnostic complete.")
