import sys
from pathlib import Path

import cv2
import numpy as np

# ============================================================
# PATHS
# ============================================================

SIGN_LANGUAGE_DIR = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(SIGN_LANGUAGE_DIR))

from scripts.feature_extractor import extract_frame_features


NUM_FRAMES = 30


# ============================================================
# VIDEO READER
# ============================================================

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


# ============================================================
# SAME SAMPLING AS TRAINING
# ============================================================

def sample_frames(frames):

    if len(frames) > NUM_FRAMES:

        indices = np.linspace(
            0,
            len(frames) - 1,
            NUM_FRAMES
        ).astype(int)

        return [
            frames[i]
            for i in indices
        ]

    while len(frames) < NUM_FRAMES:
        frames.append(frames[-1])

    return frames


# ============================================================
# EXTRACT ONE VIDEO
# ============================================================

def extract_video(video_path):

    frames = read_video(video_path)

    frames = sample_frames(frames)

    features = []

    for frame in frames:

        f = extract_frame_features(frame)

        if f is None:
            return None

        features.append(f)

    return np.asarray(
        features,
        dtype=np.float32
    )


# ============================================================
# STATISTICS
# ============================================================

def video_stats(video_path):

    X = extract_video(video_path)

    if X is None:
        return None

    return {
        "shape": X.shape,
        "mean": float(np.mean(X)),
        "std": float(np.std(X)),
        "min": float(np.min(X)),
        "max": float(np.max(X)),
        "hand_pose_energy": float(
            np.mean(np.abs(X))
        )
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    father_dir = (
        SIGN_LANGUAGE_DIR
        / "data"
        / "raw"
        / "isl_isolated_40words"
        / "father"
    )

    mother_dir = (
        SIGN_LANGUAGE_DIR
        / "data"
        / "raw"
        / "isl_isolated_40words"
        / "mother"
    )

    print()
    print("=" * 60)
    print("FATHER vs MOTHER LANDMARK DIAGNOSTIC")
    print("=" * 60)

    # --------------------------------------------------------
    # Father
    # --------------------------------------------------------

    print()
    print("FATHER")
    print("-" * 60)

    father_stats = []

    for video in sorted(father_dir.glob("*.mp4")):

        stats = video_stats(video)

        if stats is None:
            print(
                f"FAILED: {video.name}"
            )
            continue

        father_stats.append(stats)

        print(
            f"{video.name[:45]:<45} "
            f"mean={stats['mean']:.4f} "
            f"std={stats['std']:.4f}"
        )

    # --------------------------------------------------------
    # Mother
    # --------------------------------------------------------

    print()
    print("MOTHER")
    print("-" * 60)

    mother_stats = []

    for video in sorted(mother_dir.glob("*.mp4")):

        stats = video_stats(video)

        if stats is None:
            print(
                f"FAILED: {video.name}"
            )
            continue

        mother_stats.append(stats)

        print(
            f"{video.name[:45]:<45} "
            f"mean={stats['mean']:.4f} "
            f"std={stats['std']:.4f}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if father_stats:

        father_mean = np.mean(
            [x["mean"] for x in father_stats]
        )

        father_std = np.mean(
            [x["std"] for x in father_stats]
        )

        print(
            f"Father average feature mean: {father_mean:.4f}"
        )

        print(
            f"Father average feature std : {father_std:.4f}"
        )

    if mother_stats:

        mother_mean = np.mean(
            [x["mean"] for x in mother_stats]
        )

        mother_std = np.mean(
            [x["std"] for x in mother_stats]
        )

        print(
            f"Mother average feature mean: {mother_mean:.4f}"
        )

        print(
            f"Mother average feature std : {mother_std:.4f}"
        )

    print()
    print("Diagnostic complete.")
