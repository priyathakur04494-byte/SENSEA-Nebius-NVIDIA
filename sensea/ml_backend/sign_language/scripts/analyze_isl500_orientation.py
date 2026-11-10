import cv2
import pandas as pd
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# PATHS
# ============================================================

SAMPLES_PATH = "sign_language/data/processed/samples.csv"

HAND_MODEL = "sign_language/models/mediapipe/hand_landmarker.task"


# ============================================================
# LOAD DATA
# ============================================================

samples = pd.read_csv(SAMPLES_PATH)

isl500 = samples[
    samples["dataset"] == "ISL500"
].copy()

print()
print("=" * 80)
print("ISL500 ORIENTATION / HANDEDNESS ANALYSIS")
print("=" * 80)

print()
print(f"ISL500 videos: {len(isl500)}")


# ============================================================
# HAND DETECTION FUNCTION
# ============================================================

def analyze_video(video_path, flip=False):

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=HAND_MODEL
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    total = 0
    any_hand = 0
    left = 0
    right = 0
    two_hands = 0

    with vision.HandLandmarker.create_from_options(
        options
    ) as detector:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            total += 1

            if flip:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = int(
                (total - 1) * 1000 / fps
            )

            result = detector.detect_for_video(
                image,
                timestamp_ms
            )

            hands = result.hand_landmarks

            if len(hands) > 0:
                any_hand += 1

            if len(hands) >= 2:
                two_hands += 1

            for i in range(len(hands)):

                if i >= len(result.handedness):
                    continue

                label = (
                    result.handedness[i][0]
                    .category_name
                    .lower()
                )

                if label == "left":
                    left += 1

                elif label == "right":
                    right += 1

    cap.release()

    return {
        "total": total,
        "any": any_hand,
        "left": left,
        "right": right,
        "two": two_hands,
    }


# ============================================================
# ANALYZE ALL ISL500 VIDEOS
# ============================================================

results = []

for i, row in enumerate(
    isl500.itertuples(index=False),
    start=1
):

    print(
        f"[{i}/{len(isl500)}] "
        f"{row.signer} | {row.label}"
    )

    original = analyze_video(
        row.video_path,
        flip=False
    )

    flipped = analyze_video(
        row.video_path,
        flip=True
    )

    if original is None or flipped is None:
        print("  ERROR")
        continue

    original_right_pct = (
        original["right"] /
        max(original["total"], 1)
    ) * 100

    flipped_right_pct = (
        flipped["right"] /
        max(flipped["total"], 1)
    ) * 100

    original_left_pct = (
        original["left"] /
        max(original["total"], 1)
    ) * 100

    flipped_left_pct = (
        flipped["left"] /
        max(flipped["total"], 1)
    ) * 100

    results.append({
        "index": row.index,
        "signer": row.signer,
        "label": row.label,

        "original_total": original["total"],
        "original_any": original["any"],
        "original_left": original["left"],
        "original_right": original["right"],

        "flipped_any": flipped["any"],
        "flipped_left": flipped["left"],
        "flipped_right": flipped["right"],

        "original_right_pct": original_right_pct,
        "flipped_right_pct": flipped_right_pct,

        "original_left_pct": original_left_pct,
        "flipped_left_pct": flipped_left_pct,
    })


# ============================================================
# SAVE
# ============================================================

df = pd.DataFrame(results)

output_path = (
    "sign_language/data/processed/"
    "isl500_orientation_analysis.csv"
)

df.to_csv(
    output_path,
    index=False
)


# ============================================================
# SIGNER SUMMARY
# ============================================================

print()
print()
print("=" * 80)
print("SIGNER SUMMARY")
print("=" * 80)

for signer in sorted(
    df["signer"].unique()
):

    signer_df = df[
        df["signer"] == signer
    ]

    print()
    print(
        f"{signer} "
        f"({len(signer_df)} videos)"
    )

    print(
        f"  Original right detections: "
        f"{signer_df['original_right_pct'].mean():.2f}%"
    )

    print(
        f"  Flipped right detections:  "
        f"{signer_df['flipped_right_pct'].mean():.2f}%"
    )

    print(
        f"  Original left detections:  "
        f"{signer_df['original_left_pct'].mean():.2f}%"
    )

    print(
        f"  Flipped left detections:   "
        f"{signer_df['flipped_left_pct'].mean():.2f}%"
    )


# ============================================================
# USER014 VS OTHER SIGNERS
# ============================================================

print()
print("=" * 80)
print("USER014 VS OTHER ISL500 SIGNERS")
print("=" * 80)

summary = (
    df.groupby("signer")
    .agg(
        videos=("index", "count"),
        original_right_pct=(
            "original_right_pct",
            "mean"
        ),
        flipped_right_pct=(
            "flipped_right_pct",
            "mean"
        ),
        original_left_pct=(
            "original_left_pct",
            "mean"
        ),
        flipped_left_pct=(
            "flipped_left_pct",
            "mean"
        ),
    )
    .sort_index()
)

print()

print(
    summary.to_string(
        float_format=lambda x: f"{x:.2f}"
    )
)


# ============================================================
# USER014 CLASS-BY-CLASS
# ============================================================

print()
print("=" * 80)
print("USER014 CLASS-BY-CLASS")
print("=" * 80)

user014 = df[
    df["signer"] == "User014"
]

for _, row in user014.iterrows():

    print(
        f"{row['label']:12s} | "
        f"Original R: "
        f"{row['original_right_pct']:6.1f}% | "
        f"Flipped R: "
        f"{row['flipped_right_pct']:6.1f}%"
    )


print()
print("=" * 80)
print(f"Saved: {output_path}")
print("=" * 80)
