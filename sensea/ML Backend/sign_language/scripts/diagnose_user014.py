from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sign_language.scripts.feature_extractor import extract_frame_features

SAMPLES = ROOT / "sign_language" / "data" / "processed" / "samples.csv"

df = pd.read_csv(SAMPLES)

user_df = df[
    (df["dataset"] == "ISL500") &
    (df["signer"] == "User014")
].copy()

print("=" * 70)
print("USER014 FEATURE EXTRACTION DIAGNOSTIC")
print("=" * 70)

print(f"\nVideos found: {len(user_df)}")
print("\nSigns:")
print(", ".join(user_df["label"].tolist()))

results = []

for _, row in user_df.iterrows():

    video_path = ROOT / row["video_path"]

    print("\n" + "-" * 70)
    print(f"Sign: {row['label']}")
    print(f"Video: {video_path.name}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print("ERROR: Could not open video")

        results.append({
            "label": row["label"],
            "video": str(video_path),
            "frames": 0,
            "successful_frames": 0,
            "failed_frames": 0,
            "zero_frames": 0,
            "status": "VIDEO_OPEN_ERROR"
        })

        continue

    total_frames = 0
    successful_frames = 0
    failed_frames = 0
    zero_frames = 0
    feature_norms = []

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        total_frames += 1

        try:
            features = extract_frame_features(frame)

            features = np.asarray(features, dtype=np.float32)

            if features.shape != (258,):
                print(
                    f"Unexpected feature shape at frame "
                    f"{total_frames}: {features.shape}"
                )
                failed_frames += 1
                continue

            if np.isnan(features).any() or np.isinf(features).any():
                failed_frames += 1
                continue

            successful_frames += 1

            norm = float(np.linalg.norm(features))
            feature_norms.append(norm)

            if norm < 1e-6:
                zero_frames += 1

        except Exception as e:
            failed_frames += 1

    cap.release()

    if feature_norms:
        min_norm = min(feature_norms)
        max_norm = max(feature_norms)
        mean_norm = float(np.mean(feature_norms))
    else:
        min_norm = max_norm = mean_norm = 0.0

    success_rate = (
        successful_frames / total_frames * 100
        if total_frames else 0
    )

    print(f"Total frames:       {total_frames}")
    print(f"Successful frames:  {successful_frames}")
    print(f"Failed frames:      {failed_frames}")
    print(f"Success rate:       {success_rate:.2f}%")
    print(f"Near-zero frames:   {zero_frames}")
    print(
        f"Feature norm:       "
        f"min={min_norm:.4f}, "
        f"max={max_norm:.4f}, "
        f"mean={mean_norm:.4f}"
    )

    results.append({
        "label": row["label"],
        "video": str(video_path),
        "frames": total_frames,
        "successful_frames": successful_frames,
        "failed_frames": failed_frames,
        "zero_frames": zero_frames,
        "success_rate": success_rate,
        "min_norm": min_norm,
        "max_norm": max_norm,
        "mean_norm": mean_norm,
        "status": "OK"
    })


out = ROOT / "sign_language" / "data" / "processed" / "user014_diagnostic.csv"

pd.DataFrame(results).to_csv(out, index=False)

print("\n" + "=" * 70)
print("DIAGNOSTIC COMPLETE")
print("=" * 70)

print("\nSummary:")
summary = pd.DataFrame(results)

print(
    summary[
        [
            "label",
            "frames",
            "successful_frames",
            "failed_frames",
            "zero_frames",
            "success_rate"
        ]
    ].to_string(index=False)
)

print(f"\nSaved to:")
print(out)
