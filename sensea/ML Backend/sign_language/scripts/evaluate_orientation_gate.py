import numpy as np
import pandas as pd

CSV_PATH = "sign_language/data/processed/isl500_orientation_analysis.csv"

df = pd.read_csv(CSV_PATH)

print("=" * 70)
print("AUTOMATIC ORIENTATION GATE DIAGNOSTIC")
print("=" * 70)

print("\nWe will test whether MediaPipe handedness statistics")
print("can automatically detect User014-like orientation.\n")

# ---------------------------------------------------------
# Simple rule:
#
# User014 has:
#   very low original_right_pct
#   very high original_left_pct
#
# Everyone else has:
#   substantially higher original_right_pct
#   substantially lower original_left_pct
#
# We test several thresholds.
# ---------------------------------------------------------

thresholds = [5, 7.5, 10, 12.5, 15, 20]

print("Signer-level statistics:")
print()

for signer, g in df.groupby("signer"):
    right = g["original_right_pct"].mean()
    left = g["original_left_pct"].mean()

    print(
        f"{signer:8s} | "
        f"original_right={right:6.2f}% | "
        f"original_left={left:6.2f}%"
    )

print("\n" + "=" * 70)
print("THRESHOLD TEST")
print("=" * 70)

for t in thresholds:

    # Gate says "flip" when original right-hand detection
    # is below threshold.
    df["pred_flip"] = df["original_right_pct"] < t

    # User014 is the only known orientation outlier
    # in this diagnostic population.
    df["true_outlier"] = df["signer"] == "User014"

    tp = ((df["pred_flip"]) & (df["true_outlier"])).sum()
    tn = ((~df["pred_flip"]) & (~df["true_outlier"])).sum()
    fp = ((df["pred_flip"]) & (~df["true_outlier"])).sum()
    fn = ((~df["pred_flip"]) & (df["true_outlier"])).sum()

    accuracy = (tp + tn) / len(df)

    print(
        f"\nThreshold < {t}%"
    )
    print(
        f"  TP={tp:3d}  TN={tn:3d}  "
        f"FP={fp:3d}  FN={fn:3d}  "
        f"Gate accuracy={accuracy*100:.2f}%"
    )

    flipped_signers = (
        df.loc[df["pred_flip"], "signer"]
        .value_counts()
        .to_dict()
    )

    print(f"  Videos gated to flip: {df['pred_flip'].sum()}")
    print(f"  By signer: {flipped_signers}")

print("\n" + "=" * 70)
print("IMPORTANT")
print("=" * 70)

print(
    "\nThis is only an ORIENTATION diagnostic."
    "\nIt does NOT change the model."
    "\nIt does NOT retrain anything."
    "\nIt does NOT yet mean we should put this rule into realtime."
)

print("\nDone.")
