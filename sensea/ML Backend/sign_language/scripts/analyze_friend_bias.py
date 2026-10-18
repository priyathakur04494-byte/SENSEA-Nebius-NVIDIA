from pathlib import Path
import numpy as np
import pandas as pd
import torch

from sign_language.model import SignLanguageModel
from sign_language.labels import ID_TO_LABEL

ROOT = Path(__file__).resolve().parents[2]

X = np.load(ROOT / "sign_language/data/processed/X.npy")
y = np.load(ROOT / "sign_language/data/processed/y.npy")

samples = pd.read_csv(
    ROOT / "sign_language/data/processed/samples.csv"
)

split = pd.read_csv(
    ROOT / "sign_language/data/processed/signer_split.csv"
)

df = samples.merge(
    split[["index", "split"]],
    on="index",
    how="left"
).sort_values("index")

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model = SignLanguageModel(
    input_size=258,
    num_classes=20
).to(device)

checkpoint = torch.load(
    ROOT / "sign_language/models/sign_language_model.pth",
    map_location=device
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

# ------------------------------------------------------------
# Evaluate every ISL500 signer
# ------------------------------------------------------------

print("=" * 70)
print("FRIEND BIAS / SIGNER ANALYSIS")
print("=" * 70)

results = []

with torch.no_grad():

    for signer in sorted(
        df[df["signer"].str.startswith("User")]["signer"].unique()
    ):

        signer_df = df[df["signer"] == signer]

        correct = 0
        friend_predictions = 0
        total = 0

        print("\n" + "-" * 70)
        print(f"SIGNER: {signer}")

        for _, row in signer_df.iterrows():

            idx = int(row["index"])
            true_label = row["label"]

            x = torch.tensor(
                X[idx],
                dtype=torch.float32
            ).unsqueeze(0).to(device)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)

            pred_id = int(torch.argmax(probs, dim=1).item())
            pred_label = ID_TO_LABEL[pred_id]

            confidence = float(probs[0, pred_id].item())

            total += 1

            if pred_label == true_label:
                correct += 1

            if pred_label == "friend":
                friend_predictions += 1

            print(
                f"{true_label:12s} -> "
                f"{pred_label:12s} "
                f"{confidence * 100:6.2f}%"
            )

        accuracy = correct / total * 100
        friend_rate = friend_predictions / total * 100

        print(
            f"\nAccuracy: {correct}/{total} "
            f"({accuracy:.2f}%)"
        )

        print(
            f"Predicted 'friend': "
            f"{friend_predictions}/{total} "
            f"({friend_rate:.2f}%)"
        )

        results.append({
            "signer": signer,
            "samples": total,
            "correct": correct,
            "accuracy": accuracy,
            "friend_predictions": friend_predictions,
            "friend_rate": friend_rate
        })

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

summary = pd.DataFrame(results)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(summary.to_string(index=False))

out = ROOT / "sign_language/data/processed/friend_bias_analysis.csv"
summary.to_csv(out, index=False)

print(f"\nSaved to: {out}")
