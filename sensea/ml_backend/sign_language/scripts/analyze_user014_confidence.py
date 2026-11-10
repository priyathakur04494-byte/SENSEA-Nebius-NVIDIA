import numpy as np
import torch
import pandas as pd

from sign_language.model import SignLanguageModel
from sign_language.labels import LABELS

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

X = np.load("sign_language/data/processed/X.npy")
y = np.load("sign_language/data/processed/y.npy")
split_df = pd.read_csv("sign_language/data/processed/signer_split.csv")

MODEL_PATH = "sign_language/models/sign_language_model.pth"

state = torch.load(MODEL_PATH, map_location=DEVICE)
if "model_state_dict" in state:
    state = state["model_state_dict"]

model = SignLanguageModel(input_size=258, num_classes=20)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()

# Only ISL500 User013/014/015
test_rows = split_df[
    split_df["signer"].isin(["User013", "User014", "User015"])
]

print("=" * 70)
print("CONFIDENCE ANALYSIS — UNSEEN SIGNERS")
print("=" * 70)

for signer in ["User013", "User014", "User015"]:

    print(f"\n{'-' * 70}")
    print(f"SIGNER: {signer}")
    print(f"{'-' * 70}")

    rows = test_rows[test_rows["signer"] == signer]

    correct_conf = []
    wrong_conf = []

    for _, row in rows.iterrows():
        idx = int(row["index"])

        x = torch.tensor(
            X[idx],
            dtype=torch.float32
        ).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]

        pred_id = int(torch.argmax(probs).item())
        pred_label = LABELS[pred_id]
        true_label = LABELS[int(y[idx])]
        confidence = float(probs[pred_id].item() * 100)

        top3_prob, top3_idx = torch.topk(probs, 3)

        top3 = [
            f"{LABELS[int(i)]}:{float(p)*100:.1f}%"
            for p, i in zip(top3_prob, top3_idx)
        ]

        status = "✓" if pred_label == true_label else "✗"

        print(
            f"{status} {true_label:12s} -> "
            f"{pred_label:12s} | "
            f"Top: {confidence:6.2f}% | "
            f"{', '.join(top3)}"
        )

        if pred_label == true_label:
            correct_conf.append(confidence)
        else:
            wrong_conf.append(confidence)

    print()
    print(f"Correct predictions: {len(correct_conf)}")
    if correct_conf:
        print(f"Average confidence when correct: {np.mean(correct_conf):.2f}%")

    print(f"Wrong predictions: {len(wrong_conf)}")
    if wrong_conf:
        print(f"Average confidence when wrong: {np.mean(wrong_conf):.2f}%")
