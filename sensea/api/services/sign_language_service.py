# ============================================================
# SENSEA API — Sign Language Service
# ============================================================
#
# Wraps:
#   - sign_language/model.py (Conv1D + BiGRU + Attention)
#   - sign_language/scripts/feature_extractor.py (MediaPipe)
#   - sign_language/labels.py
#
# The PyTorch model is loaded once at startup.
# The API accepts:
#   - a video file upload → extract 30 frames → predict
#   - a sequence of pre-extracted features as JSON
#
# MediaPipe task files must be present at MEDIAPIPE_DIR.
# ============================================================

from __future__ import annotations

import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Lazy singletons ─────────────────────────────────────────
_sl_model = None
_sl_device = None
_sl_labels: Optional[list[str]] = None
_sl_ready = False

SEQUENCE_LENGTH = 30
CONFIDENCE_THRESHOLD = 0.80
ORIENTATION_THRESHOLD = 1.0   # % right-hand detections → flip if below


def _init_sign_language() -> None:
    """Load the SignLanguageModel once."""
    global _sl_model, _sl_device, _sl_labels, _sl_ready
    if _sl_ready:
        return

    try:
        import torch  # type: ignore
        from config import SIGN_LANGUAGE_MODEL_PATH  # type: ignore
        from model import SignLanguageModel  # type: ignore  (sign_language/ on path)
        from labels import LABELS  # type: ignore

        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        model = SignLanguageModel(input_size=258, num_classes=len(LABELS)).to(device)

        ckpt_path = Path(SIGN_LANGUAGE_MODEL_PATH)
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Sign language model not found at {ckpt_path}. "
                "Set SIGN_LANGUAGE_MODEL_PATH in .env."
            )

        checkpoint = torch.load(str(ckpt_path), map_location=device)
        state = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state)
        model.eval()

        _sl_model = model
        _sl_device = device
        _sl_labels = LABELS
        _sl_ready = True
        logger.info("Sign language model loaded on %s.", device)

    except Exception as exc:
        logger.warning("Sign language model unavailable: %s", exc)
        raise RuntimeError(
            f"Sign language model could not be loaded: {exc}"
        ) from exc


def _extract_features_from_video(video_path: str) -> tuple[np.ndarray, str]:
    """
    Extract 30-frame feature sequences from a video file using MediaPipe.

    Returns (sequence_array, orientation) where orientation is
    'original' or 'flipped'.
    """
    import cv2  # type: ignore

    from config import MEDIAPIPE_DIR  # type: ignore

    # Override MediaPipe model paths via environment
    mp_dir = Path(MEDIAPIPE_DIR)
    hand_model = str(mp_dir / "hand_landmarker.task")
    pose_model = str(mp_dir / "pose_landmarker.task")

    if not Path(hand_model).exists() or not Path(pose_model).exists():
        raise FileNotFoundError(
            f"MediaPipe task files not found in {mp_dir}. "
            "Set MEDIAPIPE_DIR in .env."
        )

    import mediapipe as mp  # type: ignore
    from mediapipe.tasks import python as mp_python  # type: ignore
    from mediapipe.tasks.python import vision as mp_vision  # type: ignore

    # Create fresh landmarkers (VIDEO mode requires increasing timestamps)
    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=hand_model),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=pose_model),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_opts)
    pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_opts)

    from services.sign_features import (  # type: ignore
        extract_features,
        normalize_hand,
        normalize_pose,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    timestamp_ms = 0
    orientation_frames = 0
    right_hand_frames = 0
    sequence: list[np.ndarray] = []

    try:
        while len(sequence) < SEQUENCE_LENGTH:
            ret, raw_frame = cap.read()
            if not ret:
                break

            timestamp_ms += 33
            rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            hand_result = hand_landmarker.detect_for_video(mp_image, timestamp_ms)
            pose_result = pose_landmarker.detect_for_video(mp_image, timestamp_ms)

            # Orientation detection (count right-hand frames)
            orientation_frames += 1
            for handedness in hand_result.handedness:
                if handedness and handedness[0].category_name.lower() == "right":
                    right_hand_frames += 1
                    break

            features = extract_features(hand_result, pose_result)
            sequence.append(features)
    finally:
        cap.release()
        hand_landmarker.close()
        pose_landmarker.close()

    if not sequence:
        raise ValueError("No frames could be extracted from the video.")

    # Pad with zeros if fewer than 30 frames
    while len(sequence) < SEQUENCE_LENGTH:
        sequence.append(np.zeros(258, dtype=np.float32))

    arr = np.array(sequence[:SEQUENCE_LENGTH], dtype=np.float32)

    right_pct = (right_hand_frames / orientation_frames * 100.0) if orientation_frames else 0
    orientation = "flipped" if right_pct < ORIENTATION_THRESHOLD else "original"

    if orientation == "flipped":
        # Swap left/right hand features (first 63 = left, next 63 = right)
        arr_flipped = arr.copy()
        arr_flipped[:, :63] = arr[:, 63:126]
        arr_flipped[:, 63:126] = arr[:, :63]
        arr = arr_flipped

    return arr, orientation


def predict_from_video(video_bytes: bytes) -> dict:
    """
    Predict the sign language gesture from a video upload.

    Parameters
    ----------
    video_bytes : bytes
        Raw video file (MP4, MOV, WebM, etc.).

    Returns
    -------
    dict with keys: prediction (label, confidence, all_scores),
                    frames_used, orientation
    """
    _init_sign_language()

    import torch  # type: ignore

    # Write to temp file (cv2.VideoCapture needs a path)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        tmp.write(video_bytes)
        tmp.close()
        sequence, orientation = _extract_features_from_video(tmp.name)
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    return _run_model_inference(sequence, orientation)


def predict_from_frames(
    frames: list[list[float]],
    orientation: str = "original",
) -> dict:
    """
    Predict sign language gesture from pre-extracted feature frames.

    Parameters
    ----------
    frames : list[list[float]]
        List of up to 30 feature vectors, each of length 258.
    orientation : str
        'original' or 'flipped' (for documentation; not reapplied here).

    Returns
    -------
    dict with keys: prediction, frames_used, orientation
    """
    _init_sign_language()

    arr = np.array(frames, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 258:
        raise ValueError(
            f"Expected frames of shape (N, 258), got {arr.shape}."
        )

    # Pad / truncate to SEQUENCE_LENGTH
    if len(arr) < SEQUENCE_LENGTH:
        pad = np.zeros((SEQUENCE_LENGTH - len(arr), 258), dtype=np.float32)
        arr = np.vstack([arr, pad])
    else:
        arr = arr[:SEQUENCE_LENGTH]

    return _run_model_inference(arr, orientation)


def _run_model_inference(sequence: np.ndarray, orientation: str) -> dict:
    """Run the PyTorch model on a (30, 258) feature array."""
    import torch  # type: ignore
    from labels import LABELS  # type: ignore  (already on path)

    input_tensor = (
        torch.tensor(sequence, dtype=torch.float32)
        .unsqueeze(0)
        .to(_sl_device)
    )

    with torch.no_grad():
        outputs = _sl_model(input_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)

    confidence = float(conf.item())
    label = LABELS[int(pred_idx.item())] if confidence >= CONFIDENCE_THRESHOLD else "uncertain"

    all_scores = {
        lbl: round(float(probs[i].item()), 4)
        for i, lbl in enumerate(LABELS)
    }

    return {
        "prediction": {
            "label": label,
            "confidence": round(confidence, 4),
            "all_scores": all_scores,
        },
        "frames_used": SEQUENCE_LENGTH,
        "orientation": orientation,
    }
