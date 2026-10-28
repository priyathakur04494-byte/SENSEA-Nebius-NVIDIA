# Pure feature functions copied from sign_language/scripts/feature_extractor.py.
# Importing that file creates two global MediaPipe landmarkers (and needs model
# files at a fixed path) as an import side effect, so the API uses this copy.
# KEEP IN SYNC with feature_extractor.py if the feature layout changes.
import numpy as np


def normalize_hand(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    pts = pts - pts[0].copy()
    ref = np.linalg.norm(pts[9])
    if ref > 1e-6:
        pts = pts / ref
    return pts.flatten()


def normalize_pose(landmarks):
    pts = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks], dtype=np.float32)
    center = (pts[11, :3] + pts[12, :3]) / 2.0
    pts[:, :3] -= center
    dist = np.linalg.norm(pts[11, :3] - pts[12, :3])
    if dist > 1e-6:
        pts[:, :3] /= dist
    return pts.flatten()


def extract_features(hand_result, pose_result):
    left = np.zeros(63, dtype=np.float32)
    right = np.zeros(63, dtype=np.float32)
    for i, hl in enumerate(hand_result.hand_landmarks):
        if i >= len(hand_result.handedness):
            continue
        side = hand_result.handedness[i][0].category_name.lower()
        feats = normalize_hand(hl)
        if side == "left":
            left = feats
        elif side == "right":
            right = feats
    pose = np.zeros(132, dtype=np.float32)
    if len(pose_result.pose_landmarks) > 0:
        pose = normalize_pose(pose_result.pose_landmarks[0])
    return np.concatenate([left, right, pose])
