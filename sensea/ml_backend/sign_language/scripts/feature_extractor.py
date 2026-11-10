import cv2
import numpy as np
import mediapipe as mp

from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MEDIAPIPE_DIR = BASE_DIR / "models" / "mediapipe"

HAND_MODEL = str(MEDIAPIPE_DIR / "hand_landmarker.task")
POSE_MODEL = str(MEDIAPIPE_DIR / "pose_landmarker.task")


# ============================================================
# CONSTANTS
# ============================================================

NUM_FRAMES = 30
FEATURE_SIZE = 258


# ============================================================
# CREATE HAND LANDMARKER
# ============================================================

hand_options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=HAND_MODEL
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

hand_landmarker = vision.HandLandmarker.create_from_options(
    hand_options
)


# ============================================================
# CREATE POSE LANDMARKER
# ============================================================

pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=POSE_MODEL
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

pose_landmarker = vision.PoseLandmarker.create_from_options(
    pose_options
)


# ============================================================
# NORMALIZE HAND
# ============================================================

def normalize_hand(landmarks):
    """
    Convert one hand into 63 normalized features.

    21 landmarks × 3 coordinates = 63 features.
    """

    points = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32
    )

    # Wrist = landmark 0
    wrist = points[0].copy()

    # Move wrist to origin
    points = points - wrist

    # Middle finger MCP = landmark 9
    reference = np.linalg.norm(points[9])

    # Avoid division by zero
    if reference > 1e-6:
        points = points / reference

    return points.flatten()


# ============================================================
# NORMALIZE POSE
# ============================================================

def normalize_pose(landmarks):
    """
    Convert pose into 132 normalized features.

    33 landmarks × 4 values = 132 features.

    Values:
        x, y, z, visibility
    """

    points = np.array(
        [
            [lm.x, lm.y, lm.z, lm.visibility]
            for lm in landmarks
        ],
        dtype=np.float32
    )

    # Left shoulder = 11
    # Right shoulder = 12
    shoulder_center = (
        points[11, :3] +
        points[12, :3]
    ) / 2.0

    # Normalize x/y/z around shoulder center
    points[:, :3] -= shoulder_center

    # Scale using shoulder distance
    shoulder_distance = np.linalg.norm(
        points[11, :3] -
        points[12, :3]
    )

    if shoulder_distance > 1e-6:
        points[:, :3] /= shoulder_distance

    return points.flatten()


# ============================================================
# EXTRACT 258 FEATURES FROM ONE FRAME
# ============================================================

def extract_features(hand_result, pose_result):

    # --------------------------------------------------------
    # HAND FEATURES
    # --------------------------------------------------------

    left_hand = np.zeros(63, dtype=np.float32)
    right_hand = np.zeros(63, dtype=np.float32)

    for i, hand_landmarks in enumerate(
        hand_result.hand_landmarks
    ):

        if i >= len(hand_result.handedness):
            continue

        handedness = (
            hand_result.handedness[i][0].category_name
        )

        features = normalize_hand(hand_landmarks)

        if handedness.lower() == "left":
            left_hand = features

        elif handedness.lower() == "right":
            right_hand = features

    # --------------------------------------------------------
    # POSE FEATURES
    # --------------------------------------------------------

    pose_features = np.zeros(132, dtype=np.float32)

    if len(pose_result.pose_landmarks) > 0:

        pose_features = normalize_pose(
            pose_result.pose_landmarks[0]
        )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    features = np.concatenate(
        [
            left_hand,
            right_hand,
            pose_features
        ]
    )

    return features


# ============================================================
# EXTRACT FEATURES FROM ONE FRAME
# ============================================================

def extract_frame_features(frame):
    """
    Extract 258 features from one camera/video frame.

    Uses the existing MediaPipe hand and pose landmarkers.
    """

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # MediaPipe VIDEO mode requires increasing timestamps.
    extract_frame_features.timestamp_ms += 33

    hand_result = hand_landmarker.detect_for_video(
        mp_image,
        extract_frame_features.timestamp_ms
    )

    pose_result = pose_landmarker.detect_for_video(
        mp_image,
        extract_frame_features.timestamp_ms
    )

    features = extract_features(
        hand_result,
        pose_result
    )

    return features


extract_frame_features.timestamp_ms = 0


# ============================================================
# CAMERA TEST
# ============================================================

if __name__ == "__main__":

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Camera could not be opened.")
        raise SystemExit

    print()
    print("=" * 60)
    print("SENSEA — 258 FEATURE EXTRACTOR TEST")
    print("=" * 60)
    print()
    print("Camera started.")
    print()
    print("Show your hands and upper body.")
    print("We will collect 30 frames.")
    print()
    print("Press Q to quit.")
    print()

    sequence = []

    while len(sequence) < NUM_FRAMES:

        ret, frame = cap.read()

        if not ret:
            print("ERROR: Could not read frame.")
            break

        features = extract_frame_features(frame)

        sequence.append(features)

        cv2.putText(
            frame,
            f"Frames: {len(sequence)}/{NUM_FRAMES}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "SENSEA Feature Extractor Test",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(sequence) == NUM_FRAMES:

        sequence = np.array(
            sequence,
            dtype=np.float32
        )

        print()
        print("=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print()
        print(f"Sequence shape: {sequence.shape}")
        print(
            f"Expected shape: "
            f"({NUM_FRAMES}, {FEATURE_SIZE})"
        )
        print(
            f"Feature count per frame: "
            f"{sequence.shape[1]}"
        )
        print()

        if sequence.shape == (
            NUM_FRAMES,
            FEATURE_SIZE
        ):
            print("✓ EXACT MODEL INPUT SHAPE!")
        else:
            print("✗ WRONG MODEL INPUT SHAPE!")

        print("=" * 60)

    else:

        print()
        print(
            f"Only collected {len(sequence)} frames."
        )