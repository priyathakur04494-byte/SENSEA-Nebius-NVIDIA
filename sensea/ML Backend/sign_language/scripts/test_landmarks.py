import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ============================================================
# MODEL PATHS
# ============================================================

HAND_MODEL = "sign_language/models/mediapipe/hand_landmarker.task"
POSE_MODEL = "sign_language/models/mediapipe/pose_landmarker.task"


# ============================================================
# CREATE HAND LANDMARKER
# ============================================================

hand_base_options = python.BaseOptions(
    model_asset_path=HAND_MODEL
)

hand_options = vision.HandLandmarkerOptions(
    base_options=hand_base_options,
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

pose_base_options = python.BaseOptions(
    model_asset_path=POSE_MODEL
)

pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base_options,
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
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print()
print("=" * 60)
print("MEDIA PIPE LANDMARK TEST")
print("=" * 60)
print()
print("Camera started.")
print("Show your hands and upper body to the camera.")
print("Press Q to quit.")
print()


frame_timestamp_ms = 0


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        print("ERROR: Could not read camera frame.")
        break

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Create MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Timestamp must increase for VIDEO mode
    frame_timestamp_ms += 33

    # --------------------------------------------------------
    # HAND DETECTION
    # --------------------------------------------------------

    hand_result = hand_landmarker.detect_for_video(
        mp_image,
        frame_timestamp_ms
    )

    # --------------------------------------------------------
    # POSE DETECTION
    # --------------------------------------------------------

    pose_result = pose_landmarker.detect_for_video(
        mp_image,
        frame_timestamp_ms
    )

    # --------------------------------------------------------
    # COUNT HANDS
    # --------------------------------------------------------

    number_of_hands = len(
        hand_result.hand_landmarks
    )

    # --------------------------------------------------------
    # COUNT POSE
    # --------------------------------------------------------

    number_of_poses = len(
        pose_result.pose_landmarks
    )

    # --------------------------------------------------------
    # PRINT DETECTION INFORMATION
    # --------------------------------------------------------

    print(
        f"\rHands: {number_of_hands} | "
        f"Pose detected: {number_of_poses > 0}",
        end=""
    )

    # --------------------------------------------------------
    # DRAW HAND LANDMARKS
    # --------------------------------------------------------

    for hand_landmarks in hand_result.hand_landmarks:

        h, w, _ = frame.shape

        for landmark in hand_landmarks:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(
                frame,
                (x, y),
                4,
                (0, 255, 0),
                -1
            )

    # --------------------------------------------------------
    # DRAW POSE LANDMARKS
    # --------------------------------------------------------

    if number_of_poses > 0:

        pose_landmarks = pose_result.pose_landmarks[0]

        h, w, _ = frame.shape

        for landmark in pose_landmarks:

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            cv2.circle(
                frame,
                (x, y),
                3,
                (255, 0, 0),
                -1
            )

    # --------------------------------------------------------
    # DISPLAY STATUS
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Hands: {number_of_hands}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Pose: {'YES' if number_of_poses > 0 else 'NO'}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    cv2.putText(
        frame,
        "Press Q to quit",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # SHOW CAMERA
    # --------------------------------------------------------

    cv2.imshow(
        "SENSEA - ISL Landmark Test",
        frame
    )

    # Quit with Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

hand_landmarker.close()
pose_landmarker.close()

print()
print()
print("Landmark test finished.")
