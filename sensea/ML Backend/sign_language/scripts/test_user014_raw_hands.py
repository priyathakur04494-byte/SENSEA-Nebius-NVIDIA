import cv2
import mediapipe as mp

VIDEO = (
    "sign_language/data/raw/isl_isolated_40words/hello/"
    "hello__ISL500__00438__Hello__session166__clip005.mp4"
)

MODEL = "sign_language/models/mediapipe/hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)

cap = cv2.VideoCapture(VIDEO)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("=" * 70)
print("RAW MEDIAPIPE HAND TEST — USER014 HELLO")
print("=" * 70)
print(f"FPS: {fps}")
print(f"Frames: {frame_count}")
print()

left_frames = 0
right_frames = 0
any_hand_frames = 0
total_frames = 0

with HandLandmarker.create_from_options(options) as detector:

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        total_frames += 1

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = int((total_frames - 1) * 1000 / fps)

        result = detector.detect_for_video(
            mp_image,
            timestamp_ms
        )

        hands = result.hand_landmarks

        if len(hands) > 0:
            any_hand_frames += 1

        detected_labels = []

        for i, hand in enumerate(hands):

            # MediaPipe Tasks returns handedness separately
            if i < len(result.handedness):
                label = result.handedness[i][0].category_name
                score = result.handedness[i][0].score

                detected_labels.append(
                    f"{label}({score:.2f})"
                )

                if label.lower() == "left":
                    left_frames += 1

                elif label.lower() == "right":
                    right_frames += 1

        print(
            f"Frame {total_frames:02d}: "
            f"{', '.join(detected_labels) if detected_labels else 'NO HAND'}"
        )

cap.release()

print()
print("=" * 70)
print("RESULT")
print("=" * 70)
print(f"Total frames:       {total_frames}")
print(f"Any hand detected:  {any_hand_frames}/{total_frames}")
print(f"Left hand detected: {left_frames}/{total_frames}")
print(f"Right hand detected:{right_frames}/{total_frames}")
print("=" * 70)
