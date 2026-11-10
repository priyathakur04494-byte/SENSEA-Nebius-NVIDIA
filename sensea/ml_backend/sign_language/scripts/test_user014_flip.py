import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

VIDEO = (
    "sign_language/data/raw/isl_isolated_40words/hello/"
    "hello__ISL500__00438__Hello__session166__clip005.mp4"
)

MODEL = "sign_language/models/mediapipe/hand_landmarker.task"


def test_video(flip=False):

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=MODEL
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    cap = cv2.VideoCapture(VIDEO)

    fps = cap.get(cv2.CAP_PROP_FPS)

    total = 0
    any_hand = 0
    left = 0
    right = 0
    two_hands = 0

    with vision.HandLandmarker.create_from_options(options) as detector:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            total += 1

            if flip:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = int((total - 1) * 1000 / fps)

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

                label = result.handedness[i][0].category_name

                if label.lower() == "left":
                    left += 1

                elif label.lower() == "right":
                    right += 1

    cap.release()

    return total, any_hand, left, right, two_hands


print("=" * 70)
print("USER014 HELLO — ORIGINAL VS FLIPPED")
print("=" * 70)

print("\nTesting ORIGINAL video...")

original = test_video(flip=False)

print("\nTesting FLIPPED video...")

flipped = test_video(flip=True)


print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print("\nORIGINAL:")
print(f"Total frames:      {original[0]}")
print(f"Any hand:          {original[1]}")
print(f"Left detections:   {original[2]}")
print(f"Right detections:  {original[3]}")
print(f"Two-hand frames:   {original[4]}")

print("\nFLIPPED:")
print(f"Total frames:      {flipped[0]}")
print(f"Any hand:          {flipped[1]}")
print(f"Left detections:   {flipped[2]}")
print(f"Right detections:  {flipped[3]}")
print(f"Two-hand frames:   {flipped[4]}")

print("\n" + "=" * 70)

if flipped[3] > original[3]:
    print(">>> FLIPPING IMPROVES RIGHT-HAND DETECTION")
elif original[3] > flipped[3]:
    print(">>> ORIGINAL HAS BETTER RIGHT-HAND DETECTION")
else:
    print(">>> FLIPPING DOES NOT IMPROVE RIGHT-HAND DETECTION")

print("=" * 70)
