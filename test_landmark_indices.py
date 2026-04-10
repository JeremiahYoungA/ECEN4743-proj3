"""
Test to identify correct MediaPipe eye corner landmark indices
and visualize what landmarks 130, 243, 359, 133 actually represent
"""
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize MediaPipe
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.IMAGE)
face_landmarker = FaceLandmarker.create_from_options(options)

# Load first frame
cap = cv2.VideoCapture(input("Enter video path: "))
ret, frame = cap.read()
cap.release()

if not ret:
    print("Could not read frame")
    exit(1)

# Detect landmarks
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
results = face_landmarker.detect(mp_image)

if not results.face_landmarks:
    print("No face detected")
    exit(1)

landmarks = results.face_landmarks[0]

# MediaPipe Eye Landmark Reference:
# Eye indices vary by what part of the eye you want to measure
# Common eye-related indices (approximate ranges):
# - Eye contours: 33-133 (right), 263-362 (left)
# - Iris: 468-471 (left iris), 472-475 (right iris)

print("\n=== CURRENT INDICES BEING USED ===")
print(f"Left corner indices:  130 → 243")
print(f"Right corner indices: 359 → 133")
print()

print("=== LANDMARK VALUES AT CURRENT INDICES ===")
idx_to_test = [130, 243, 359, 133, 33, 133, 263, 362, 468, 469]

for idx in sorted(set(idx_to_test)):
    if idx < len(landmarks):
        lm = landmarks[idx]
        print(f"Index {idx:3d}: x={lm.x:.4f}, y={lm.y:.4f}, z={lm.z:+.4f}")

print("\n=== TESTING DIFFERENT EYE CORNER COMBINATIONS ===")

# Common eye corner patterns to test
test_patterns = [
    ("Current (130→243, 359→133)", (130, 243, 359, 133)),
    ("Try (33→133, 263→362)", (33, 133, 263, 362)),
    ("Try (46→53, 276→283)", (46, 53, 276, 283)),
    ("Left only (130→243)", (130, 243, None, None)),
]

for name, (l_in, l_out, r_in, r_out) in test_patterns:
    if l_in is not None and l_out is not None:
        left_w = abs(landmarks[l_out].x - landmarks[l_in].x)
        print(f"{name}")
        print(f"  Left:  {l_in} → {l_out} = {left_w:.4f}")
    if r_in is not None and r_out is not None:
        right_w = abs(landmarks[r_out].x - landmarks[r_in].x)
        print(f"  Right: {r_in} → {r_out} = {right_w:.4f}")
    print()

print("\n📌 IMPORTANT: Check MediaPipe Face Landmarks documentation:")
print("   https://mediapipe.dev/images/face_landmarks_2d.png")
print("   Look for which indices represent actual EYE CORNERS (not iris or other parts)")
