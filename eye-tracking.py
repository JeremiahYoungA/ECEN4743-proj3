import cv2
import numpy as np
import math
import csv
import json

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

# Download face landmarker model if not present
MODEL_PATH = 'face_landmarker.task'
if not os.path.exists(MODEL_PATH):
    print("Downloading face landmarker model...")
    url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Model downloaded successfully!")

# Initialize Face Landmarker (provides detailed facial landmarks like FaceMesh)
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create Face Landmarker with model
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO)

face_landmarker = FaceLandmarker.create_from_options(options)

# Eye landmark indices for MediaPipe Face Landmarker
LEFT_IRIS = 468  # Left iris center
RIGHT_IRIS = 469  # Right iris center

def calculate_eye_angle(landmarks, eye_center_idx, iris_idx):
    """Calculate angular position of eye (pitch/yaw)"""
    eye_center = landmarks[eye_center_idx]
    iris = landmarks[iris_idx]
    
    # Vector from eye center to iris
    dx = iris.x - eye_center.x
    dy = iris.y - eye_center.y
    
    # Calculate angles (in degrees)
    horizontal_angle = math.degrees(math.atan2(dx, 1))  # Yaw
    vertical_angle = math.degrees(math.atan2(dy, 1))    # Pitch
    
    return horizontal_angle, vertical_angle

def process_video(video_path, output_csv=None, output_json=None):
    """Process video file and extract eye angles, save to files"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval_ms = 1000 / fps  # Convert FPS to milliseconds per frame
    
    print(f"Video FPS: {fps}")
    print(f"Frame interval: {frame_interval_ms:.2f}ms\n")
    
    frame_count = 0
    timestamp_ms = 0
    eye_data = []  # Store all eye tracking data
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Convert frame to MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect faces and landmarks
        results = face_landmarker.detect_for_video(mp_image, int(timestamp_ms))
        
        if results.face_landmarks:
            landmarks = results.face_landmarks[0]
            
            # Get angles for both eyes (landmark indices 33 and 362 are eye centers)
            left_h, left_v = calculate_eye_angle(landmarks, 33, LEFT_IRIS)
            right_h, right_v = calculate_eye_angle(landmarks, 362, RIGHT_IRIS)
            
            print(f"Frame {frame_count}:")
            print(f"  Left Eye  - Horizontal: {left_h:7.2f}°, Vertical: {left_v:7.2f}°")
            print(f"  Right Eye - Horizontal: {right_h:7.2f}°, Vertical: {right_v:7.2f}°")
            print()
            
            # Store data
            eye_data.append({
                'frame': frame_count,
                'timestamp_ms': timestamp_ms,
                'left_horizontal': left_h,
                'left_vertical': left_v,
                'right_horizontal': right_h,
                'right_vertical': right_v
            })
        
        timestamp_ms += frame_interval_ms
    
    cap.release()
    print(f"Processing complete. Processed {frame_count} frames.\n")
    
    # Save to CSV
    if output_csv:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['frame', 'timestamp_ms', 'left_horizontal', 'left_vertical', 'right_horizontal', 'right_vertical'])
            writer.writeheader()
            writer.writerows(eye_data)
        print(f"✓ CSV saved to: {output_csv}")
    
    # Save to JSON
    if output_json:
        with open(output_json, 'w') as f:
            json.dump(eye_data, f, indent=2)
        print(f"✓ JSON saved to: {output_json}")

if __name__ == "__main__":
    # Replace with your video file path
    video_file = input("Enter the path to the video file: ")
    process_video(
        video_file,
        output_csv="eye_tracking_data.csv",
        output_json=None
    )