#!/usr/bin/env python3
"""
Eye Tracking v2: Pure OpenCV Image Processing Approach
Uses Haar Cascade for eye detection + Hough Circle detection for pupil
This approach avoids MediaPipe's coordinate normalization issues by working directly in pixel space.
"""
import cv2
import numpy as np
import math
import csv
import json
from threading import Thread
from queue import Queue
import time
import subprocess
import os

# Check for GPU availability
def detect_nvidia_gpu():
    """Detect NVIDIA GPU using nvidia-smi"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout:
            gpu_info = result.stdout.strip().split(',')
            gpu_name = gpu_info[0].strip()
            driver_version = gpu_info[1].strip() if len(gpu_info) > 1 else "unknown"
            gpu_memory = gpu_info[2].strip() if len(gpu_info) > 2 else "unknown"
            print(f"🔍 GPU Detection: NVIDIA {gpu_name}")
            print(f"   Driver version: {driver_version}")
            print(f"   Memory: {gpu_memory}\n")
            return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return False

gpu_available = detect_nvidia_gpu()

# Load OpenCV's pre-trained Haar Cascade classifiers
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

if face_cascade.empty() or eye_cascade.empty():
    print("Warning: Could not load cascade classifiers. Eye detection may fail.")

def detect_pupil_hough(eye_roi, min_radius=5, max_radius=30):
    """Detect pupil using Hough Circle Detection
    
    Args:
        eye_roi: ROI containing the eye
        min_radius: Minimum pupil radius in pixels
        max_radius: Maximum pupil radius in pixels
    
    Returns:
        tuple: (center_x, center_y, radius) or None if not found
    """
    # Convert to grayscale
    gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 1.5)
    
    # Apply histogram equalization for better contrast
    equalized = cv2.equalizeHist(blurred)
    
    # Detect circles using Hough
    circles = cv2.HoughCircles(
        equalized,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,
        param1=100,        # Upper threshold for Canny edge detection
        param2=20,         # Accumulator threshold
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        if len(circles[0]) > 0:
            # Return the most prominent circle
            x, y, r = circles[0][0]
            return int(x), int(y), int(r)
    
    return None

def detect_pupil_threshold(eye_roi, min_radius=5, max_radius=30):
    """Fallback: Detect pupil using morphological operations and contours
    
    Args:
        eye_roi: ROI containing the eye
        min_radius: Minimum pupil radius in pixels
        max_radius: Maximum pupil radius in pixels
    
    Returns:
        tuple: (center_x, center_y, radius) or None if not found
    """
    gray = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
    
    # Inverse binary threshold (pupil is dark)
    _, binary = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the largest contour (likely the pupil)
        largest = max(contours, key=cv2.contourArea)
        
        # Fit a circle to the contour
        (center_x, center_y), radius = cv2.minEnclosingCircle(largest)
        
        if min_radius <= radius <= max_radius and cv2.contourArea(largest) > 50:
            return int(center_x), int(center_y), int(radius)
    
    return None

def calibrate_eyes_from_video(cap, sample_rate=30):
    """Calibrate eye regions and reference measurements from video
    
    Args:
        cap: OpenCV VideoCapture object
        sample_rate: Sample every N frames
    
    Returns:
        dict: Contains eye ROI bounds, calibration data, etc.
    """
    print("📊 Calibration pass: analyzing eye regions...")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    left_pupil_sizes = []
    right_pupil_sizes = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        if frame_count % sample_rate != 0:
            continue
        
        # Detect faces
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        
        if len(faces) == 0:
            continue
        
        # Use the first (largest) face
        x, y, w, h = faces[0]
        face_roi = frame[y:y+h, x:x+w]
        
        # Detect eyes within face
        eyes = eye_cascade.detectMultiScale(face_roi)
        
        if len(eyes) >= 2:
            # Typically left eye is first, right eye is second
            for i, (eye_x, eye_y, eye_w, eye_h) in enumerate(eyes[:2]):
                eye_roi = face_roi[eye_y:eye_y+eye_h, eye_x:eye_x+eye_w]
                
                # Try Hough first, fall back to threshold
                pupil = detect_pupil_hough(eye_roi)
                if pupil is None:
                    pupil = detect_pupil_threshold(eye_roi)
                
                if pupil:
                    px, py, pr = pupil
                    if i == 0:
                        left_pupil_sizes.append(pr)
                    else:
                        right_pupil_sizes.append(pr)
    
    # Calculate average pupil sizes
    calibration = {
        'frame_width': frame_width,
        'frame_height': frame_height,
        'fps': fps,
        'total_frames': total_frames,
        'left_avg_pupil_radius': np.mean(left_pupil_sizes) if left_pupil_sizes else 10,
        'right_avg_pupil_radius': np.mean(right_pupil_sizes) if right_pupil_sizes else 10,
        'left_pupil_std': np.std(left_pupil_sizes) if left_pupil_sizes else 2,
        'right_pupil_std': np.std(right_pupil_sizes) if right_pupil_sizes else 2,
    }
    
    print(f"✓ Calibration complete:")
    print(f"   Left eye - avg pupil radius: {calibration['left_avg_pupil_radius']:.1f}px (std: {calibration['left_pupil_std']:.1f}px)")
    print(f"   Right eye - avg pupil radius: {calibration['right_avg_pupil_radius']:.1f}px (std: {calibration['right_pupil_std']:.1f}px)")
    print(f"   Frame dimensions: {frame_width}x{frame_height}")
    print(f"   Total frames: {total_frames}\n")
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    return calibration

def calculate_eye_angle_from_pupil(face_roi, face_roi_offset, eye_roi_bounds, eye_side='left'):
    """Calculate eye angle from pupil position within eye ROI
    
    Args:
        face_roi: The ROI containing the face
        face_roi_offset: (x, y) offset of face_roi in the full frame
        eye_roi_bounds: (x, y, w, h) of the eye within face_roi
        eye_side: 'left' or 'right'
    
    Returns:
        tuple: (horizontal_angle, vertical_angle) in degrees
    """
    eye_x, eye_y, eye_w, eye_h = eye_roi_bounds
    eye_roi = face_roi[eye_y:eye_y+eye_h, eye_x:eye_x+eye_w]
    
    # Detect pupil
    pupil = detect_pupil_hough(eye_roi)
    if pupil is None:
        pupil = detect_pupil_threshold(eye_roi)
    
    if pupil is None:
        return None, None
    
    # Pupil center in eye_roi coordinates
    px_eye, py_eye, pupil_r = pupil
    
    # Eye center (geometric center of the eye ROI)
    eye_center_x = eye_w / 2.0
    eye_center_y = eye_h / 2.0
    
    # Displacement from eye center to pupil (in pixels)
    dx = px_eye - eye_center_x
    dy = py_eye - eye_center_y
    
    # Estimate eyeball depth from pupil radius and expected physiology
    # Pupil is typically 3-4mm diameter, visible eye width is ~30mm
    # This gives us a rough depth estimate
    estimated_eyeball_depth = eye_w * 0.6  # Heuristic: depth ≈ 60% of eye width
    
    # Calculate angles
    horizontal_angle = math.degrees(math.atan2(dx, estimated_eyeball_depth))
    vertical_angle = math.degrees(math.atan2(dy, estimated_eyeball_depth))
    
    return horizontal_angle, vertical_angle

def process_video(video_path, output_csv=None, output_json=None, buffer_size=30):
    """Process video file using OpenCV Haar Cascades + Hough Circle Detection
    
    Args:
        video_path: Path to video file
        output_csv: Output CSV file path
        output_json: Output JSON file path
        buffer_size: Number of frames to buffer
    """
    cap = cv2.VideoCapture(video_path)
    
    # Enable hardware video decoder if available
    try:
        cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_D3D11)
        print("✓ Hardware video decoder: Attempting D3D11/NVDEC...\n")
    except:
        pass
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval_ms = 1000 / fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Extract first frame and get resolution from actual frame data
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame")
        return
    
    frame_height, frame_width = first_frame.shape[:2]  # OpenCV returns (height, width, channels)
    
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print(f"Video FPS: {fps}")
    print(f"Frame interval: {frame_interval_ms:.2f}ms")
    print(f"Total frames: {total_frames}")
    print(f"Resolution (from frame data): {frame_width}x{frame_height}\n")
    
    # CALIBRATION PASS
    calibration = calibrate_eyes_from_video(cap, sample_rate=10)
    
    # Frame buffer
    frame_queue = Queue(maxsize=buffer_size)
    
    def producer():
        """Read frames from video into queue"""
        frame_count = 0
        timestamp_ms = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    frame_queue.put(None)
                    break
                
                frame_count += 1
                frame_queue.put((frame_count, frame, timestamp_ms))
                timestamp_ms += frame_interval_ms
        except Exception as e:
            print(f"Producer error: {e}")
            frame_queue.put(None)
    
    # Start producer thread
    reader_thread = Thread(target=producer, daemon=True)
    reader_thread.start()
    
    # PROCESSING PASS
    frame_count = 0
    eye_data = []
    start_time = time.time()
    
    while True:
        item = frame_queue.get()
        if item is None:
            break
        
        frame_num, frame, timestamp_ms = item
        frame_count += 1
        
        # Detect faces
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        
        if len(faces) == 0:
            continue
        
        # Use first face
        fx, fy, fw, fh = faces[0]
        face_roi = frame[fy:fy+fh, fx:fx+fw]
        
        # Detect eyes
        eyes = eye_cascade.detectMultiScale(face_roi)
        
        if len(eyes) < 2:
            continue
        
        # Process eyes (left and right)
        angles = {'left': (None, None), 'right': (None, None)}
        
        for i, (ex, ey, ew, eh) in enumerate(eyes[:2]):
            side = 'left' if i == 0 else 'right'
            h_angle, v_angle = calculate_eye_angle_from_pupil(
                face_roi, (fx, fy), (ex, ey, ew, eh), side
            )
            angles[side] = (h_angle, v_angle)
        
        if frame_count % 30 == 0:
            print(f"Frame {frame_num}/{total_frames}:")
            if angles['left'][0] is not None:
                print(f"  Left Eye  - Horizontal: {angles['left'][0]:7.2f}°, Vertical: {angles['left'][1]:7.2f}°")
            if angles['right'][0] is not None:
                print(f"  Right Eye - Horizontal: {angles['right'][0]:7.2f}°, Vertical: {angles['right'][1]:7.2f}°")
        
        # Store data
        left_h, left_v = angles['left']
        right_h, right_v = angles['right']
        
        if left_h is not None and right_h is not None:
            eye_data.append({
                'frame': frame_num,
                'timestamp_ms': timestamp_ms,
                'left_horizontal': left_h,
                'left_vertical': left_v,
                'right_horizontal': right_h,
                'right_vertical': right_v
            })
    
    elapsed_time = time.time() - start_time
    cap.release()
    
    print(f"\n✓ Processing complete. Processed {frame_count} frames in {elapsed_time:.2f}s")
    print(f"  Speed: {frame_count / elapsed_time:.1f} fps (processing speed)")
    print(f"  Speedup vs realtime: {(frame_count / elapsed_time) / fps:.2f}x\n")
    
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
    
    buffer_size = 100
    
    process_video(
        video_file,
        output_csv="eye_tracking_data_v2.csv",
        output_json=None,
        buffer_size=buffer_size
    )
