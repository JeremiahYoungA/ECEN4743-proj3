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

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request

# Check for GPU availability using nvidia-smi (no PyTorch needed)
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
# NOTE: GPU acceleration requires MediaPipe GPU package: pip install mediapipe-gpu
# Standard MediaPipe uses CPU with TFLite optimizations (XNNPACK)
print("Initializing face landmarker...")
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO)
face_landmarker = FaceLandmarker.create_from_options(options)
print("✓ Face landmarker ready (using XNNPACK CPU optimization)\n")

# Eye landmark indices for MediaPipe Face Landmarker
LEFT_IRIS = 468  # Left iris center
RIGHT_IRIS = 469  # Right iris center

def measure_eye_width(landmarks):
    """Measure left and right eye widths separately from landmarks (horizontal distance only)
    
    Important: Uses only X-coordinate (horizontal) distance, not Euclidean distance.
    This avoids mixing normalized X and Y dimensions that scale differently.
    
    Returns:
        tuple: (left_eye_width, right_eye_width)
    
    Note: Indices determined from interactive landmark selector
    - Left eye:  173 (inner) → 33 (outer)
    - Right eye: 398 (inner) → 263 (outer)
    """
    # Left eye corners (indices from interactive selection)
    left_eye_inner = landmarks[173]
    left_eye_outer = landmarks[33]
    
    # Right eye corners (indices from interactive selection)
    right_eye_inner = landmarks[398]
    right_eye_outer = landmarks[263]
    
    # Use only horizontal distance (X component) for each eye separately
    # This is physiologically correct: eye width is measured horizontally
    left_eye_width = abs(left_eye_outer.x - left_eye_inner.x)
    right_eye_width = abs(right_eye_outer.x - right_eye_inner.x)
    
    return left_eye_width, right_eye_width


def measure_eye_width_debug(landmarks):
    """Debug version: shows landmark coordinates for verification
    
    MediaPipe Face Landmark indices (verify these are correct):
    - Left eye: inner (130), outer (243) corners
    - Right eye: inner (359), outer (133) corners
    """
    # Left eye corners
    l_inner = landmarks[130]
    l_outer = landmarks[243]
    
    # Right eye corners
    r_inner = landmarks[359]
    r_outer = landmarks[133]
    
    left_width = abs(l_outer.x - l_inner.x)
    right_width = abs(r_outer.x - r_inner.x)
    
    print(f"    LEFT EYE (idx 130 vs 243): L_inner={l_inner.x:.4f}, L_outer={l_outer.x:.4f} → width={left_width:.4f}")
    print(f"    RIGHT EYE (idx 359 vs 133): R_inner={r_inner.x:.4f}, R_outer={r_outer.x:.4f} → width={right_width:.4f}")
    
    return left_width, right_width

def calibrate_from_video(cap, face_landmarker_cal, sample_rate=30):
    """Two-pass calibration: measure eye widths separately for left and right eye
    
    Args:
        cap: OpenCV VideoCapture object
        face_landmarker_cal: MediaPipe FaceLandmarker for calibration
        sample_rate: Measure every N frames to speed up calibration
    
    Returns:
        dict: Contains calibration data for left and right eyes separately
    """
    print("📊 Calibration pass: measuring eye widths from entire video (per-eye)...")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Use image-mode detection for calibration (doesn't maintain timestamp state)
    # This avoids interfering with video-mode detection later
    try:
        face_landmarker_image = FaceLandmarker.create_from_options(
            FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=VisionRunningMode.IMAGE))
    except:
        # Fallback to provided landmarker if fresh instance fails
        face_landmarker_image = face_landmarker_cal
    
    left_eye_widths = []
    right_eye_widths = []
    frame_count = 0
    timestamp_ms = 0
    frame_interval_ms = 1000 / fps
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Sample every Nth frame for speed
        if frame_count % sample_rate == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = face_landmarker_image.detect(mp_image)  # Use image mode to avoid timestamp tracking
            
            if results.face_landmarks:
                left_w, right_w = measure_eye_width(results.face_landmarks[0])
                left_eye_widths.append(left_w)
                right_eye_widths.append(right_w)
                
                # DEBUG: Show frame-by-frame measurements (sample first 10 frames)
                if len(left_eye_widths) <= 10:
                    print(f"  Frame {frame_count}: LEFT={left_w:.4f}, RIGHT={right_w:.4f}, Ratio={right_w/left_w:.2f}x")
        
        timestamp_ms += frame_interval_ms
    
    # Calculate statistics for each eye separately
    calibration = {
        'fps': fps,
        'total_frames': total_frames,
        'frame_count': frame_count
    }
    
    if left_eye_widths and right_eye_widths:
        # Left eye stats
        calibration['left_avg_width'] = np.mean(left_eye_widths)
        calibration['left_std_width'] = np.std(left_eye_widths)
        calibration['left_min_width'] = np.min(left_eye_widths)
        calibration['left_max_width'] = np.max(left_eye_widths)
        
        # Right eye stats
        calibration['right_avg_width'] = np.mean(right_eye_widths)
        calibration['right_std_width'] = np.std(right_eye_widths)
        calibration['right_min_width'] = np.min(right_eye_widths)
        calibration['right_max_width'] = np.max(right_eye_widths)
        
        # Asymmetry analysis
        asymmetry_pct = abs(calibration['left_avg_width'] - calibration['right_avg_width']) / ((calibration['left_avg_width'] + calibration['right_avg_width']) / 2) * 100
        
        print(f"✓ Calibration complete:")
        print(f"   Measured {len(left_eye_widths)} frames (every {sample_rate}th frame)")
        print(f"")
        print(f"   LANDMARK INDICES BEING USED (from interactive selection):")
        print(f"      Left eye:  173 (inner) → 33 (outer)")
        print(f"      Right eye: 398 (inner) → 263 (outer)")
        print(f"")
        print(f"   LEFT EYE:")
        print(f"      Average width: {calibration['left_avg_width']:.4f}")
        print(f"      Std deviation: {calibration['left_std_width']:.4f}")
        print(f"      Range: {calibration['left_min_width']:.4f} - {calibration['left_max_width']:.4f}")
        print(f"")
        print(f"   RIGHT EYE:")
        print(f"      Average width: {calibration['right_avg_width']:.4f}")
        print(f"      Std deviation: {calibration['right_std_width']:.4f}")
        print(f"      Range: {calibration['right_min_width']:.4f} - {calibration['right_max_width']:.4f}")
        print(f"")
        print(f"   ASYMMETRY: {asymmetry_pct:.1f}% difference between eyes")
        print(f"   ⚠️  IF ASYMMETRY > 50%: Landmark indices may be incorrect!\n")
    else:
        print("⚠ No faces detected during calibration\n")
        calibration['left_avg_width'] = 0.05
        calibration['right_avg_width'] = 0.05
    
    # Reset video to beginning for processing pass
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    return calibration

def calculate_eyeball_radius_from_width(eye_width, radius_multiplier=0.5):
    """Convert eye width to eyeball radius using multiplier
    
    Physiological basis (Zhang et al. 2012):
    - eye_width ≈ 2 × eyeball_radius
    - eyeball_radius = eye_width × multiplier (default 0.5)
    """
    return eye_width * radius_multiplier

def calculate_eye_angle(landmarks, left_corner_idx, right_corner_idx, iris_idx, eyeball_radius, frame_w, frame_h):
    """Calculate angular position of eye (pitch/yaw) using geometric normalization
    
    Fixes three key scaling issues:
    1. Pivot Point: Uses true eye center (midpoint of corners) not corner landmark
    2. Aspect Ratio: Converts to pixel space to maintain 1:1 aspect ratio
    3. Radius Scaling: Uses separate radius for each axis to maintain sphere geometry
       across different video orientations (landscape vs portrait)
    
    Args:
        landmarks: Face landmarks from MediaPipe
        left_corner_idx: Index of left eye corner
        right_corner_idx: Index of right eye corner
        iris_idx: Index of iris center
        eyeball_radius: Calibrated radius (normalized coordinates)
        frame_w: Video frame width in pixels
        frame_h: Video frame height in pixels
    
    Returns:
        tuple: (horizontal_angle in degrees, vertical_angle in degrees)
    """
    # 1. Get landmarks
    l_corner = landmarks[left_corner_idx]
    r_corner = landmarks[right_corner_idx]
    iris = landmarks[iris_idx]
    
    # 2. Calculate the TRUE center of the eye (midpoint of corners)
    # This acts as the 'zero' point for the rotation (not the corner)
    center_x = (l_corner.x + r_corner.x) / 2
    center_y = (l_corner.y + r_corner.y) / 2
    
    # 3. Convert displacement to Pixel Space
    # This removes the normalization squashing and maintains aspect ratio
    dx_px = (iris.x - center_x) * frame_w
    dy_px = (iris.y - center_y) * frame_h
    
    # 4. Convert calibrated radius to Pixel Space with proper aspect ratio
    # KEY FIX: Use frame_w for horizontal angles, frame_h for vertical angles
    # This maintains eyeball sphere geometry across different video orientations
    # For portrait video (1080x1920): prevents vertical angles from being artificially inflated
    radius_px_h = eyeball_radius * frame_w  # Horizontal angle calculation
    radius_px_v = eyeball_radius * frame_h  # Vertical angle calculation
    
    # 5. Calculate angles with proper per-axis scaling
    # Both horizontal and vertical use their respective radius scaling
    horizontal_angle = math.degrees(math.atan2(dx_px, radius_px_h))  # Yaw
    vertical_angle = math.degrees(math.atan2(dy_px, radius_px_v))    # Pitch
    
    return horizontal_angle, vertical_angle

def process_video(video_path, output_csv=None, output_json=None, use_gpu=True, buffer_size=30, use_hardware_decoder=True, radius_multiplier=0.5):
    """Process video file with GPU acceleration + multithreaded frame buffering for faster processing
    
    Args:
        video_path: Path to video file
        output_csv: Output CSV file path
        output_json: Output JSON file path
        use_gpu: Enable GPU acceleration (CUDA if available)
        buffer_size: Number of frames to buffer (larger = more memory, faster processing)
        use_hardware_decoder: Use GPU hardware video decoder (NVIDIA NVDEC/AMF/VT) if available
        radius_multiplier: Eyeball radius multiplier (default 0.5 based on Zhang et al. 2012)
                          Adjusted for geometric normalization in pixel space
                          Typical range: 0.45-0.55 for East Asian populations
    
    Key Improvements:
    - Uses true eye center (midpoint of corners) as pivot point
    - Converts to pixel space to eliminate aspect ratio compression
    - Maintains 1:1 aspect ratio for eyeball sphere geometry
    """
    # Create a temporary FaceLandmarker for calibration (uses video mode timestamps)
    # This will be replaced with a fresh instance before main processing
    face_landmarker_calibration = face_landmarker
    
    cap = cv2.VideoCapture(video_path)
    
    # Enable hardware video decoder if requested
    if use_hardware_decoder:
        # Try NVIDIA NVDEC (supported on RTX/GeForce/Quadro)
        try:
            cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_D3D11)
            print("✓ Hardware video decoder: Attempting D3D11/NVDEC...")
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
    
    # Estimate frame size in VRAM
    frame_memory_mb = (frame_width * frame_height * 3) / (1024**2)  # RGB = 3 bytes per pixel
    buffer_memory_mb = frame_memory_mb * buffer_size
    
    print(f"Video FPS: {fps}")
    print(f"Frame interval: {frame_interval_ms:.2f}ms")
    print(f"Total frames: {total_frames}")
    print(f"Resolution (from frame data): {frame_width}x{frame_height} ({frame_memory_mb:.1f}MB per frame)")
    print(f"GPU acceleration: {use_gpu}")
    print(f"Hardware decoder: {use_hardware_decoder}")
    print(f"Frame buffer size: {buffer_size} frames ({buffer_memory_mb:.1f}MB total)\n")
    
    # CALIBRATION PASS: Measure eye widths separately for left and right
    calibration = calibrate_from_video(cap, face_landmarker_calibration, sample_rate=10)
    
    # Calculate separate radius for each eye
    left_eyeball_radius = calculate_eyeball_radius_from_width(calibration['left_avg_width'], radius_multiplier)
    right_eyeball_radius = calculate_eyeball_radius_from_width(calibration['right_avg_width'], radius_multiplier)
    
    print(f"📐 Per-eye calibrated eyeball radius:")
    print(f"   Left eye:  {left_eyeball_radius:.4f} (width={calibration['left_avg_width']:.4f} × multiplier={radius_multiplier})")
    print(f"   Right eye: {right_eyeball_radius:.4f} (width={calibration['right_avg_width']:.4f} × multiplier={radius_multiplier})\n")
    
    # Create FRESH FaceLandmarker instance for main processing
    # This resets MediaPipe's internal timestamp state so timestamps can restart from 0
    print("Initializing face landmarker for main processing pass...")
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO)
    face_landmarker_processor = FaceLandmarker.create_from_options(options)
    print("✓ Face landmarker ready for processing\n")
    
    # Frame buffer: queue of (frame_number, rgb_frame, timestamp_ms)
    frame_queue = Queue(maxsize=buffer_size)
    
    def producer():
        """Read frames from video into queue"""
        frame_count = 0
        timestamp_ms = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    frame_queue.put(None)  # Signal end of video
                    break
                
                frame_count += 1
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_queue.put((frame_count, rgb_frame, timestamp_ms))
                timestamp_ms += frame_interval_ms
        except Exception as e:
            print(f"Producer error: {e}")
            frame_queue.put(None)
    
    # Start frame reader thread (producer)
    reader_thread = Thread(target=producer, daemon=True)
    reader_thread.start()
    
    # PROCESSING PASS: Use calibrated radius for all frames
    frame_count = 0
    eye_data = []
    
    # Process remaining frames
    start_time = time.time()
    
    while True:
        # Get next frame from queue
        item = frame_queue.get()
        if item is None:  # End of video
            break
        
        frame_num, rgb_frame, timestamp_ms = item
        frame_count += 1
        
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect faces and landmarks using fresh processor instance
        results = face_landmarker_processor.detect_for_video(mp_image, int(timestamp_ms))
        
        if results.face_landmarks:
            landmarks = results.face_landmarks[0]
            
            # Get angles for both eyes using individual calibrated radius
            # Left eye: corners at indices 173 (inner) and 33 (outer) - from interactive selection
            left_h, left_v = calculate_eye_angle(landmarks, 173, 33, LEFT_IRIS, left_eyeball_radius, frame_width, frame_height)
            # Right eye: corners at indices 398 (inner) and 263 (outer) - from interactive selection
            right_h, right_v = calculate_eye_angle(landmarks, 398, 263, RIGHT_IRIS, right_eyeball_radius, frame_width, frame_height)
            
            if frame_count % 30 == 0:  # Print every 30 frames to reduce output
                print(f"Frame {frame_num}/{total_frames}:")
                print(f"  Left Eye  - Horizontal: {left_h:7.2f}°, Vertical: {left_v:7.2f}°")
                print(f"  Right Eye - Horizontal: {right_h:7.2f}°, Vertical: {right_v:7.2f}°")
            
            # Store data
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
    
    # Optional: adjust parameters for speed/memory tradeoff
    buffer_size = 100  # Adjust if needed (larger = more memory, faster when GPU can't keep up)
    use_hardware_decoder = True  # Use GPU's hardware video decoder (NVDEC on NVIDIA, AMF on AMD, VT on Intel)
    
    # Population-specific calibration: eyeball radius multiplier
    # Default 0.5 = eye_width / 2 (Zhang et al. 2012 - Caucasian eyes)
    # Range 0.45-0.55 typical for East Asian (Korean, Chinese, Japanese) populations
    # Increase if angles too small, decrease if too large
    # NOTE: Geometric normalization (pixel space conversion) now handles aspect ratio properly
    radius_multiplier = 0.5  # Reduced from 0.8 to allow proper angular range (30-40° per side)
    
    process_video(
        video_file,
        output_csv="eye_tracking_data.csv",
        output_json=None,
        use_gpu=True,  # Auto-detects CUDA if available
        buffer_size=buffer_size,
        use_hardware_decoder=use_hardware_decoder,
        radius_multiplier=radius_multiplier
    )