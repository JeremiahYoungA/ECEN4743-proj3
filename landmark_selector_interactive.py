"""
Interactive MediaPipe Landmark Selector
Click on landmarks to identify their indices and measure eye corners
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

# Global state
current_frame = None
landmarks = None
frame_width = 0
frame_height = 0
selected_indices = {"left_inner": None, "left_outer": None, "right_inner": None, "right_outer": None}
current_selection = None
selection_order = ["left_inner", "left_outer", "right_inner", "right_outer"]
selection_index = 0

def mouse_callback(event, x, y, flags, param):
    """Handle mouse clicks to select landmarks"""
    global current_frame, landmarks, selected_indices, current_selection, selection_index
    
    if event != cv2.EVENT_LBUTTONDOWN or landmarks is None:
        return
    
    # Find closest landmark to click
    min_dist = float('inf')
    closest_idx = -1
    
    for i, lm in enumerate(landmarks):
        lm_x = int(lm.x * frame_width)
        lm_y = int(lm.y * frame_height)
        dist = np.sqrt((lm_x - x)**2 + (lm_y - y)**2)
        
        if dist < min_dist:
            min_dist = dist
            closest_idx = i
    
    if min_dist < 30:  # Within 30 pixels
        selection_name = selection_order[selection_index]
        selected_indices[selection_name] = closest_idx
        current_selection = selection_name
        print(f"\n✓ Selected {selection_name}: Index {closest_idx}")
        print(f"  Landmark coords: x={landmarks[closest_idx].x:.4f}, y={landmarks[closest_idx].y:.4f}")
        selection_index = (selection_index + 1) % len(selection_order)

def draw_frame():
    """Draw current frame with landmarks and selections"""
    global current_frame, landmarks, frame_width, frame_height, selected_indices
    
    if current_frame is None or landmarks is None:
        return
    
    display = current_frame.copy()
    
    # Draw ALL landmarks as small dots
    for i, lm in enumerate(landmarks):
        x = int(lm.x * frame_width)
        y = int(lm.y * frame_height)
        cv2.circle(display, (x, y), 2, (100, 100, 100), -1)  # Gray dots
    
    # Highlight selected landmarks
    colors = {
        "left_inner": (0, 255, 0),    # Green
        "left_outer": (0, 255, 255),   # Cyan
        "right_inner": (255, 0, 0),    # Blue
        "right_outer": (255, 0, 255)   # Magenta
    }
    
    for name, idx in selected_indices.items():
        if idx is not None:
            lm = landmarks[idx]
            x = int(lm.x * frame_width)
            y = int(lm.y * frame_height)
            cv2.circle(display, (x, y), 8, colors[name], 2)
            cv2.putText(display, f"{name}\n#{idx}", (x + 10, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[name], 1)
    
    # Draw connecting lines if both corners selected for each eye
    if selected_indices["left_inner"] is not None and selected_indices["left_outer"] is not None:
        l_inner_idx = selected_indices["left_inner"]
        l_outer_idx = selected_indices["left_outer"]
        lm1 = landmarks[l_inner_idx]
        lm2 = landmarks[l_outer_idx]
        x1, y1 = int(lm1.x * frame_width), int(lm1.y * frame_height)
        x2, y2 = int(lm2.x * frame_width), int(lm2.y * frame_height)
        cv2.line(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
        width = abs(lm2.x - lm1.x)
        cv2.putText(display, f"LEFT: {width:.4f}", (x1, y1 - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
    if selected_indices["right_inner"] is not None and selected_indices["right_outer"] is not None:
        r_inner_idx = selected_indices["right_inner"]
        r_outer_idx = selected_indices["right_outer"]
        lm1 = landmarks[r_inner_idx]
        lm2 = landmarks[r_outer_idx]
        x1, y1 = int(lm1.x * frame_width), int(lm1.y * frame_height)
        x2, y2 = int(lm2.x * frame_width), int(lm2.y * frame_height)
        cv2.line(display, (x1, y1), (x2, y2), (255, 0, 255), 2)
        width = abs(lm2.x - lm1.x)
        cv2.putText(display, f"RIGHT: {width:.4f}", (x1, y1 - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    # Draw instructions
    next_selection = selection_order[selection_index] if selection_index < len(selection_order) else "DONE"
    cv2.putText(display, f"Click on landmarks. Next: {next_selection}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(display, "Press 'c' to clear, 'r' to reset, 'q' to quit", (10, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    return display

def main():
    global current_frame, landmarks, frame_width, frame_height, selected_indices, selection_index
    
    video_path = input("Enter path to video file: ")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        return
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not read frame")
        return
    
    # Convert to RGB and detect landmarks
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_height, frame_width = frame.shape[:2]
    current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = face_landmarker.detect(mp_image)
    
    if not results.face_landmarks:
        print("Error: No face detected in first frame")
        return
    
    landmarks = results.face_landmarks[0]
    
    print(f"\n{'='*70}")
    print(f"INTERACTIVE LANDMARK SELECTOR")
    print(f"{'='*70}")
    print(f"Total landmarks detected: {len(landmarks)}")
    print(f"Frame size: {frame_width}x{frame_height}")
    print(f"\nInstructions:")
    print(f"1. Click on LEFT INNER corner (closer to nose)")
    print(f"2. Click on LEFT OUTER corner (farther from nose)")
    print(f"3. Click on RIGHT INNER corner (closer to nose)")
    print(f"4. Click on RIGHT OUTER corner (farther from nose)")
    print(f"\nColor coding in display:")
    print(f"  🟢 LEFT INNER  (green)")
    print(f"  🔵 LEFT OUTER  (cyan)")
    print(f"  🔴 RIGHT INNER (blue)")
    print(f"  🟣 RIGHT OUTER (magenta)")
    print(f"\nKeybinds:")
    print(f"  'c' - Clear current selection")
    print(f"  'r' - Reset all selections")
    print(f"  'q' - Exit")
    print(f"{'='*70}\n")
    
    # Create window
    window_name = "Click on Eye Landmarks"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)
    
    while True:
        display = draw_frame()
        if display is not None:
            cv2.imshow(window_name, display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('c'):
            selected_indices[selection_order[selection_index - 1]] = None
            selection_index = max(0, selection_index - 1)
            print(f"Cleared {selection_order[selection_index]}")
        elif key == ord('r'):
            selected_indices = {"left_inner": None, "left_outer": None, "right_inner": None, "right_outer": None}
            selection_index = 0
            print("Reset all selections")
        elif key == ord('s'):  # Save result
            if all(selected_indices.values()):
                print("\n" + "="*70)
                print("SELECTED INDICES:")
                print("="*70)
                print(f"Left eye:  {selected_indices['left_inner']} → {selected_indices['left_outer']}")
                print(f"Right eye: {selected_indices['right_inner']} → {selected_indices['right_outer']}")
                
                # Calculate widths
                left_w = abs(landmarks[selected_indices['left_outer']].x - landmarks[selected_indices['left_inner']].x)
                right_w = abs(landmarks[selected_indices['right_outer']].x - landmarks[selected_indices['right_inner']].x)
                
                print(f"\nMeasured widths (normalized):")
                print(f"  Left:  {left_w:.4f}")
                print(f"  Right: {right_w:.4f}")
                print(f"  Ratio: {right_w/left_w:.2f}x")
                print(f"\n✓ Use these indices in eye-tracking.py:")
                print(f"  LEFT_LEFT_CORNER = {selected_indices['left_inner']}")
                print(f"  LEFT_RIGHT_CORNER = {selected_indices['left_outer']}")
                print(f"  RIGHT_LEFT_CORNER = {selected_indices['right_inner']}")
                print(f"  RIGHT_RIGHT_CORNER = {selected_indices['right_outer']}")
                print("="*70)
            else:
                print("Not all landmarks selected yet!")
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
