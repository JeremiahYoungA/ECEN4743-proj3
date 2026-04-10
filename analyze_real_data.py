#!/usr/bin/env python3
"""
Analyze real saccade data to understand scale and characteristics
"""
import numpy as np
import pandas as pd

# Load data
df = pd.read_csv('eye_tracking_data.csv')

print("="*70)
print("ACTUAL EYE TRACKING DATA ANALYSIS")
print("="*70)

print(f"\nDataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")

# Extract frames 3000-3200 (where we see the saccade in your image)
frame_start = 3000
frame_end = 3200
saccade_data = df.iloc[frame_start:frame_end].reset_index(drop=True)

print(f"\nAnalyzing frames {frame_start}-{frame_end}:")
print(f"  Samples: {len(saccade_data)}")

# Get timestamp info to calculate frame rate
timestamps = saccade_data['Timestamp'].values if 'Timestamp' in df.columns else None

if timestamps is not None:
    dt_array = np.diff(timestamps)
    avg_dt = np.mean(dt_array)
    frame_rate = 1.0 / avg_dt
    print(f"  Time span: {timestamps[-1] - timestamps[0]:.3f} seconds")
    print(f"  Estimated frame rate: {frame_rate:.1f} Hz")
    print(f"  Average dt: {avg_dt*1000:.2f} ms")
else:
    print(f"  (No timestamp column - assuming uniform frame rate)")

# Find position and velocity columns
print(f"\nSearching for eye position/velocity columns:")
for col in df.columns:
    if 'Left' in col or 'Right' in col:
        print(f"  - {col}")

# Look for the main position columns
pos_cols = [col for col in df.columns if 'Position' in col or 'X Position' in col or 'Left' in col.strip() and 'Position' in col]
print(f"\nPosition columns found: {pos_cols}")

if df.columns[0] == 'Frame':
    print(f"\nFrame column detected")
    frames = df['Frame'].values
    print(f"  Frames: {frames[0]} to {frames[-1]}")

# Check what actual columns we have
print(f"\nActual columns in CSV:")
for i, col in enumerate(df.columns[:10]):
    print(f"  {i}: {col}")
    print(f"      Sample values: {df[col].iloc[frame_start:frame_start+3].values}")

# Try to find position data
print(f"\n--- SACCADE CHARACTERISTICS (frames {frame_start}-{frame_end}) ---")

# Most likely column names based on typical eye tracking formats
left_pos_col = None
right_pos_col = None
for col in df.columns:
    if 'L' in col and 'X' in col:
        left_pos_col = col
    if 'R' in col and 'X' in col:
        right_pos_col = col

if left_pos_col:
    left_pos = saccade_data[left_pos_col].values
    left_vel = np.gradient(left_pos)  # Velocity in °/frame
    print(f"\nLeft Eye:")
    print(f"  Position range: [{np.min(left_pos):.4f}, {np.max(left_pos):.4f}]°")
    print(f"  Position change: {np.max(left_pos) - np.min(left_pos):.4f}°")
    print(f"  Peak velocity: {np.max(np.abs(left_vel)):.4f} °/frame")
    print(f"  Velocity range: [{np.min(left_vel):.4f}, {np.max(left_vel):.4f}] °/frame")

if right_pos_col:
    right_pos = saccade_data[right_pos_col].values
    right_vel = np.gradient(right_pos)
    print(f"\nRight Eye:")
    print(f"  Position range: [{np.min(right_pos):.4f}, {np.max(right_pos):.4f}]°")
    print(f"  Position change: {np.max(right_pos) - np.min(right_pos):.4f}°")
    print(f"  Peak velocity: {np.max(np.abs(right_vel)):.4f} °/frame")
    print(f"  Velocity range: [{np.min(right_vel):.4f}, {np.max(right_vel):.4f}] °/frame")

print(f"\n--- SCALE ANALYSIS ---")
print(f"If frame rate = 60 Hz (16.67 ms/frame):")
print(f"  0.05 °/frame →  3.0 °/second")
print(f"  0.06 °/frame →  3.6 °/second")
print(f"")
print(f"If frame rate = 120 Hz (8.33 ms/frame):")
print(f"  0.05 °/frame →  6.0 °/second")
print(f"  0.06 °/frame →  7.2 °/second")
print(f"")
print(f"If frame rate = 200 Hz (5 ms/frame):")
print(f"  0.05 °/frame → 10.0 °/second")
print(f"  0.06 °/frame → 12.0 °/second")
