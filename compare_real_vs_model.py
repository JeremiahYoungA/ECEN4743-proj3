#!/usr/bin/env python3
"""
Extract real saccade and compare with model
"""
import numpy as np
import pandas as pd

# Load data
df = pd.read_csv('eye_tracking_data.csv')

# Calculate frame rate from timestamps
timestamps = df['timestamp_ms'].values / 1000.0  # Convert to seconds
dt_array = np.diff(timestamps)
avg_dt = np.mean(dt_array)
frame_rate = 1.0 / avg_dt

print("="*70)
print("REAL SACCADE EXTRACTION & ANALYSIS")
print("="*70)

print(f"\nFrame rate: {frame_rate:.1f} Hz")
print(f"Average inter-frame time: {avg_dt*1000:.2f} ms")
print(f"Total recording time: {timestamps[-1] - timestamps[0]:.2f} seconds")
print(f"Total frames: {len(df)}")

# Extract the saccade region (frames 3000-3100 based on your plot)
frame_start = 3000
frame_end = 3100
saccade_data = df.iloc[frame_start:frame_end].reset_index(drop=True)
saccade_times = saccade_data['timestamp_ms'].values / 1000.0
saccade_times = saccade_times - saccade_times[0]  # Start from 0

# Get position data
left_horiz = saccade_data['left_horizontal'].values
right_horiz = saccade_data['right_horizontal'].values

# Calculate velocities and accelerations
left_vel = np.gradient(left_horiz, avg_dt)  # °/second
right_vel = np.gradient(right_horiz, avg_dt)  # °/second

left_acc = np.gradient(left_vel, avg_dt)  # °/s²
right_acc = np.gradient(right_vel, avg_dt)  # °/s²

print(f"\n--- REAL SACCADE (frames {frame_start}-{frame_end}) ---")
print(f"Duration: {saccade_times[-1]*1000:.1f} ms")
print(f"Samples: {len(saccade_data)}")

print(f"\nLEFT EYE:")
print(f"  Position:")
print(f"    Range: [{np.min(left_horiz):.4f}°, {np.max(left_horiz):.4f}°]")
print(f"    Change: {np.max(left_horiz) - np.min(left_horiz):.4f}°")
print(f"  Velocity:")
print(f"    Range: [{np.min(left_vel):.2f}°/s, {np.max(left_vel):.2f}°/s]")
print(f"    Peak: {np.max(np.abs(left_vel)):.2f}°/s")
print(f"    Peak at time: {saccade_times[np.argmax(np.abs(left_vel))]*1000:.1f} ms")
print(f"  Acceleration:")
print(f"    Range: [{np.min(left_acc):.1f}°/s², {np.max(left_acc):.1f}°/s²]")

print(f"\nRIGHT EYE:")
print(f"  Position:")
print(f"    Range: [{np.min(right_horiz):.4f}°, {np.max(right_horiz):.4f}°]")
print(f"    Change: {np.max(right_horiz) - np.min(right_horiz):.4f}°")
print(f"  Velocity:")
print(f"    Range: [{np.min(right_vel):.2f}°/s, {np.max(right_vel):.2f}°/s]")
print(f"    Peak: {np.max(np.abs(right_vel)):.2f}°/s")
print(f"    Peak at time: {saccade_times[np.argmax(np.abs(right_vel))]*1000:.1f} ms")
print(f"  Acceleration:")
print(f"    Range: [{np.min(right_acc):.1f}°/s², {np.max(right_acc):.1f}°/s²]")

print(f"\n" + "="*70)
print("COMPARISON: REAL vs MODEL")
print("="*70)

print(f"""\nREAL DATA CHARACTERISTICS:
  Peak velocity:        {np.max(np.abs(left_vel)):.1f}°/s (left) or {np.max(np.abs(right_vel)):.1f}°/s (right)
  Saccade amplitude:    {np.max(left_horiz) - np.min(left_horiz):.3f}° (left) or {np.max(right_horiz) - np.min(right_horiz):.3f}° (right)
  Saccade duration:     {saccade_times[-1]*1000:.0f} ms
  Peak acceleration:    {np.max(np.abs(left_acc)):.0f}°/s² (left) or {np.max(np.abs(right_acc)):.0f}°/s² (right)

CURRENT MODEL OUTPUT:
  Peak velocity:        0.2°/s
  Target amplitude:     15°
  Saccade duration:     ~20 ms
  Peak acceleration:    ~71°/s²

SCALE FACTOR NEEDED:
  Velocity scaling:     {(np.max(np.abs(left_vel)) / 0.2):.1f}x
  Amplitude scaling:    {((np.max(left_horiz) - np.min(left_horiz)) / 15):.3f}x
""")

print("KEY INSIGHT:")
print("  - Real saccades are SLOW (5-10 °/s), not fast (500+ °/s in Robinson)")
print("  - Real amplitudes are SMALL (0.3-0.5°), not large (15°)")
print("  - Your data appears to be smooth pursuit or drift, not ballistic saccades!")
print("  - We need to adjust model parameters to match YOUR hardware, not textbook values")
