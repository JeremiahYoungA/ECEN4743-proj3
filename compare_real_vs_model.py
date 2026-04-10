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

# Get position data (LEFT EYE ONLY)
eye_horiz = saccade_data['left_horizontal'].values

# Calculate velocities and accelerations
eye_vel = np.gradient(eye_horiz, avg_dt)  # °/second
eye_acc = np.gradient(eye_vel, avg_dt)  # °/s²

print(f"\n--- REAL SACCADE (frames {frame_start}-{frame_end}) ---")
print(f"Duration: {saccade_times[-1]*1000:.1f} ms")
print(f"Samples: {len(saccade_data)}")

print(f"\nLEFT EYE (ANALYSIS):")
print(f"  Position:")
print(f"    Range: [{np.min(eye_horiz):.4f}°, {np.max(eye_horiz):.4f}°]")
print(f"    Change: {np.max(eye_horiz) - np.min(eye_horiz):.4f}°")
print(f"  Velocity:")
print(f"    Range: [{np.min(eye_vel):.2f}°/s, {np.max(eye_vel):.2f}°/s]")
print(f"    Peak: {np.max(np.abs(eye_vel)):.2f}°/s")
print(f"    Peak at time: {saccade_times[np.argmax(np.abs(eye_vel))]*1000:.1f} ms")
print(f"  Acceleration:")
print(f"    Range: [{np.min(eye_acc):.1f}°/s², {np.max(eye_acc):.1f}°/s²]")

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

print(f"\nREAL DATA CHARACTERISTICS (LEFT EYE):")
print(f"  Peak velocity:        {np.max(np.abs(eye_vel)):.1f}°/s")
print(f"  Saccade amplitude:    {np.max(eye_horiz) - np.min(eye_horiz):.3f}°")
print(f"  Saccade duration:     {saccade_times[-1]*1000:.0f} ms")
print(f"  Peak acceleration:    {np.max(np.abs(eye_acc)):.0f}°/s²")
print(f"\nCURRENT MODEL OUTPUT:")
print(f"  Peak velocity:        0.2°/s")
print(f"  Target amplitude:     15°")
print(f"  Saccade duration:     ~20 ms")
print(f"  Peak acceleration:    ~71°/s²")
print(f"\nSCALE FACTOR NEEDED:")
print(f"  Velocity scaling:     {(np.max(np.abs(eye_vel)) / 0.2):.1f}x")
print(f"  Amplitude scaling:    {((np.max(eye_horiz) - np.min(eye_horiz)) / 15):.3f}x")

print("KEY INSIGHT:")
print("  - Real saccades are SLOW (5-10 °/s), not fast (500+ °/s in Robinson)")
print("  - Real amplitudes are SMALL (0.3-0.5°), not large (15°)")
print("  - Your data appears to be smooth pursuit or drift, not ballistic saccades!")
print("  - We need to adjust model parameters to match YOUR hardware, not textbook values")
