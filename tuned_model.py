#!/usr/bin/env python3
"""
Tuned saccadic model to match actual eye tracking data
"""
import numpy as np
from saccadic_eye_model import SaccadicEyeModel
import matplotlib.pyplot as plt
import pandas as pd
import os

print("="*70)
print("TUNED SACCADIC MODEL - MATCHING REAL DATA")
print("="*70)

# Real data targets:
# - Peak velocity: 12.8-13.2 °/s
# - Amplitude: 0.57°
# - Duration: ~431 ms
# - Peak acceleration: 792-845 °/s²

# Extended time array for 500 ms simulation at 229.4 Hz sampling
frame_rate = 229.4  # From real data
dt = 1.0 / frame_rate  # 4.36 ms per frame
duration = 0.5  # 500 ms to match real data
t = np.arange(0, duration, dt)

print(f"\nSimulation parameters:")
print(f"  Duration: {duration*1000:.0f} ms")
print(f"  Frame rate: {frame_rate:.1f} Hz")
print(f"  Samples: {len(t)}")
print(f"  dt: {dt*1000:.2f} ms")

# Initialize model with same physical parameters
J_p = 2.3e-7
B_p = 0.1
K_p = 120
K_se = 20
K_lt = 100.0

model = SaccadicEyeModel(
    J_p=J_p, B_p=B_p, K_p=K_p, K_se=K_se, K_lt=K_lt,
    T_ag=0.010, T_ant=0.015, r=0.011
)

print(f"\n--- Running TUNED simulation ---")

# ===== KEY TUNING PARAMETERS =====
ramp_delay = 0.33             # Delay before ramp starts [seconds] - TUNABLE
peak_neural_input = 750         # Neural command magnitude - TUNABLE
ramp_duration_ms = 20           # Ramp rise time [ms] - edit in saccadic_eye_model.py

print(f"Tuning parameters:")
print(f"  Ramp delay: {ramp_delay*1000:.1f} ms")
print(f"  Peak neural input: {peak_neural_input}")
print(f"  Ramp duration: {ramp_duration_ms} ms (steep rise)")

time, position, velocity, acceleration, E_ag, E_ant = model.simulate_saccade(
    t,
    saccade_onset=ramp_delay,      # Delay before ramp starts (TUNABLE)
    saccade_magnitude=0.57,         # 0.57° target
    initial_position=0,             # Model starts at rest at 0
    use_ramp=True,                  # Use LINEAR RAMP (smooth pursuit)
    peak_velocity=peak_neural_input # Neural command magnitude (TUNABLE)
)

# The model needs much larger neural input to match real velocities
# Create simpler focused plot: Neural Input → Model Outputs vs Real Data
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Real data already loaded above, now reload fresh for plotting full segment
export_dir = 'export_20260409_201829_frames_3000-3200'
if os.path.exists(export_dir):
    real_df = pd.read_csv(os.path.join(export_dir, 'time_domain_data.csv'))
    
    # Use ALL frames from the export directory
    frame_start = 0
    frame_end = len(real_df)  # Use all frames, not just first 100
    real_data = real_df.iloc[frame_start:frame_end].reset_index(drop=True)
    
    print(f"\n📊 Loaded {len(real_data)} frames from export folder")
    
    # Create synthetic time array based on frame rate
    real_times = np.arange(len(real_data)) / 229.4
    
    # USE FILTERED POSITION DATA FOR COMPARISON
    real_right_pos = real_data['Right_Position_Filtered'].values
    
    # USE pre-computed velocity from CSV - BUT CONVERT from °/frame to °/s
    frame_rate = 229.4  # Hz
    real_right_vel = real_data['Right_Velocity'].values * frame_rate  # Convert °/frame → °/s
    real_right_acc = real_data['Right_Acceleration'].values * (frame_rate ** 2)  # Convert °/frame² → °/s²
    
    # Calculate offset: shift real data to start at 0 (same as model)
    real_right_offset = real_right_pos - real_right_pos[0]
    
    # CALCULATE SCALE FACTORS FROM ACTUAL REAL DATA
    real_peak_velocity = np.max(np.abs(real_right_vel))
    real_peak_position = np.max(real_right_offset)
    velocity_scale = real_peak_velocity / np.max(np.abs(velocity))
    position_scale = real_peak_position / np.max(position)
    
else:
    # Fall back to main CSV with raw data
    real_df = pd.read_csv('eye_tracking_data.csv')
    frame_start = 3000
    frame_end = 3100
    real_data = real_df.iloc[frame_start:frame_end].reset_index(drop=True)
    real_times = (real_data['timestamp_ms'].values / 1000.0) - (real_data['timestamp_ms'].values[0] / 1000.0)
    real_right_pos = real_data['right_horizontal'].values
    real_right_vel = np.gradient(real_right_pos, np.mean(np.diff(real_times)))
    real_right_acc = np.gradient(real_right_vel, np.mean(np.diff(real_times)))
    real_right_offset = real_right_pos - real_right_pos[0]
    
    # Calculate scale factors from actual real data
    real_peak_velocity = np.max(np.abs(real_right_vel))
    real_peak_position = np.max(real_right_offset)
    velocity_scale = real_peak_velocity / np.max(np.abs(velocity))
    position_scale = real_peak_position / np.max(position)

# Print actual data analysis
print(f"\n--- SCALE ANALYSIS (from ACTUAL real eye data) ---")
print(f"Model outputs:")
print(f"  Peak velocity: {np.max(np.abs(velocity)):.2f} °/s")
print(f"  Final position: {np.max(position):.4f}°")
print(f"  Peak acceleration: {np.max(np.abs(acceleration)):.1f}°/s²")
print(f"\nActual real data (Right Eye):")
print(f"  Peak velocity: {real_peak_velocity:.6f} °/s")
print(f"  Peak position: {real_peak_position:.6f}°")
print(f"\nScale factors (to match real data):")
print(f"  Velocity scale: {velocity_scale:.6f}x")
print(f"  Position scale: {position_scale:.6f}x")

# (0, 0): Neural Input Commands
axes[0, 0].plot(time*1000, E_ag, 'purple', linewidth=2.5, label='Agonist Command (E_ag)')
axes[0, 0].plot(time*1000, E_ant, 'brown', linewidth=2.5, label='Antagonist Command (E_ant)')
axes[0, 0].set_ylabel('Neural Command [a.u.]', fontweight='bold')
axes[0, 0].set_xlabel('Time [ms]', fontweight='bold')
axes[0, 0].set_title('Neural Input Commands to Muscles', fontweight='bold', fontsize=12)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend(loc='upper left')

# (0, 1): Position - Model vs Right Eye
axes[0, 1].plot(time*1000, position, 'b-', linewidth=2.5, label='Model')
axes[0, 1].plot(real_times*1000, real_right_offset, 'r--', linewidth=2, alpha=0.8, label='Right Eye (Real)')
axes[0, 1].set_ylabel('Position [°]', fontweight='bold')
axes[0, 1].set_xlabel('Time [ms]', fontweight='bold')
axes[0, 1].set_title('Position: Model vs Real Data (Right Eye)', fontweight='bold', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# (1, 0): Velocity - Model vs Right Eye
axes[1, 0].plot(time*1000, velocity, 'orange', linewidth=2.5, label='Model')
axes[1, 0].plot(real_times*1000, real_right_vel, 'r--', linewidth=2, alpha=0.8, label='Right Eye (Real)')
axes[1, 0].set_ylabel('Velocity [°/s]', fontweight='bold')
axes[1, 0].set_xlabel('Time [ms]', fontweight='bold')
axes[1, 0].set_title('Velocity: Model vs Real Data (Right Eye)', fontweight='bold', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# (1, 1): Acceleration - Model vs Right Eye
axes[1, 1].plot(time*1000, acceleration, 'green', linewidth=2.5, label='Model')
axes[1, 1].plot(real_times*1000, real_right_acc, 'r--', linewidth=2, alpha=0.8, label='Right Eye (Real)')
axes[1, 1].set_ylabel('Acceleration [°/s²]', fontweight='bold')
axes[1, 1].set_xlabel('Time [ms]', fontweight='bold')
axes[1, 1].set_title('Acceleration: Model vs Real Data (Right Eye)', fontweight='bold', fontsize=12)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('model_vs_real_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: model_vs_real_comparison.png")
plt.show()
