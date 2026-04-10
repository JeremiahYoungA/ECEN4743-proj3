#!/usr/bin/env python3
"""
Tuned saccadic model to match actual eye tracking data

PARAMETER TUNING GUIDE:
======================
If model output doesn't match real data, adjust these parameters:

Physical Model Parameters (lines 50-58):
  - J_p: Eyeball inertia
    • Increase → slower response, more overshoot
    • Decrease → faster response, less overshoot
  - B_p: Passive viscosity  
    • Increase → more sluggish, heavily damped
    • Decrease → snappier, more oscillatory
  - K_p, K_se, K_lt: Elasticity/stiffness parameters
    • Increase → stiffer, requires more force to move
    • Decrease → more compliant, moves easier
    
Neural Input Parameters (lines 76-79):
  - peak_neural_input: Strength of neural command
    • Increase → larger position and velocity
    • Decrease → smaller movements
  - saccade_magnitude: Target amplitude (°)
    • Set to match real data amplitude
    
Timing Parameters (line 76):
  - ramp_delay: When movement starts (seconds)
    • Adjust to match when real eye starts moving
    
If position is too small: Increase peak_neural_input or K values
If velocity is too small: Increase peak_neural_input or decrease B_p
If response too sluggish: Decrease J_p or B_p
If response too snappy/overshoots: Increase J_p or B_p

TYPICAL PARAMETER RANGES (Bahill et al. 1976 - CORRECTED):
==============================================
╔═════════════════════════════════════════════════════════════════╗
║ Parameter    │ Units        │ Typical Range    │ Current Value  ║
╠═════════════════════════════════════════════════════════════════╣
║ J_p          │ kg·m²        │ 1.5-3.5 × 10⁻⁷   │ 2.3 × 10⁻⁷     ║
║ B_p          │ N·s/rad      │ 0.001 - 0.01     │ 0.002          ║
║ K_p          │ N·m/rad      │ 0.5 - 1.5        │ 1.0            ║
║ K_se         │ N/m (trans)  │ 10 - 30          │ 20             ║
║ K_lt         │ N/m (trans)  │ 50 - 150         │ 100            ║
║ T_ag         │ s            │ 0.003 - 0.010    │ 0.003          ║
║ T_ant        │ s            │ 0.005 - 0.015    │ 0.005          ║
║ r            │ m            │ 0.008 - 0.015    │ 0.011          ║
╚═════════════════════════════════════════════════════════════════╝
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

# Load real data FIRST to determine exact number of frames
export_dir = 'export_20260409_201829_frames_3000-3200'
if os.path.exists(export_dir):
    real_df = pd.read_csv(os.path.join(export_dir, 'time_domain_data.csv'))
    num_frames = len(real_df)
    print(f"📊 Found {num_frames} frames in export folder")
    use_export = True
else:
    # Load main CSV to determine frame count for the fallback region
    real_df = pd.read_csv('eye_tracking_data.csv')
    frame_start = 3000
    frame_end = 3000 + min(200, len(real_df) - 3000)
    num_frames = frame_end - frame_start
    print(f"⚠ Export folder not found, using {num_frames} frames from main CSV")
    use_export = False

frame_rate = 229.4  # From real data
dt = 1.0 / frame_rate  # 4.36 ms per frame
duration = (num_frames - 1) / frame_rate  # Duration between first and last frame

# Create time array with EXACT same number of samples as real data
t = np.linspace(0, duration, num_frames)

print(f"\n⏱️  Time array created with {len(t)} samples (matching {num_frames} data frames)")

print(f"\nSimulation parameters:")
print(f"  Duration: {duration*1000:.0f} ms ({len(t)} samples to match all real data)")
print(f"  Frame rate: {frame_rate:.1f} Hz")
print(f"  Samples: {len(t)}")
print(f"  dt: {dt*1000:.2f} ms")

# Initialize model with baseline physical parameters
# REVERT to rotation-only formulation for numerical stability
# --- TUNED FOR SACCADIC MAIN SEQUENCE VELOCITY ---
J_p = 2.2e-7        # Standard eye inertia
B_p = 0.008         # Damping
K_p = 0.02           # Weaker orbital stiffness for larger range (was 0.8)
K_se = 2000.0       # Stiffer series elasticity for better velocity
K_lt = 1200.0       # Realistic muscle length-tension

model = SaccadicEyeModel(
    J_p=J_p, B_p=B_p, K_p=K_p, K_se=K_se, K_lt=K_lt,
    T_ag=0.002, T_ant=0.005, r=0.011
)

print(f"\n🔧 PHYSICAL PARAMETERS (Bahill et al. 1976 - Dimensionally Corrected):")
print(f"  J_p (inertia):           {J_p:.2e} kg·m²")
print(f"  B_p (viscosity):         {B_p:.4f} N·s/rad (increased to prevent overshoot)")
print(f"  K_p (passive stiffness): {K_p:.1f} N·m/rad (realistic orbital restraint)")
print(f"  K_se (series elasticity):{K_se:.1f} N/m (translational, will convert to {K_se * (0.011**2):.6f} N·m/rad)")
print(f"  K_lt (length-tension):   {K_lt:.1f} N/m (translational, will convert to {K_lt * (0.011**2):.6f} N·m/rad)")

print(f"\n--- Running TUNED simulation ---")

# ===== KEY TUNING PARAMETERS FOR SACCADE =====
ramp_delay = 0.33          # Delay before saccade starts [seconds]
peak_neural_input = 1.0       # Scaling factor - actual torques calculated from K_p * target
saccade_magnitude = 11.7      # Match real data amplitude (TUNABLE)
ramp_duration_ms = 20           # Ramp rise time [ms] - edit in saccadic_eye_model.py

print(f"Tuning parameters for SACCADE:")
print(f"  Saccade onset: {ramp_delay*1000:.1f} ms")
print(f"  Target amplitude: {saccade_magnitude}° (matches real data)")
print(f"  Series elasticity: {K_se} N/m (stiff tendon for rapid velocity)")
print(f"  Passive stiffness: {K_p} N·m/rad (holds eye at target)")
print(f"  Muscle time constants: T_ag=2ms, T_ant=5ms (fast activation)")
print(f"  Neural step balances K_p * target_position for steady-state hold")

time, position, velocity, acceleration, E_ag, E_ant = model.simulate_saccade(
    t,
    saccade_onset=ramp_delay,      # Delay before saccade starts
    saccade_magnitude=saccade_magnitude, # Match real data amplitude
    initial_position=0,             # Model starts at rest at 0
    use_ramp=True,                 # PULSE-STEP for ballistic saccade
    peak_velocity=peak_neural_input # Neural command for dynamics
)

# Trim outputs to match input t length (in case simulate_saccade generates extra samples)
time = time[:len(t)]
position = position[:len(t)]
velocity = velocity[:len(t)]
acceleration = acceleration[:len(t)]
E_ag = E_ag[:len(t)]
E_ant = E_ant[:len(t)]

print(f"✓ Model outputs: {len(time)} samples (time), {len(position)} position, {len(velocity)} velocity, {len(acceleration)} acceleration")

# The model needs much larger neural input to match real velocities
# Create simpler focused plot: Neural Input → Model Outputs vs Real Data
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Real data already loaded above, now extract left eye data for analysis
if os.path.exists(export_dir):
    # Use ALL frames from the export directory
    frame_start = 0
    frame_end = len(real_df)
    real_data = real_df.iloc[frame_start:frame_end].reset_index(drop=True)
    
    print(f"\n📊 Using ALL {len(real_data)} frames from export folder for comparison")
    
    # Create synthetic time array based on frame rate
    real_times = np.arange(len(real_data)) / frame_rate
    
    # USE FILTERED POSITION DATA FOR COMPARISON (LEFT EYE)
    real_left_pos = real_data['Left_Position_Filtered'].values
    
    # USE pre-computed velocity from CSV - BUT CONVERT from °/frame to °/s
    real_left_vel = real_data['Left_Velocity'].values * frame_rate  # Convert °/frame → °/s
    real_left_acc = real_data['Left_Acceleration'].values * (frame_rate ** 2)  # Convert °/frame² → °/s²
    
    # Calculate offset: shift real data to start at 0 (same as model)
    real_left_offset = real_left_pos - real_left_pos[0]
    
    # CALCULATE SCALE FACTORS FROM ACTUAL REAL DATA
    real_peak_velocity = np.max(np.abs(real_left_vel))
    real_peak_position = np.max(real_left_offset)
    velocity_scale = real_peak_velocity / np.max(np.abs(velocity)) if np.max(np.abs(velocity)) > 0 else 0
    position_scale = real_peak_position / np.max(position) if np.max(position) > 0 else 0
    
else:
    # Fall back to main CSV with raw data (LEFT EYE)
    real_data = real_df.iloc[frame_start:frame_end].reset_index(drop=True)
    
    real_times = (real_data['timestamp_ms'].values / 1000.0) - (real_data['timestamp_ms'].values[0] / 1000.0)
    real_left_pos = real_data['left_horizontal'].values
    real_left_vel = np.gradient(real_left_pos, np.mean(np.diff(real_times)))
    real_left_acc = np.gradient(real_left_vel, np.mean(np.diff(real_times)))
    real_left_offset = real_left_pos - real_left_pos[0]
    
    print(f"📊 Using {len(real_data)} frames from main CSV")
    
    # Calculate scale factors from actual real data
    real_peak_velocity = np.max(np.abs(real_left_vel))
    real_peak_position = np.max(real_left_offset)
    velocity_scale = real_peak_velocity / np.max(np.abs(velocity)) if np.max(np.abs(velocity)) > 0 else 0
    position_scale = real_peak_position / np.max(position) if np.max(position) > 0 else 0

# Print actual data analysis
print(f"\n--- SCALE ANALYSIS (from ACTUAL real eye data) ---")
print(f"Model outputs:")
print(f"  Peak velocity: {np.max(np.abs(velocity)):.2f} °/s")
print(f"  Final position: {np.max(position):.4f}°")
print(f"  Peak acceleration: {np.max(np.abs(acceleration)):.1f}°/s²")
print(f"\nActual real data (Left Eye):")
print(f"  Peak velocity: {real_peak_velocity:.6f} °/s")
print(f"  Peak position: {real_peak_position:.6f}°")
print(f"\nScale factors (to match real data):")
print(f"  Velocity scale: {velocity_scale:.6f}x")
print(f"  Position scale: {position_scale:.6f}x")

# ===== L2 ERROR CALCULATION =====
# Use ALL data - model and real data should now have same length
# Pad shorter array if needed (shouldn't happen with linspace approach)
if len(position) < len(real_left_offset):
    position = np.pad(position, (0, len(real_left_offset) - len(position)), mode='edge')
    velocity = np.pad(velocity, (0, len(real_left_offset) - len(velocity)), mode='edge')
    acceleration = np.pad(acceleration, (0, len(real_left_offset) - len(acceleration)), mode='edge')
elif len(position) > len(real_left_offset):
    position = position[:len(real_left_offset)]
    velocity = velocity[:len(real_left_offset)]
    acceleration = acceleration[:len(real_left_offset)]

# Use ALL aligned data
model_pos_aligned = position
real_pos_aligned = real_left_offset
model_vel_aligned = velocity
real_vel_aligned = real_left_vel
model_acc_aligned = acceleration
real_acc_aligned = real_left_acc
min_len = len(model_pos_aligned)

# Calculate L2 errors
l2_position = np.sqrt(np.mean((model_pos_aligned - real_pos_aligned) ** 2))
l2_velocity = np.sqrt(np.mean((model_vel_aligned - real_vel_aligned) ** 2))
l2_acceleration = np.sqrt(np.mean((model_acc_aligned - real_acc_aligned) ** 2))

# Calculate normalized L2 errors (as percentage of real data range)
pos_range = np.max(real_pos_aligned) - np.min(real_pos_aligned)
vel_range = np.max(real_vel_aligned) - np.min(real_vel_aligned)
acc_range = np.max(real_acc_aligned) - np.min(real_acc_aligned)

norm_l2_pos = (l2_position / pos_range * 100) if pos_range > 0 else 0
norm_l2_vel = (l2_velocity / vel_range * 100) if vel_range > 0 else 0
norm_l2_acc = (l2_acceleration / acc_range * 100) if acc_range > 0 else 0

print(f"\n--- L2 ERROR ANALYSIS (over {min_len} aligned samples) ---")
print(f"Absolute L2 Errors:")
print(f"  Position: {l2_position:.6f}° (RMS)")
print(f"  Velocity: {l2_velocity:.6f}°/s (RMS)")
print(f"  Acceleration: {l2_acceleration:.6f}°/s² (RMS)")
print(f"\nNormalized L2 Errors (% of real data range):")
print(f"  Position: {norm_l2_pos:.2f}%")
print(f"  Velocity: {norm_l2_vel:.2f}%")
print(f"  Acceleration: {norm_l2_acc:.2f}%")

# (0, 0): Neural Input Commands
axes[0, 0].plot(time*1000, E_ag, 'purple', linewidth=2.5, label='Agonist Command (E_ag)')
axes[0, 0].plot(time*1000, E_ant, 'brown', linewidth=2.5, label='Antagonist Command (E_ant)')
axes[0, 0].set_ylabel('Neural Command [a.u.]', fontweight='bold')
axes[0, 0].set_xlabel('Time [ms]', fontweight='bold')
axes[0, 0].set_title('Neural Input Commands to Muscles', fontweight='bold', fontsize=12)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend(loc='upper left')

# (0, 1): Position - Model vs Left Eye
axes[0, 1].plot(time*1000, position, 'b-', linewidth=2.5, label='Model')
axes[0, 1].plot(real_times*1000, real_left_offset, 'g--', linewidth=2, alpha=0.8, label='Left Eye (Real)')
axes[0, 1].set_ylabel('Position [°]', fontweight='bold')
axes[0, 1].set_xlabel('Time [ms]', fontweight='bold')
axes[0, 1].set_title('Position: Model vs Real Data (Left Eye)', fontweight='bold', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# (1, 0): Velocity - Model vs Left Eye
axes[1, 0].plot(time*1000, velocity, 'orange', linewidth=2.5, label='Model')
axes[1, 0].plot(real_times*1000, real_left_vel, 'g--', linewidth=2, alpha=0.8, label='Left Eye (Real)')
axes[1, 0].set_ylabel('Velocity [°/s]', fontweight='bold')
axes[1, 0].set_xlabel('Time [ms]', fontweight='bold')
axes[1, 0].set_title('Velocity: Model vs Real Data (Left Eye)', fontweight='bold', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# (1, 1): Acceleration - Model vs Left Eye
axes[1, 1].plot(time*1000, acceleration, 'orange', linewidth=2.5, label='Model')
axes[1, 1].plot(real_times*1000, real_left_acc, 'g--', linewidth=2, alpha=0.8, label='Left Eye (Real)')
axes[1, 1].set_ylabel('Acceleration [°/s²]', fontweight='bold')
axes[1, 1].set_xlabel('Time [ms]', fontweight='bold')
axes[1, 1].set_title('Acceleration: Model vs Real Data (Left Eye)', fontweight='bold', fontsize=12)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

plt.tight_layout()
plt.savefig('model_vs_real_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: model_vs_real_comparison.png")
plt.show()
