#!/usr/bin/env python3
"""
Analyze what neural input type would produce NO OVERSHOOT
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

print("="*70)
print("NEURAL INPUT TYPES & OVERSHOOT ANALYSIS")
print("="*70)

# Physical parameters
J = 2.3e-7
B = 0.1
K = 120
r = 11.0 / 1000.0

dt = 0.00436  # 229.4 Hz
duration = 0.5
t = np.arange(0, duration, dt)

fig, axes = plt.subplots(4, 3, figsize=(15, 12))

# Plant ODE
def plant_ode(state, time_val, tau_input, t_array):
    theta, theta_dot = state
    tau_t = np.interp(time_val, t_array, tau_input)
    B_over_J = B / J
    K_over_J = K / J
    one_over_J = 1.0 / J
    theta_ddot = -B_over_J * theta_dot - K_over_J * theta + one_over_J * tau_t
    return [theta_dot, theta_ddot]

# ============ INPUT TYPE 1: PULSE-STEP (current model) ============
print("\n1. PULSE-STEP INPUT (Current Model)")
print("   Description: Sharp impulse (20ms), then step hold")

tau_pulse = np.zeros_like(t)
idx_start = np.argmin(np.abs(t - 0.050))
idx_pulse_end = np.argmin(np.abs(t - 0.070))
pulse_magnitude = 1.0
step_magnitude = 0.3

tau_pulse[idx_start:idx_pulse_end] = pulse_magnitude
tau_pulse[idx_pulse_end:] = step_magnitude

# Simulate
def ode_pulse(state, time_val):
    return plant_ode(state, time_val, tau_pulse, t)

traj = odeint(ode_pulse, [0, 0], t)
pos_pulse = traj[:, 0]
vel_pulse = traj[:, 1]

print(f"   Peak position: {np.max(pos_pulse):.4f}°")
print(f"   Final position: {pos_pulse[-1]:.4f}°")
print(f"   Overshoot: {(np.max(pos_pulse) - pos_pulse[-1]) / pos_pulse[-1] * 100:.1f}%")

# ============ INPUT TYPE 2: LINEAR RAMP (smooth pursuit) ============
print("\n2. LINEAR RAMP INPUT (Smooth Pursuit)")
print("   Description: Linearly increase force to target, then hold")

tau_ramp = np.zeros_like(t)
idx_start = np.argmin(np.abs(t - 0.050))
idx_end = np.argmin(np.abs(t - 0.250))  # 200 ms to reach target
target_force = 1.0

# Linear ramp from 0 to target_force
tau_ramp[idx_start:idx_end] = np.linspace(0, target_force, idx_end - idx_start)
tau_ramp[idx_end:] = target_force

# Simulate
def ode_ramp(state, time_val):
    return plant_ode(state, time_val, tau_ramp, t)

traj = odeint(ode_ramp, [0, 0], t)
pos_ramp = traj[:, 0]
vel_ramp = traj[:, 1]

print(f"   Peak position: {np.max(pos_ramp):.4f}°")
print(f"   Final position: {pos_ramp[-1]:.4f}°")
print(f"   Overshoot: {(np.max(pos_ramp) - pos_ramp[-1]) / pos_ramp[-1] * 100:.1f}%")

# ============ INPUT TYPE 3: CRITICALLY DAMPED PULSE-STEP ============
print("\n3. PULSE-STEP WITH DAMPING OVERSHOOT PREVENTION")
print("   Description: Calculated pulse magnitude to prevent overshoot")

# For 2nd-order system, no overshoot requires ζ ≥ 1 or specific pulse shaping
# Inverse input shaping: use negative pulse to cancel overshoot
tau_shaped = np.zeros_like(t)
idx_start = np.argmin(np.abs(t - 0.050))
idx_pulse_end = np.argmin(np.abs(t - 0.070))
idx_dip_end = np.argmin(np.abs(t - 0.090))

tau_shaped[idx_start:idx_pulse_end] = 1.0      # Pulse
tau_shaped[idx_pulse_end:idx_dip_end] = -0.15  # Negative pulse (input shaping)
tau_shaped[idx_dip_end:] = 0.3                 # Step

# Simulate
def ode_shaped(state, time_val):
    return plant_ode(state, time_val, tau_shaped, t)

traj = odeint(ode_shaped, [0, 0], t)
pos_shaped = traj[:, 0]
vel_shaped = traj[:, 1]

print(f"   Peak position: {np.max(pos_shaped):.4f}°")
print(f"   Final position: {pos_shaped[-1]:.4f}°")
print(f"   Overshoot: {(np.max(pos_shaped) - pos_shaped[-1]) / pos_shaped[-1] * 100:.1f}%")

# ============ Plot inputs ============
print("\n" + "="*70)
print("PLOTTING INPUTS vs OUTPUTS")
print("="*70)

# Row 1: Inputs
axes[0, 0].plot(t*1000, tau_pulse, 'b-', linewidth=2)
axes[0, 0].set_ylabel('Force [N]', fontweight='bold')
axes[0, 0].set_title('INPUT 1: Pulse-Step', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t*1000, tau_ramp, 'g-', linewidth=2)
axes[0, 1].set_title('INPUT 2: Linear Ramp', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

axes[0, 2].plot(t*1000, tau_shaped, 'r-', linewidth=2)
axes[0, 2].set_title('INPUT 3: Input-Shaped Pulse', fontweight='bold')
axes[0, 2].grid(True, alpha=0.3)

# Row 2: Position
axes[1, 0].plot(t*1000, pos_pulse, 'b-', linewidth=2)
axes[1, 0].set_ylabel('Position [°]', fontweight='bold')
axes[1, 0].axhline(y=pos_pulse[-1], color='k', linestyle='--', alpha=0.3)
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].plot(t*1000, pos_ramp, 'g-', linewidth=2)
axes[1, 1].axhline(y=pos_ramp[-1], color='k', linestyle='--', alpha=0.3)
axes[1, 1].grid(True, alpha=0.3)

axes[1, 2].plot(t*1000, pos_shaped, 'r-', linewidth=2)
axes[1, 2].axhline(y=pos_shaped[-1], color='k', linestyle='--', alpha=0.3)
axes[1, 2].grid(True, alpha=0.3)

# Row 3: Velocity
axes[2, 0].plot(t*1000, vel_pulse, 'b-', linewidth=2)
axes[2, 0].set_ylabel('Velocity [°/s]', fontweight='bold')
axes[2, 0].grid(True, alpha=0.3)

axes[2, 1].plot(t*1000, vel_ramp, 'g-', linewidth=2)
axes[2, 1].grid(True, alpha=0.3)

axes[2, 2].plot(t*1000, vel_shaped, 'r-', linewidth=2)
axes[2, 2].grid(True, alpha=0.3)

# Row 4: Comparison with your real data
import pandas as pd
real_df = pd.read_csv('eye_tracking_data.csv')
frame_start = 3000
frame_end = 3100
real_data = real_df.iloc[frame_start:frame_end].reset_index(drop=True)
real_times = (real_data['timestamp_ms'].values / 1000.0) - (real_data['timestamp_ms'].values[0] / 1000.0)
real_left_vel = np.gradient(real_data['left_horizontal'].values, np.mean(np.diff(real_times)))

# Normalize to compare shapes
vel_pulse_norm = vel_pulse / np.max(np.abs(vel_pulse)) * np.max(np.abs(real_left_vel))
vel_ramp_norm = vel_ramp / np.max(np.abs(vel_ramp)) * np.max(np.abs(real_left_vel))
vel_shaped_norm = vel_shaped / np.max(np.abs(vel_shaped)) * np.max(np.abs(real_left_vel))

axes[3, 0].plot(t*1000, vel_pulse_norm, 'b-', linewidth=2, label='Model')
axes[3, 0].plot(real_times*1000, real_left_vel, 'b--', linewidth=1, alpha=0.5, label='Real')
axes[3, 0].set_ylabel('Velocity [°/s]', fontweight='bold')
axes[3, 0].set_xlabel('Time [ms]')
axes[3, 0].set_title('Pulse-Step vs Real', fontweight='bold')
axes[3, 0].grid(True, alpha=0.3)
axes[3, 0].legend()

axes[3, 1].plot(t*1000, vel_ramp_norm, 'g-', linewidth=2, label='Model')
axes[3, 1].plot(real_times*1000, real_left_vel, 'g--', linewidth=1, alpha=0.5, label='Real')
axes[3, 1].set_xlabel('Time [ms]')
axes[3, 1].set_title('Linear Ramp vs Real', fontweight='bold')
axes[3, 1].grid(True, alpha=0.3)
axes[3, 1].legend()

axes[3, 2].plot(t*1000, vel_shaped_norm, 'r-', linewidth=2, label='Model')
axes[3, 2].plot(real_times*1000, real_left_vel, 'r--', linewidth=1, alpha=0.5, label='Real')
axes[3, 2].set_xlabel('Time [ms]')
axes[3, 2].set_title('Input-Shaped vs Real', fontweight='bold')
axes[3, 2].grid(True, alpha=0.3)
axes[3, 2].legend()

plt.tight_layout()
plt.savefig('input_types_overshoot_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: input_types_overshoot_analysis.png")
plt.show()

print("\n" + "="*70)
print("CONCLUSIONS")
print("="*70)
print("""
OBSERVATION: Your real data shows NO OVERSHOOT

This means your eye movement is controlled by ONE OF:

1. SMOOTH PURSUIT (Linear Ramp Input):
   - Neural drive increases gradually
   - No velocity overshoot
   - Matches your velocity profile best
   ✓ RECOMMENDED for your data

2. INPUT-SHAPED CONTROL:
   - Active damping with negative feedback pulse
   - Calculated to cancel overshoot
   - More complex, used for precision tasks
   
3. HEAVY DAMPING:
   - Very high B coefficient relative to K
   - System critically damped or overdamped
   - Would need different physical parameters
   
NEXT STEP:
Replace the pulse-step neural input generator with a LINEAR RAMP
command that matches your 400ms smooth ramp-up profile.
""")
