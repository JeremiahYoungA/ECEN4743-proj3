#!/usr/bin/env python3
"""
Quick test of saccadic model without plotting to debug issues
"""
import numpy as np
from scipy import signal
from saccadic_eye_model import SaccadicEyeModel

print("="*60)
print("ROBINSON SACCADIC EYE MODEL - DEBUG TEST")
print("="*60)

# Time array
dt = 0.001
t = np.arange(0, 0.2, dt)
print(f"\nSimulation: {len(t)} samples over {t[-1]} seconds at {1/dt} Hz")

# Initialize model
J = 2.3e-7
B = 0.1
K = 120
K_se = 20

print(f"\nPhysical parameters:")
print(f"  J = {J:.2e}, B = {B}, K = {K}, K_se = {K_se}")

model = SaccadicEyeModel(J=J, B=B, K=K, K_se=K_se, T_ag=0.010, T_ant=0.015, r=11.0)

print(f"\nPlant model parameters:")
print(f"  J/K = {model.J_over_K:.2e}")
print(f"  B/K = {model.B_over_K:.2e}")
print(f"  DC gain (1/K) = {1/model.K:.4f} rad/Nm")

#Run simulation
print("\n" + "="*60)
print("Running simulation...")
print("="*60)

time, pos, vel, acc, torque = model.simulate_saccade(
    t,
    saccade_onset=0.020,
    saccade_magnitude=15,
    initial_position=0
)

print(f"\nResults:")
print(f"  Time array: {len(time)} samples")
print(f"  Position range: [{np.min(pos):.3f}, {np.max(pos):.3f}] degrees")
print(f"  Velocity range: [{np.min(vel):.1f}, {np.max(vel):.1f}] °/s")
print(f"  Acceleration range: [{np.min(acc):.1f}, {np.max(acc):.1f}] °/s²")
print(f"  Torque range: [{np.min(torque):.3e}, {np.max(torque):.3e}] N·m")

# Detailed analysis
saccade_start_idx = np.argmin(np.abs(time - 0.020))
peak_vel_idx = np.argmax(np.abs(vel))
print(f"\nSaccade analysis:")
print(f"  Peak velocity: {vel[peak_vel_idx]:.1f} °/s at t={time[peak_vel_idx]*1000:.1f} ms")
print(f"  Final position: {pos[-1]:.3f}°")
print(f"  Settling time (to <10°/s): {(np.where(np.abs(vel[saccade_start_idx:]) < 10)[0][0] + saccade_start_idx)*dt*1000:.1f} ms" if np.any(np.abs(vel[saccade_start_idx:]) < 10) else "  Settling time: N/A")

print("\n✓ Test complete")
