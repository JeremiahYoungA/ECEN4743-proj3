#!/usr/bin/env python3
"""
Test individual steps of the simulation
"""
import numpy as np
from scipy import signal
from saccadic_eye_model import SaccadicEyeModel

# Time array
dt = 0.001
t = np.arange(0, 0.2, dt)

# Initialize model
model = SaccadicEyeModel(J=2.3e-7, B=0.1, K=120, K_se=20, T_ag=0.010, T_ant=0.015, r=11.0)

# Generate neural inputs
F_ag, F_ant = model.generate_neural_inputs(t, saccade_onset=0.020, saccade_magnitude=15, peak_velocity=500)
print(f"Neural inputs generated:")
print(f"  Agonist force range: [{np.min(F_ag):.3f}, {np.max(F_ag):.3f}]")
print(f"  Antagonist force range: [{np.min(F_ant):.3f}, {np.max(F_ant):.3f}]")

# Apply muscle filter
tau_ag, tau_ant = model.apply_muscle_viscoelasticity(t, F_ag, F_ant)
print(f"\nTorque after muscle filter:")
print(f"  Agonist torque range: [{np.min(tau_ag):.3e}, {np.max(tau_ag):.3e}]")
print(f"  Antagonist torque range: [{np.min(tau_ant):.3e}, {np.max(tau_ant):.3e}]")

# Net torque
net_torque = tau_ag - tau_ant
print(f"\nNet torque:")
print(f"  Net torque range: [{np.min(net_torque):.3e}, {np.max(net_torque):.3e}]")
print(f"  Peak torque at index: {np.argmax(np.abs(net_torque))} ({t[np.argmax(np.abs(net_torque))]*1000:.1f} ms)")

# Now test transfer function with scipy
print(f"\n--- Testing Transfer Function ---")
print(f"Numerator (gain): {model.gain}")
print(f"Denominator: [1, {model.C3:.2e}, {model.C2:.2e}, {model.C1:.2e}, {model.C0:.2e}]")

# Create system and test step response
num = [model.gain]
den = [1, model.C3, model.C2, model.C1, model.C0]
sys = signal.TransferFunction(num, den)

# Check if we can even simulate
print(f"\nTesting with small test input (1 N·m step)...")
test_input = np.ones_like(t)
try:
    t_out, y_out, x_out = signal.lsim(sys, U=test_input, T=t)
    print(f"  Output range: [{np.min(y_out):.3e}, {np.max(y_out):.3e}]")
    print(f"  Peak output value: {np.max(np.abs(y_out)):.3e} at t={t_out[np.argmax(np.abs(y_out))]*1000:.1f} ms")
except Exception as e:
    print(f"  ERROR: {e}")

print(f"\nTesting with actual saccade torque...")
try:
    t_out, y_out, x_out = signal.lsim(sys, U=net_torque, T=t)
    print(f"  Output range: [{np.min(y_out):.3e}, {np.max(y_out):.3e}]")
    print(f"  Peak output: {np.max(np.abs(y_out)):.3e}")
except Exception as e:
    print(f"  ERROR: {e}")

# Try with scaled torque
print(f"\nTesting with 1000x scaled torque...")
try:
    t_out, y_out, x_out = signal.lsim(sys, U=net_torque*1000, T=t)
    print(f"  Output range: [{np.min(y_out):.3e}, {np.max(y_out):.3e}]")
    print(f"  Peak output: {np.max(np.abs(y_out)):.3e}")
except Exception as e:
    print(f"  ERROR: {e}")

# Try alternative: use ctrl library if available
print(f"\nTrying to use scipy's TransferFunction.to_ss() to check system...")
try:
    sys_ss = sys.to_ss()
    print(f"  A matrix shape: {sys_ss.A.shape}")
    print(f"  Eigenvalues: {np.linalg.eigvals(sys_ss.A)}")
except Exception as e:
    print(f"  ERROR: {e}")
