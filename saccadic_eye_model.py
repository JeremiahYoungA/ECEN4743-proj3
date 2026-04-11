import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class SaccadicEyeModel:
    """
    Bahill et al. (1976) nonlinear saccadic eye movement model.
    
    States: eyeball position/velocity, muscle displacements, muscle activation forces.
    Uses Hill force-velocity relationship for nonlinear muscle mechanics.
    """
    
    def __init__(self, J_p=2.3e-7, K_se=20.0, K_lt=100.0, K_p=120.0, B_p=0.1, 
                 T_ag=0.010, T_ant=0.015, r=0.011):
        """Initialize model with physical parameters."""
        self.J_p = J_p          # Eyeball inertia [kg·m²]
        self.K_se = K_se        # Series elastic stiffness [N/rad]
        self.K_lt = K_lt        # Length-tension elastic stiffness [N/rad]
        self.K_p = K_p          # Passive eyeball elasticity [N/rad]
        self.B_p = B_p          # Passive eyeball viscosity [N·s/rad]
        self.T_ag = T_ag        # Agonist muscle time constant [s]
        self.T_ant = T_ant      # Antagonist muscle time constant [s]
        self.r = r              # Muscle moment arm [m]
        self.v_max = 1000.0     # Max muscle shortening velocity [mm/s]
        self.B_max = 5.0        # Max muscle viscosity [N·s/rad]
        
        print(f"Nonlinear Saccadic Model (Bahill et al., 1976)")
        print(f"  Eyeball inertia J_p = {self.J_p:.2e} kg·m²")
        print(f"  Series elasticity K_se = {self.K_se:.1f} N/rad")
        print(f"  Length-tension K_lt = {self.K_lt:.1f} N/rad")
        print(f"  Passive elasticity K_p = {self.K_p:.1f} N/rad")
        print(f"  Passive viscosity B_p = {self.B_p:.3f} N·s/rad")
    
    def get_nonlinear_viscosity(self, tension, velocity):
        """
        Calculate nonlinear viscosity using Hill's force-velocity relationship.
        
        Governing equation (Hill hyperbola):
        B(v) = B_max * (v_max + |v|) / (v_max * (1 + |v|/v_max))
        """
        v_normalized = velocity / self.v_max
        
        if abs(v_normalized) < 0.01:
            B_nonlinear = self.B_max
        else:
            numerator = self.v_max + abs(velocity)
            denominator = self.v_max * (1.0 + abs(v_normalized))
            B_nonlinear = self.B_max * numerator / denominator
        
        return max(0.1, B_nonlinear)
    
    def generate_neural_inputs(self, t, saccade_onset=0.01, saccade_magnitude=10, peak_velocity=500, use_ramp=False):
        """Generate neural drive signals E_ag(t) and E_ant(t) (agonist/antagonist commands)."""
        E_ag = np.zeros_like(t)
        E_ant = np.zeros_like(t)
        
        idx_start = np.argmin(np.abs(t - saccade_onset))
        
        if use_ramp:
            ramp_duration = 0.020
            idx_ramp_end = np.argmin(np.abs(t - (saccade_onset + ramp_duration)))
            target_command = peak_velocity
            
            E_ag[idx_start:idx_ramp_end] = np.linspace(0, target_command, idx_ramp_end - idx_start)
            E_ag[idx_ramp_end:] = target_command
            E_ant[:idx_start] = target_command * 0.1
            E_ant[idx_start:idx_ramp_end] = np.linspace(target_command * 0.1, 0, idx_ramp_end - idx_start)
        else:
            pulse_duration = 0.020
            idx_pulse_end = np.argmin(np.abs(t - (saccade_onset + pulse_duration)))
            pulse_magnitude = peak_velocity
            step_magnitude = 0.3 * pulse_magnitude
            
            E_ag[idx_start:idx_pulse_end] = pulse_magnitude
            E_ag[idx_pulse_end:] = step_magnitude
            E_ant[:idx_start] = step_magnitude * 0.1
        
        return E_ag, E_ant

    def simulate_saccade(self, t, saccade_onset=0.01, saccade_magnitude=10, initial_position=0, use_ramp=False, peak_velocity=500):
        """
        Simulate saccadic eye movement using coupled ODEs with 6 state variables.
        
        Governing equations (6-state Bahill model):
        dx1/dt = v1
        dv1/dt = (torque_ag - torque_ant - B_p*v1 - K_p*x1) / J_p
        dx2/dt = (F_ag - K_lt*x2 - K_se*(x2-x1)) / B_ag
        dx3/dt = (F_ant - K_lt*x3 - K_se*(x3-x1)) / B_ant
        dF_ag/dt = (E_ag - F_ag) / T_ag
        dF_ant/dt = (E_ant - F_ant) / T_ant
        
        States: [x1=position, v1=velocity, x2=agonist, x3=antagonist, F_ag=force_ag, F_ant=force_ant]
        """
        print("\n=== NONLINEAR SACCADIC MODEL (Bahill et al., 1976) ===")
        
        input_type = "LINEAR RAMP" if use_ramp else "PULSE-STEP"
        print(f"1. Generating neural {input_type} commands...")
        E_ag_array, E_ant_array = self.generate_neural_inputs(t, saccade_onset, saccade_magnitude, peak_velocity, use_ramp=use_ramp)
        
        print("2. Setting up coupled ODEs (6 states: eyeball + muscles)...")
        
        def nonlinear_ode(state, time_val):
            """Bahill model: eyeball mechanics + muscle activation dynamics."""
            x1, v1, x2, x3, F_ag, F_ant = state
            
            E_ag = np.interp(time_val, t, E_ag_array)
            E_ant = np.interp(time_val, t, E_ant_array)
            
            v_ag = (self.K_se * (x2 - x1)) / (self.B_max * 1.0) if self.B_max != 0 else 0
            B_ag = self.get_nonlinear_viscosity(F_ag, v_ag)
            
            v_ant = (self.K_se * (x3 - x1)) / (self.B_max * 1.0) if self.B_max != 0 else 0
            B_ant = self.get_nonlinear_viscosity(F_ant, v_ant)
            
            dx1_dt = v1
            
            torque_ag = self.K_se * (x2 - x1)
            torque_ant = self.K_se * (x1 - x3)
            passive_force = self.B_p * v1 + self.K_p * x1
            
            dv1_dt = (torque_ag - torque_ant - passive_force) / self.J_p
            
            net_ag = F_ag - self.K_lt * x2 - self.K_se * (x2 - x1)
            dx2_dt = net_ag / (B_ag + 0.01)
            
            net_ant = F_ant - self.K_lt * x3 - self.K_se * (x3 - x1)
            dx3_dt = net_ant / (B_ant + 0.01)
            
            dF_ag_dt = (E_ag - F_ag) / self.T_ag
            dF_ant_dt = (E_ant - F_ant) / self.T_ant
            
            return [dx1_dt, dv1_dt, dx2_dt, dx3_dt, dF_ag_dt, dF_ant_dt]
        
        print("3. Integrating nonlinear state-variable ODEs...")
        
        initial_state = [initial_position, 0.0, 0.0, 0.0, 0.0, 0.0]
        state_trajectory = odeint(nonlinear_ode, initial_state, t)
        
        eye_position = state_trajectory[:, 0]
        eye_velocity = state_trajectory[:, 1]
        eye_acceleration = np.gradient(eye_velocity, t)
        
        print(f"   Peak velocity: {np.max(np.abs(eye_velocity)):.2f} °/s")
        print(f"   Final position: {eye_position[-1]:.4f} °")
        print(f"   Max overshoot: {(np.max(np.abs(eye_position)) - np.abs(eye_position[-1])):.4f} °")
        print("   Integration complete\n")
        
        return t, eye_position, eye_velocity, eye_acceleration, E_ag_array, E_ant_array


# ============================================================================
# FUNCTION DOCUMENTATION
# ============================================================================
# 
# SaccadicEyeModel.__init__()
#   Initialize model parameters. Physical properties (inertia, elasticity,
#   viscosity) and muscle time constants define model behavior.
#
# get_nonlinear_viscosity(tension, velocity)
#   Calculate muscle viscosity using Hill force-velocity relationship.
#   Viscosity is nonlinear and depends on muscle activation and velocity.
#
# generate_neural_inputs(t, saccade_onset, saccade_magnitude, peak_velocity, use_ramp)
#   Generate neural commands (E_ag, E_ant) as functions of time.
#   Two modes: PULSE-STEP (ballistic saccade) or LINEAR RAMP.
#
# simulate_saccade(t, saccade_onset, saccade_magnitude, initial_position, use_ramp, peak_velocity)
#   Full forward simulation of saccadic eye movement. Integrates 6-state ODE:
#   [eyeball position, velocity, agonist displacement, antagonist displacement, 
#    agonist force, antagonist force]. Returns position, velocity, acceleration,
#   and neural inputs over time.
#
# ============================================================================
