import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class SaccadicEyeModel:
    """
    Nonlinear Reciprocal Innervation Saccadic Eye Movement Model (Bahill et al., 1976)
    
    PHYSICAL STATE-VARIABLE APPROACH:
    Instead of abstract polynomials, this model tracks the actual physical elements:
    
    x1 = θ         : Eyeball angular position [rad]
    v1 = θ̇         : Eyeball angular velocity [rad/s]
    x2             : Agonist muscle displacement [mm]
    x3             : Antagonist muscle displacement [mm]
    
    The mechanical elements are:
    - J_p          : Eyeball inertia
    - B_p, K_p     : Passive eyeball viscosity & elasticity
    - K_se         : Series elastic stiffness (tendon)
    - K_lt         : Length-tension elastic element
    - B_ag, B_ant  : NONLINEAR muscle viscosities (Hill hyperbola)
    
    KEY ADVANTAGE: Avoids high-order polynomial evaluation → numerically stable!
    """
    
    def __init__(self, J_p=2.3e-7, K_se=20.0, K_lt=100.0, K_p=120.0, B_p=0.1, 
                 T_ag=0.010, T_ant=0.015, r=0.011):
        """
        Initialize the nonlinear saccadic model with physical parameters.
        
        Parameters:
        -----------
        J_p : float
            Inertia of the eyeball plant [kg·m²] (typical ~2.3e-7)
        K_se : float
            Series elastic stiffness [N/rad] (typical ~20)
        K_lt : float
            Length-tension elastic stiffness [N/rad] (typical ~100)
        K_p : float
            Passive eyeball elasticity [N/rad] (typical ~120)
        B_p : float
            Passive eyeball viscosity [N·s/rad] (typical ~0.1)
        T_ag : float
            Agonist muscle time constant [s] (typical ~10 ms)
        T_ant : float
            Antagonist muscle time constant [s] (typical ~15 ms)
        r : float
            Moment arm of muscles [m] (typical ~11 mm)
        """
        # Physical eyeball and passive tissue properties
        self.J_p = J_p          # Eyeball inertia
        self.K_se = K_se        # Series elasticity (tendon)
        self.K_lt = K_lt        # Length-tension elasticity
        self.K_p = K_p          # Passive elasticity
        self.B_p = B_p          # Passive viscosity
        
        # Neural time constants
        self.T_ag = T_ag        # Agonist activation time constant
        self.T_ant = T_ant      # Antagonist activation time constant
        self.r = r              # Moment arm
        
        # Hill force-velocity relationship parameters
        self.v_max = 1000.0     # Maximum shortening velocity [mm/s]
        self.B_max = 5.0        # Maximum muscle viscosity [N·s/rad]
        
        print(f"Nonlinear Saccadic Model (Bahill et al., 1976)")
        print(f"  Eyeball inertia J_p = {self.J_p:.2e} kg·m²")
        print(f"  Series elasticity K_se = {self.K_se:.1f} N/rad")
        print(f"  Length-tension K_lt = {self.K_lt:.1f} N/rad")
        print(f"  Passive elasticity K_p = {self.K_p:.1f} N/rad")
        print(f"  Passive viscosity B_p = {self.B_p:.3f} N·s/rad")
    
    def get_nonlinear_viscosity(self, tension, velocity):
        """
        Calculates the nonlinear viscosity using Hill's force-velocity relationship.
        
        HILL HYPERBOLA MODEL:
        The Hill equation describes how muscle force decreases as shortening 
        velocity increases. This creates a nonlinear, velocity-dependent viscosity.
        
        Mathematical Form:
        B(v) = B_max * (v_max + v_rel) / (v_max * (1 + |v_rel|/v_max))
        
        Where:
        - v_rel = velocity relative to muscle reference
        - v_max = maximum shortening velocity
        - B_max = viscosity at isometric contraction
        
        This avoids the old linear viscosity assumption and provides better match
        to real muscle mechanics!
        """
        # Normalize velocity relative to max
        v_normalized = velocity / self.v_max
        
        # Hill force-velocity relationship (nonlinear)
        if abs(v_normalized) < 0.01:
            # Near isometric (v ≈ 0), use maximum damping
            B_nonlinear = self.B_max
        else:
            # Hill hyperbola formula
            numerator = self.v_max + abs(velocity)
            denominator = self.v_max * (1.0 + abs(v_normalized))
            B_nonlinear = self.B_max * numerator / denominator
        
        # Ensure positive viscosity
        B_nonlinear = max(0.1, B_nonlinear)
        
        return B_nonlinear
    
    def generate_neural_inputs(self, t, saccade_onset=0.01, saccade_magnitude=10, peak_velocity=500, use_ramp=False):
        """
        Generates neural drive signals: E_ag(t) and E_ant(t) - the RAW NEURAL COMMANDS
        
        These are the high-level neural commands from the brain stem that get filtered
        through muscle activation dynamics (T_ag, T_ant time constants).
        
        Returns:
        --------
        E_ag, E_ant : Neural commands (input signal)
        """
        if use_ramp:
            # LINEAR RAMP MODE - but with STEEP rise (more like square wave)
            E_ag = np.zeros_like(t)
            E_ant = np.zeros_like(t)
            
            idx_start = np.argmin(np.abs(t - saccade_onset))
            ramp_duration = 0.020  # 20 ms ramp (very steep, nearly square wave)
            idx_ramp_end = np.argmin(np.abs(t - (saccade_onset + ramp_duration)))
            
            target_command = peak_velocity  # Raw neural command
            E_ag[idx_start:idx_ramp_end] = np.linspace(0, target_command, idx_ramp_end - idx_start)
            E_ag[idx_ramp_end:] = target_command
            
            E_ant[:idx_start] = target_command * 0.1
            E_ant[idx_start:idx_ramp_end] = np.linspace(target_command * 0.1, 0, idx_ramp_end - idx_start)
            E_ant[idx_ramp_end:] = 0
        else:
            # PULSE-STEP MODE - ballistic saccade
            E_ag = np.zeros_like(t)
            E_ant = np.zeros_like(t)
            
            idx_start = np.argmin(np.abs(t - saccade_onset))
            pulse_duration = 0.020
            idx_pulse_end = np.argmin(np.abs(t - (saccade_onset + pulse_duration)))
            
            pulse_magnitude = peak_velocity
            step_magnitude = 0.3 * pulse_magnitude
            
            E_ag[idx_start:idx_pulse_end] = pulse_magnitude
            E_ag[idx_pulse_end:] = step_magnitude
            
            E_ant[:idx_start] = step_magnitude * 0.1
            E_ant[idx_start:idx_pulse_end] = 0
            E_ant[idx_pulse_end:] = 0
        
        return E_ag, E_ant

    def simulate_saccade(self, t, saccade_onset=0.01, saccade_magnitude=10, initial_position=0, use_ramp=False, peak_velocity=500):
        """
        Full forward simulation using NONLINEAR state-variable dynamics.
        
        PHYSICAL STATE VARIABLES (Bahill et al., 1976):
        x1 = θ          : Eyeball angular position [rad]
        v1 = θ̇          : Eyeball angular velocity [rad/s]
        x2              : Agonist muscle displacement [mm]
        x3              : Antagonist muscle displacement [mm]
        x4, x5          : Muscle activation states (F_ag, F_ant)
        
        COUPLED 1ST-ORDER ODES:
        dx1/dt = v1
        dv1/dt = [K_se*(x2 - x1) - K_se*(x1 - x3) - B_p*v1 - K_p*x1] / J_p
        dx2/dt = [x4 - K_lt*x2 - K_se*(x2 - x1)] / B_ag
        dx3/dt = [x5 - K_lt*x3 - K_se*(x3 - x1)] / B_ant
        dx4/dt = (E_ag(t) - x4) / T_ag   [muscle activation filter]
        dx5/dt = (E_ant(t) - x5) / T_ant [muscle activation filter]
        """
        print("\n=== NONLINEAR SACCADIC MODEL (Bahill et al., 1976) ===")
        
        input_type = "LINEAR RAMP" if use_ramp else "PULSE-STEP"
        print(f"1. Generating neural {input_type} commands...")
        E_ag_array, E_ant_array = self.generate_neural_inputs(t, saccade_onset, saccade_magnitude, peak_velocity, use_ramp=use_ramp)
        
        print("2. Setting up coupled ODEs (6 states: mechanical + muscle activation)...")
        print(f"   States: [x1=θ, v1=θ̇, x2=agonist, x3=antagonist, x4=F_ag, x5=F_ant]")
        print(f"   ✓ Neural commands filtered through muscle activation dynamics!")
        
        def nonlinear_ode(state, time_val):
            """
            6-state ODE system: Bahill model with explicit muscle activation filtering.
            
            x1, v1   : Eyeball angle and velocity
            x2, x3   : Muscle displacements
            x4, x5   : Muscle active forces (filtered neural commands)
            """
            x1, v1, x2, x3, x4, x5 = state
            
            # Interpolate neural COMMANDS at current time (the input signal)
            E_ag = np.interp(time_val, t, E_ag_array)
            E_ant = np.interp(time_val, t, E_ant_array)
            
            # --- NONLINEAR VISCOSITIES (Hill force-velocity) ---
            v_ag = (self.K_se * (x2 - x1)) / (self.B_max * 1.0) if self.B_max != 0 else 0
            B_ag = self.get_nonlinear_viscosity(x4, v_ag)  # Use active force x4, not command
            
            v_ant = (self.K_se * (x3 - x1)) / (self.B_max * 1.0) if self.B_max != 0 else 0
            B_ant = self.get_nonlinear_viscosity(x5, v_ant)  # Use active force x5, not command
            
            # --- STATE DERIVATIVES ---
            dx1_dt = v1
            
            # Eyeball acceleration: net torque from muscles divided by inertia
            torque_ag = self.K_se * (x2 - x1)
            torque_ant = self.K_se * (x1 - x3)
            passive_force = self.B_p * v1 + self.K_p * x1
            
            net_torque = torque_ag - torque_ant - passive_force
            dv1_dt = net_torque / self.J_p
            
            # Muscle displacement dynamics: massless nodes, forces sum to zero
            force_ag = x4 - self.K_lt * x2 - self.K_se * (x2 - x1)
            dx2_dt = force_ag / (B_ag + 0.01)
            
            force_ant = x5 - self.K_lt * x3 - self.K_se * (x3 - x1)
            dx3_dt = force_ant / (B_ant + 0.01)
            
            # MUSCLE ACTIVATION FILTERING: first-order lag (key physiological feature!)
            # Neural command E_ag gets filtered through time constant T_ag to produce active force x4
            dx4_dt = (E_ag - x4) / self.T_ag
            dx5_dt = (E_ant - x5) / self.T_ant
            
            return [dx1_dt, dv1_dt, dx2_dt, dx3_dt, dx4_dt, dx5_dt]
        
        print("3. Integrating nonlinear state-variable ODEs...")
        
        # Initial state: [position, velocity, agonist, antagonist, F_ag, F_ant]
        initial_state = [initial_position, 0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Solve ODEs
        state_trajectory = odeint(nonlinear_ode, initial_state, t)
        
        # Extract eyeball position (x1) and velocity (v1)
        eye_position = state_trajectory[:, 0]
        eye_velocity = state_trajectory[:, 1]
        F_ag_actual = state_trajectory[:, 4]  # Actual muscle forces after filtering
        F_ant_actual = state_trajectory[:, 5]
        
        t_out = t
        
        # Compute acceleration from velocity (for analysis/plotting)
        eye_acceleration = np.gradient(eye_velocity, t_out)
        
        print(f"   Peak velocity: {np.max(np.abs(eye_velocity)):.2f} °/s")
        print(f"   Final position: {eye_position[-1]:.4f} °")
        print(f"   Max overshoot: {(np.max(np.abs(eye_position)) - np.abs(eye_position[-1])):.4f} °")
        print("   ✓ Integration complete\n")
        
        return t_out, eye_position, eye_velocity, eye_acceleration, E_ag_array, E_ant_array
