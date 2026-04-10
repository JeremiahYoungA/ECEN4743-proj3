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
    
    def __init__(self, J_p=2.3e-7, K_se=20.0, K_lt=100.0, K_p=1.0, B_p=0.002, 
                 T_ag=0.010, T_ant=0.015, r=0.011):
        """
        Initialize the nonlinear saccadic model with physical parameters.
        
        Parameters:
        -----------
        J_p : float
            Inertia of the eyeball plant [kg·m²] (typical ~2.3e-7)
        K_se : float
            Series elastic stiffness [N/m] TRANSLATIONAL (typical ~20)
        K_lt : float
            Length-tension elastic stiffness [N/m] TRANSLATIONAL (typical ~100)
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
        self.J_p = J_p          # Eyeball inertia [kg·m²]
        self.K_se = K_se        # Series elasticity [N/m] TRANSLATIONAL
        self.K_lt = K_lt        # Length-tension elasticity [N/m] TRANSLATIONAL
        self.K_p = K_p          # Passive eyeball elasticity [N/rad] ROTATIONAL
        self.B_p = B_p          # Passive eyeball viscosity [N·s/rad]
        
        # CONVERT TRANSLATIONAL TO ROTATIONAL using moment arm
        # K_rot = K_trans * r^2  (dimensional conversion)
        self.K_se_rot = K_se * (r ** 2)  # Series elasticity [N·m/rad] ROTATIONAL
        self.K_lt_rot = K_lt * (r ** 2)  # Length-tension [N·m/rad] ROTATIONAL
        
        # Neural time constants
        self.T_ag = T_ag        # Agonist activation time constant [s]
        self.T_ant = T_ant      # Antagonist activation time constant [s]
        self.r = r              # Moment arm [m] - converts N (linear) → N·m (torque)
        
        # Hill force-velocity relationship parameters - CONVERT TO ROTATIONAL
        self.v_max = 0.01       # Maximum shortening velocity [m/s] (for reference)
        self.v_max_rot = 0.5    # Maximum rotational velocity [rad/s]
        self.B_max = 50.0       # Maximum muscle viscosity [N·s/m] TRANSLATIONAL
        self.B_max_rot = self.B_max * (r ** 2)  # Converted to rotational [N·s·m/rad]
        
        print(f"Nonlinear Saccadic Model (Bahill et al., 1976) - DIMENSIONALLY CORRECTED")
        print(f"  Eyeball inertia J_p = {self.J_p:.2e} kg·m²")
        print(f"  Series elasticity K_se = {self.K_se:.1f} N/m (trans) → {self.K_se_rot:.6f} N·m/rad (rot)")
        print(f"  Length-tension K_lt = {self.K_lt:.1f} N/m (trans) → {self.K_lt_rot:.6f} N·m/rad (rot)")
        print(f"  Passive eyeball K_p = {self.K_p:.3f} N·m/rad (realistic: 0.5-1.5)")
        print(f"  Passive eyeball B_p = {self.B_p:.4f} N·s/rad (realistic: 0.001-0.01)")
        print(f"  Muscle viscosity B_max = {self.B_max:.1f} N·s/m (trans) → {self.B_max_rot:.6f} N·s·m/rad (rot)")
        print(f"  Moment arm r = {self.r*1000:.1f} mm")
    
    def get_nonlinear_viscosity(self, tension, velocity):
        """
        Calculates the nonlinear viscosity using Hill's force-velocity relationship.
        NOW USING ROTATIONAL UNITS to match the ODE system.
        
        HILL HYPERBOLA MODEL:
        B(v) = B_max_rot * (v_max_rot + |v|) / (v_max_rot * (1 + |v|/v_max_rot))
        """
        # Normalize velocity relative to rotational max
        v_normalized = velocity / self.v_max_rot
        
        # Hill force-velocity relationship (nonlinear)
        if abs(v_normalized) < 0.01:
            # Near isometric (v ≈ 0), use maximum damping
            B_nonlinear = self.B_max_rot
        else:
            # Hill hyperbola formula (using rotational units)
            numerator = self.v_max_rot + abs(velocity)
            denominator = self.v_max_rot * (1.0 + abs(v_normalized))
            B_nonlinear = self.B_max_rot * numerator / denominator
        
        # Ensure positive viscosity
        B_nonlinear = max(0.001, B_nonlinear)
        
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
            # PULSE-STEP MODE - ballistic saccade with BALANCED commands
            E_ag = np.zeros_like(t)
            E_ant = np.zeros_like(t)
            
            # Convert target position to radians for torque calculation
            target_rad = np.deg2rad(saccade_magnitude)
            
            # Step torque must balance passive stiffness at target position
            # At steady state: torque_ag - torque_ant = K_p * x1
            step_torque = self.K_p * target_rad  # This holds the eye at target
            
            # Pulse needs to be significantly higher to overcome inertia/damping
            # Increase this multiplier to reach higher peak velocities
            pulse_torque = step_torque * 40000 # 40x step for aggressive saccade
            
            idx_start = np.argmin(np.abs(t - saccade_onset))
            pulse_duration = 0.030  # 30ms pulse for sharp onset
            idx_pulse_end = np.argmin(np.abs(t - (saccade_onset + pulse_duration)))
            
            # Agonist: pulse then steady step
            E_ag[idx_start:idx_pulse_end] = pulse_torque
            E_ag[idx_pulse_end:] = step_torque
            
            # Antagonist: quiet during pulse, small holding force during step
            E_ant[:idx_start] = 0
            E_ant[idx_start:idx_pulse_end] = 0
            E_ant[idx_pulse_end:] = step_torque * 0.1  # Minimal antagonism to hold position
        
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
            6-state ODE system with CORRECTED UNITS and MUSCLE CONTRACTILE VELOCITY.
            
            x1, v1   : Eyeball angle [rad] and angular velocity [rad/s]
            x2, x3   : Series elastic element displacement [rad]
            x4, x5   : Muscle contractile element displacement [rad] (filtered target)
            
            FIXED ISSUES:
            1. Torque now uses K_rot = K_trans * r^2 for correct dimensions
            2. Hill viscosity uses muscle shortening velocity (x4-x2)/T, not eyeball velocity
            3. Muscle dynamics include both K_se_rot and K_lt_rot with proper dimensions
            """
            x1, v1, x2, x3, x4, x5 = state
            
            # Interpolate neural signals (target displacement commands in radians)
            E_ag = np.interp(time_val, t, E_ag_array)
            E_ant = np.interp(time_val, t, E_ant_array)
            
            # MUSCLE CONTRACTILE VELOCITY (shortening rate of muscle fibers)
            # This is the rate of compression of the series elastic element
            v_muscle_ag = (x4 - x2) / (self.T_ag if self.T_ag > 0 else 0.01)
            v_muscle_ant = (x5 - x3) / (self.T_ant if self.T_ant > 0 else 0.01)
            
            # Hill viscosity using MUSCLE VELOCITY (not eyeball velocity)
            # This correctly captures damping during the rapid pulse phase
            B_ag = self.get_nonlinear_viscosity(x4, v_muscle_ag)
            B_ant = self.get_nonlinear_viscosity(x5, v_muscle_ant)
            
            # --- CORRECTED ROTATIONAL SPRING TORQUES (using K_rot = K_trans * r^2) ---
            # Torque transmitted to eyeball is only from series elastic element
            torque_ag = self.K_se_rot * (x2 - x1)
            torque_ant = self.K_se_rot * (x3 - x1)
            
            # Passive eyeball resistance (orbital tissues)
            passive_torque = self.B_p * v1 + self.K_p * x1
            
            # EYEBALL DYNAMICS
            dx1_dt = v1
            dv1_dt = (torque_ag - torque_ant - passive_torque) / self.J_p
            
            # MUSCLE SPRING STATES (corrected with rotational stiffness)
            # Force balance: x4 (active) = K_lt_rot*x2 + K_se_rot*(x2-x1) + B*dx2/dt
            dx2_dt = (x4 - self.K_lt_rot * x2 - self.K_se_rot * (x2 - x1)) / B_ag
            dx3_dt = (x5 - self.K_lt_rot * x3 - self.K_se_rot * (x3 - x1)) / B_ant
            
            # MUSCLE ACTIVATION FILTERING (target displacement is filtered through muscle)
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
