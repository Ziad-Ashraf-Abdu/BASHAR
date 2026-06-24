import numpy as np
from bashar.core.kinematics import BasharKinematics
from bashar.core.dynamics import BasharDynamics

class RobotState:
    """
    Dynamically manages the state (joint angles/positions) of an N-DOF robot.
    Enforces mechanical limits automatically based on the compiled JSON profile.
    """
    def __init__(self, profile_dict: dict, initial_positions: list = None):
        _active_types = {'revolute', 'continuous', 'prismatic'}
        _active_joints = [
            j for j in profile_dict['kinematic_tree']['joints']
            if j.get('type', 'fixed') in _active_types
        ]
        self.num_joints = len(_active_joints)

        # Extract the specific safety limits for every active joint
        self.limits = [j['limits'] for j in _active_joints]
        
        if initial_positions is not None:
            if len(initial_positions) != self.num_joints:
                raise ValueError(f"Expected {self.num_joints} initial positions, got {len(initial_positions)}")
            self.positions = np.array(initial_positions, dtype=float)
        else:
            self.positions = np.zeros(self.num_joints, dtype=float)
            
        self.clip() # Enforce limits immediately upon initialization
        
    @property
    def as_list(self) -> list:
        return self.positions.tolist()
        
    def clip(self):
        """Restricts joint positions to their hardware limits."""
        for i in range(self.num_joints):
            self.positions[i] = np.clip(
                self.positions[i], 
                self.limits[i]['lower'], 
                self.limits[i]['upper']
            )
        return self
        
    def apply_delta(self, delta_theta: np.ndarray):
        """Applies a velocity/delta vector and automatically clips the result."""
        self.positions += np.array(delta_theta, dtype=float)
        self.clip()
        return self


# FIXED: CollisionGuard is now correctly at the module level (un-nested)
class CollisionGuard:
    """
    Hardware-agnostic safety layer using Artificial Potential Fields (APF).
    
    CRITICAL FRAME REQUIREMENT:
    All obstacle coordinates passed to this class MUST be relative to the robot's 
    Global Base Frame (x=0, y=0, z=0).
    """
    def __init__(self, kin: BasharKinematics, influence_radius=0.20, repulse_gain=1.5, body_radius=0.08, damping=0.05):
        self.kin = kin
        self.influence_radius = influence_radius  
        self.repulse_gain = repulse_gain          
        self.body_radius = body_radius
        self.damping = damping  # DLS damping factor for Jacobian inversion near singularities

    def _sanitize_obstacles(self, raw_obstacles, current_z: float) -> list:
        if not raw_obstacles:
            return []
            
        if isinstance(raw_obstacles, (list, tuple, np.ndarray)):
            if len(raw_obstacles) > 0 and not isinstance(raw_obstacles[0], (list, tuple, np.ndarray)):
                raw_obstacles = [raw_obstacles]
                
        clean_obs = []
        for obs in raw_obstacles:
            try:
                obs_array = np.array(obs, dtype=float)
                if obs_array.shape == (3,):
                    clean_obs.append(obs_array)
                elif obs_array.shape == (2,):
                    clean_obs.append(np.array([obs_array[0], obs_array[1], current_z]))
            except Exception:
                pass # Silently ignore bad data to keep the control loop running
                
        return clean_obs

    def filter_command(self, state: RobotState, d_theta: list, obstacles: list) -> RobotState:
        T_sb = self.kin.forward_kinematics_space(state.positions)
        tip_xyz = T_sb[:3, 3]

        clean_obstacles = self._sanitize_obstacles(obstacles, current_z=tip_xyz[2])
        V_repulse = np.zeros(6) 
        effective_boundary = self.influence_radius + self.body_radius
        
        for obs_xyz in clean_obstacles:
            direction_vector = tip_xyz - obs_xyz
            actual_distance = np.linalg.norm(direction_vector)
            
            if 0 < actual_distance < effective_boundary:
                penetration_depth = effective_boundary - actual_distance
                force_magnitude = self.repulse_gain * (penetration_depth / actual_distance) ** 2
                push_vector = (direction_vector / actual_distance) * force_magnitude
                V_repulse[3:] += push_vector

        d_theta_correction = np.zeros(state.num_joints)
        if np.any(V_repulse[3:] != 0):
            Js = self.kin.jacobian_space(state.positions)
            d_theta_correction = self.kin.dls_pinv(Js, self.damping) @ V_repulse

        d_theta_correction = np.clip(d_theta_correction, -0.6, 0.6)
        final_d_theta = np.array(d_theta, dtype=float) + d_theta_correction
        
        state.apply_delta(final_d_theta)
        return state


# NEW: The Generalized AutoPilot
class AutoPilot:
    """
    Hardware-agnostic Autonomous Navigation.
    Drives the robot to a specific [x, y, z] target while avoiding obstacles.
    """
    def __init__(self, kin: BasharKinematics, kp_linear=1.0, influence_radius=0.20, repulse_gain=1.5, body_radius=0.08, damping=0.05):
        self.kin = kin
        self.kp_linear = kp_linear
        self.damping = damping  # DLS damping factor
        self.guard = CollisionGuard(kin, influence_radius, repulse_gain, body_radius, damping)

    def step(self, state: RobotState, target_xyz: list, obstacles: list, threshold=0.01) -> tuple[RobotState, bool]:
        """
        Executes one control loop tick towards the target.
        Returns: (Updated RobotState, Boolean flag indicating if the target is reached)
        """
        # 1. Find where the robot currently is
        T_sb = self.kin.forward_kinematics_space(state.positions)
        tip_xyz = T_sb[:3, 3]
        
        target = np.array(target_xyz, dtype=float)
        error_vector = target - tip_xyz
        distance_to_goal = np.linalg.norm(error_vector)
        
        # Check if we have successfully reached the destination
        if distance_to_goal <= threshold:
            return state, True
            
        # 2. Calculate the Attractive Velocity (Pulling toward the goal)
        speed_cap = 0.1  # Maximum task-space velocity (meters/tick)
        v_pull = (error_vector / distance_to_goal) * min(self.kp_linear * distance_to_goal, speed_cap)
        
        # Pack it into a spatial velocity vector [omega_x, omega_y, omega_z, v_x, v_y, v_z]
        V_attract = np.array([0.0, 0.0, 0.0, v_pull[0], v_pull[1], v_pull[2]])
        
        # 3. Map the task-space pull into joint-space angles via DLS (stable near singularities)
        Js = self.kin.jacobian_space(state.positions)
        d_theta_attract = self.kin.dls_pinv(Js, self.damping) @ V_attract
        # 4. The Magic: Pass the intended trajectory through the Collision Guard
        # If the direct path goes through a wall, the Guard will automatically bend the arm around it.
        state = self.guard.filter_command(state, d_theta_attract.tolist(), obstacles)
        
        return state, False
    
class ComputedTorqueController:
    """
    Executes true Feedforward + Feedback control.
    Predicts the physics of the arm (gravity, inertia, Coriolis) and 
    calculates the exact motor torques (tau) required to track a trajectory safely.
    """
    def __init__(self, dynamics: BasharDynamics, kp: float = 150.0, kd: float = 40.0):
        self.dyn = dynamics
        
        # Proportional Gain (Stiffness): How aggressively it corrects position errors
        self.kp = kp
        
        # Derivative Gain (Damping): How smoothly it brakes to prevent overshoot/oscillations
        self.kd = kd

    def compute_torques(self, state: RobotState, current_dtheta: list, 
                        desired_theta: list, desired_dtheta: list, desired_ddtheta: list = None) -> np.ndarray:
        """
        Calculates the exact torque (tau) for every motor.
        
        Equation: tau = M(θ) * [θ_ddot_des + Kp(error) + Kd(error_dot)] + c(θ, θ_dot) + g(θ)
        """
        theta = state.positions
        dtheta = np.array(current_dtheta, dtype=float)
        
        theta_des = np.array(desired_theta, dtype=float)
        dtheta_des = np.array(desired_dtheta, dtype=float)
        
        # If no acceleration profile is provided, default to 0 (cruising speed)
        ddtheta_des = np.array(desired_ddtheta, dtype=float) if desired_ddtheta is not None else np.zeros(state.num_joints)

        # 1. FEEDFORWARD: Predict the physics (The heavy lifting)
        # We calculate the Mass Matrix, Coriolis effects, and Gravity vectors instantly
        M = self.dyn.mass_matrix(theta)
        c = self.dyn.velocity_quadratic_forces(theta, dtheta)
        g = self.dyn.gravity_forces(theta)

        # 2. FEEDBACK: Calculate the real-time errors
        error = theta_des - theta
        error_dot = dtheta_des - dtheta

        # 3. THE CONTROL LAW: Combine physics prediction with error correction
        # pd_correction acts as a virtual spring-damper pulling the robot to the perfect path
        pd_correction = (self.kp * error) + (self.kd * error_dot)
        
        # Calculate the final required motor torques (tau)
        tau = M @ (ddtheta_des + pd_correction) + c + g
        
        return tau