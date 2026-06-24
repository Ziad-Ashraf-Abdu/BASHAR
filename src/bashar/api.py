import os
import numpy as np

# Import the hidden internal engines
from bashar.utils.model_compiler import ModelCompiler
from bashar.utils.config_parser import ProfileParser
from bashar.core.kinematics import BasharKinematics
from bashar.core.dynamics import BasharDynamics
from bashar.core.controller import RobotState, CollisionGuard, AutoPilot, ComputedTorqueController


def compile_profile(model_path: str, output_name: str, output_dir: str = "config/profiles", verbose: bool = False) -> bool:
    """
    Utility hook for the user to compile their URDF/XACRO into a BASHAR JSON profile.
    """
    compiler = ModelCompiler()
    return compiler.compile(model_path, output_name, output_dir, verbose)


class BasharSystem:
    """
    The main Application Programming Interface (API) for the BASHAR library.
    Encapsulates all Kinematics, Dynamics, and Control logic behind a clean, high-level interface.
    """
    def __init__(self, profile_path: str, initial_positions: list = None):
        """
        Initializes the entire "Spinal Cord" from a compiled JSON profile.
        """
        # --- The Gatekeeper: Safely loads and validates the JSON before booting ---
        profile_dict = ProfileParser.load(profile_path)

        self.robot_name = profile_dict.get("robot_name", "Unknown Robot")
        
        # 1. Boot the Mathematical Engines
        self.kinematics = BasharKinematics(profile_dict)
        self.dynamics = BasharDynamics(self.kinematics)
        
        # 2. Boot the State Manager
        self.state = RobotState(profile_dict, initial_positions)
        
        # 3. Boot the Control Layers
        self.guard = CollisionGuard(self.kinematics)
        self.autopilot = AutoPilot(self.kinematics)
        self.torque_controller = ComputedTorqueController(self.dynamics)

    # -----------------------------------------------------------------------
    # STATE MANAGEMENT
    # -----------------------------------------------------------------------
    def update_state(self, current_positions: list):
        """Updates the internal mathematical state to match the physical hardware encoders."""
        n = self.state.num_joints
        if len(current_positions) != n:
            raise ValueError(
                f"[BASHAR] update_state received {len(current_positions)} positions "
                f"but the robot '{self.robot_name}' has {n} active joints. "
                f"Pass exactly {n} values."
            )
        self.state.positions = np.array(current_positions, dtype=float)
        self.state.clip()

    def get_state(self) -> list:
        """Returns the current clipped joint positions."""
        return self.state.as_list

    def get_tip_position(self) -> list:
        """Returns the current [X, Y, Z] spatial coordinate of the end-effector."""
        T_sb = self.kinematics.forward_kinematics_space(self.state.positions)
        return T_sb[:3, 3].tolist()

    # -----------------------------------------------------------------------
    # NAVIGATION & CONTROL MODES
    # -----------------------------------------------------------------------
    def manual_step(self, desired_dtheta: list, obstacles: list = None) -> list:
        """
        [Manual Mode] Safely filters user velocity commands through the APF Collision Guard.
        Returns the safe target joint positions for this tick.
        """
        if obstacles is None:
            obstacles = []
        self.state = self.guard.filter_command(self.state, desired_dtheta, obstacles)
        return self.state.as_list

    def auto_step(self, target_xyz: list, obstacles: list = None) -> tuple[list, bool]:
        """
        [Autonomous Mode] Drives the robot towards an XYZ coordinate while dodging obstacles.
        Returns (New Joint Positions, Boolean indicating if the target was reached).
        """
        if obstacles is None:
            obstacles = []
        self.state, reached = self.autopilot.step(self.state, target_xyz, obstacles)
        return self.state.as_list, reached

    # -----------------------------------------------------------------------
    # DYNAMICS (HARDWARE EXECUTION)
    # -----------------------------------------------------------------------
    def calculate_motor_torques(self, current_dtheta: list, desired_theta: list, desired_dtheta: list) -> list:
        """
        Executes Computed Torque Control (Feedforward + Feedback).
        Returns the exact torque (Newton-meters) required by each physical motor to achieve the desired state.
        """
        tau = self.torque_controller.compute_torques(
            state=self.state, 
            current_dtheta=current_dtheta, 
            desired_theta=desired_theta, 
            desired_dtheta=desired_dtheta
        )
        return tau.tolist()