import numpy as np
from bashar.api import BasharSystem, compile_profile
import os

def run_integration_test():
    print("--- BASHAR INTEGRATION TEST START ---")
    
    # 1. Compile the Test URDF
    # We use the lift_mechanism.xacro as it has clear joint limits and mass properties
    model_path = "/home/zee/Endoscopy_Robot/smart_endoscopy/src/smart_endoscope/urdf/smart_broncho.urdf"
    profile_name = "test_profile"
    
    success = compile_profile(model_path, profile_name, verbose=True)
    if not success:
        print("[TEST FAILED] Compilation aborted.")
        return

    # 2. Boot the Spinal Cord
    robot = BasharSystem(f"config/profiles/{profile_name}.json")
    print(f"System Booted: {robot.robot_name}")

    # 3. Test Kinematics: Calculate a forward pose
    # 7 active joints: insertion(prismatic), proximal_yaw, proximal_pitch, mid_yaw, mid_pitch, distal_yaw, distal_roll
    test_pose = [0.1, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0]
    robot.update_state(test_pose)
    tip = robot.get_tip_position()
    print(f"Test Kinematics: Tip position = {tip}")

    # 4. Test Dynamics: Verify torque calculation
    tau = robot.calculate_motor_torques(
        current_dtheta=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        desired_theta=[0.1, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0],
        desired_dtheta=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    print(f"Test Dynamics: Required motor torques = {tau} Nm")

    # 5. Test Safety: Collision Guard
    # We command a move that would collide with a wall at [0.5, 0.0, 0.0]
    dangerous_move = [0.5, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0]
    obstacles = [[0.6, 0.0, 0.0]] 
    
    safe_pose = robot.manual_step(dangerous_move, obstacles=obstacles)
    print(f"Test Safety: Collision Guard filtered move to = {safe_pose}")
    
    print("--- BASHAR INTEGRATION TEST PASSED ---")

if __name__ == "__main__":
    run_integration_test()