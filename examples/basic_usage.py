"""
examples/basic_usage.py
=======================
End-to-end walkthrough of the BASHAR library.

This script demonstrates the full pipeline:
  1. Compiling a robot profile from a URDF
  2. Booting the system
  3. Forward kinematics
  4. Trajectory planning (joint-space)
  5. Computed torque control along the trajectory
  6. Manual mode with collision avoidance
  7. Autonomous stepping toward a target

Replace MODEL_PATH with your own URDF or XACRO file path.
"""

import numpy as np
from bashar.api import BasharSystem, compile_profile, Trajectory

# ---------------------------------------------------------------------------
# Configuration — edit these for your robot
# ---------------------------------------------------------------------------
MODEL_PATH   = "/path/to/your_robot.urdf"   # URDF or XACRO
PROFILE_NAME = "my_robot"
PROFILE_PATH = f"config/profiles/{PROFILE_NAME}.json"

N_JOINTS = 7   # Update to match your robot's active DOF after compiling

# ---------------------------------------------------------------------------
# Step 1: Compile the robot profile (run once, then reuse the JSON)
# ---------------------------------------------------------------------------
print("=== Step 1: Compiling robot profile ===")
success = compile_profile(MODEL_PATH, PROFILE_NAME, verbose=True)
if not success:
    raise RuntimeError("Profile compilation failed. Check your URDF path and syntax.")

# ---------------------------------------------------------------------------
# Step 2: Boot the system
# ---------------------------------------------------------------------------
print("\n=== Step 2: Booting BASHAR ===")
robot = BasharSystem(PROFILE_PATH)
print(f"System online: {robot.robot_name}  ({robot.state.num_joints} active joints)")

# ---------------------------------------------------------------------------
# Step 3: Forward kinematics — where is the tip right now?
# ---------------------------------------------------------------------------
print("\n=== Step 3: Forward Kinematics ===")
home = [0.0] * robot.state.num_joints
robot.update_state(home)
tip = robot.get_tip_position()
print(f"Tip at home config: {[round(x, 4) for x in tip]}")

# ---------------------------------------------------------------------------
# Step 4: Plan a joint-space trajectory from home to a target config
# ---------------------------------------------------------------------------
print("\n=== Step 4: Trajectory Planning ===")
target_config = [0.1, 0.0, 0.4, 0.0, -0.2, 0.0, 0.0]  # adjust for your robot

# quintic gives zero velocity AND zero acceleration at start/end — best for payload safety
path = Trajectory.joint_trajectory(home, target_config, Tf=3.0, N=100, method='quintic')
path_with_vel = Trajectory.joint_trajectory_velocities(home, target_config, Tf=3.0, N=100)

print(f"Generated {len(path)} waypoints")
print(f"Waypoint 0:  {[round(x, 4) for x in path[0]]}")
print(f"Waypoint 50: {[round(x, 4) for x in path[50]]}")
print(f"Waypoint 99: {[round(x, 4) for x in path[99]]}")

# ---------------------------------------------------------------------------
# Step 5: Simulate computed torque control along the trajectory
# ---------------------------------------------------------------------------
print("\n=== Step 5: Computed Torque Control (first 5 waypoints) ===")
robot.update_state(home)
for i, (theta, theta_dot) in enumerate(path_with_vel[:5]):
    torques = robot.calculate_motor_torques(
        current_dtheta=robot.get_state(),    # pretend current velocity = current position (demo only)
        desired_theta=theta.tolist(),
        desired_dtheta=theta_dot.tolist()
    )
    print(f"  Waypoint {i:2d} | θ[0]={theta[0]:.4f}  τ[0]={torques[0]:.4f} Nm")

# ---------------------------------------------------------------------------
# Step 6: Manual mode — move with collision avoidance active
# ---------------------------------------------------------------------------
print("\n=== Step 6: Manual Step with Collision Guard ===")
robot.update_state(home)

# The brain detected two obstacles in the robot's base frame
obstacles = [
    [0.40,  0.10, 0.20],   # obstacle 1
    [0.35, -0.05, 0.15],   # obstacle 2
]
command = [0.02, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0]  # intended joint velocity

safe_joints = robot.manual_step(command, obstacles=obstacles)
print(f"Commanded: {command}")
print(f"Safe output: {[round(x, 4) for x in safe_joints]}")

# ---------------------------------------------------------------------------
# Step 7: Autonomous mode — step toward a task-space target
# ---------------------------------------------------------------------------
print("\n=== Step 7: Autonomous Stepping (10 ticks) ===")
robot.update_state(home)
target_xyz = [tip[0] + 0.10, tip[1], tip[2] + 0.05]   # 10cm forward, 5cm up

for tick in range(10):
    joints, reached = robot.auto_step(target_xyz, obstacles=obstacles)
    current_tip = robot.get_tip_position()
    dist = np.linalg.norm(np.array(target_xyz) - np.array(current_tip))
    print(f"  Tick {tick:2d} | tip={[round(x, 4) for x in current_tip]}  dist_to_goal={dist:.4f}m")
    if reached:
        print("  Target reached!")
        break

print("\n=== Done ===")
