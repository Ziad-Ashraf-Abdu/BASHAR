"""
examples/hardware_loop.py
=========================
Template for a bare-metal / non-ROS hardware control loop.

Shows the pattern for integrating BASHAR with any hardware interface —
whether that's a serial port, a CAN bus, a custom SDK, or just mock data.

The pattern is always:
    1. Read encoders  →  robot.update_state()
    2. Decide mode    →  manual_step() or auto_step()
    3. Compute torques → robot.calculate_motor_torques()
    4. Write to motors

Replace the stubbed hardware functions at the bottom with your actual driver calls.
"""

import time
import signal
import numpy as np

from bashar.api import BasharSystem, compile_profile, Trajectory

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROFILE_PATH = "config/profiles/my_robot.json"
LOOP_HZ      = 100         # control loop frequency (Hz)
DT           = 1.0 / LOOP_HZ

# Tuning — adjust to match your robot and task
INFLUENCE_RADIUS = 0.20    # APF influence zone in meters
BODY_RADIUS      = 0.08    # physical arm thickness in meters
DLS_DAMPING      = 0.05    # DLS damping factor; increase if arm is near singularity


# ---------------------------------------------------------------------------
# Hardware stub functions — replace these with your actual driver calls
# ---------------------------------------------------------------------------

def read_encoders(n_joints: int) -> list:
    """
    Read joint positions from hardware encoders.
    Returns a list of floats, one per active joint, in radians/meters.
    """
    # --- Replace with your actual encoder read ---
    # e.g., return your_sdk.get_joint_positions()
    return [0.0] * n_joints   # stub: all joints at zero


def read_joint_velocities(n_joints: int) -> list:
    """Read joint velocities (rad/s or m/s)."""
    # --- Replace with your actual velocity read ---
    return [0.0] * n_joints   # stub


def write_motor_torques(torques: list):
    """
    Send computed torques to motor drivers.
    torques: list of floats in Newton-meters, one per active joint.
    """
    # --- Replace with your actual motor write ---
    # e.g., your_sdk.set_torques(torques)
    pass   # stub


def get_obstacle_positions() -> list:
    """
    Read obstacle positions from the Brain (vision system, proximity sensors, etc.).
    Returns a list of [x, y, z] coordinates in the robot's base frame.
    Return an empty list if no obstacles are detected.
    """
    # --- Replace with your actual perception input ---
    return []   # stub: no obstacles


def get_mode() -> str:
    """
    Return the current control mode: 'idle', 'manual', or 'auto'.
    In a real system this might come from a joystick, a web UI, or a state machine.
    """
    return 'idle'   # stub


def get_manual_command(n_joints: int) -> list:
    """
    Return desired joint velocity deltas for manual mode.
    Typically comes from a joystick or keyboard input handler.
    """
    return [0.0] * n_joints   # stub


def get_auto_target() -> list:
    """
    Return the current [x, y, z] navigation target in the robot's base frame.
    """
    return [0.3, 0.0, 0.2]   # stub


# ---------------------------------------------------------------------------
# Main control loop
# ---------------------------------------------------------------------------

def main():
    # Boot the spinal cord
    print(f"Booting BASHAR from {PROFILE_PATH} ...")
    robot = BasharSystem(PROFILE_PATH)
    n     = robot.state.num_joints
    print(f"Online: {robot.robot_name}  ({n} active joints)")
    print(f"Loop frequency: {LOOP_HZ} Hz  |  DT: {DT*1000:.1f} ms")
    print("Press Ctrl+C to stop.\n")

    # Graceful shutdown on Ctrl+C
    running = True
    def _stop(sig, frame):
        nonlocal running
        running = False
    signal.signal(signal.SIGINT, _stop)

    # Optional: pre-plan a trajectory to home position on startup
    current_pos = read_encoders(n)
    home        = [0.0] * n
    startup_path = Trajectory.joint_trajectory_velocities(current_pos, home, Tf=2.0, N=200)

    print("Moving to home position...")
    for theta, theta_dot in startup_path:
        robot.update_state(read_encoders(n))
        torques = robot.calculate_motor_torques(
            current_dtheta=read_joint_velocities(n),
            desired_theta=theta.tolist(),
            desired_dtheta=theta_dot.tolist()
        )
        write_motor_torques(torques)
        time.sleep(DT)
    print("Home reached. Entering control loop.\n")

    # -----------------------------------------------------------------------
    # Real-time control loop
    # -----------------------------------------------------------------------
    loop_count = 0
    while running:
        t_start = time.monotonic()

        # 1. Read hardware state
        positions   = read_encoders(n)
        velocities  = read_joint_velocities(n)
        obstacles   = get_obstacle_positions()
        mode        = get_mode()

        # 2. Update BASHAR's internal model
        try:
            robot.update_state(positions)
        except ValueError as e:
            print(f"[WARN] {e}")
            time.sleep(DT)
            continue

        # 3. Execute the current control mode
        if mode == 'manual':
            # Brain provides joint velocity commands; BASHAR enforces safety
            command = get_manual_command(n)
            safe_joints = robot.manual_step(command, obstacles=obstacles)

            # Compute the torques needed to track the safe target positions
            torques = robot.calculate_motor_torques(
                current_dtheta=velocities,
                desired_theta=safe_joints,
                desired_dtheta=[0.0] * n   # targeting a stationary pose
            )
            write_motor_torques(torques)

        elif mode == 'auto':
            # Brain provides a target XYZ; BASHAR navigates there
            target_xyz = get_auto_target()
            new_joints, reached = robot.auto_step(target_xyz, obstacles=obstacles)

            torques = robot.calculate_motor_torques(
                current_dtheta=velocities,
                desired_theta=new_joints,
                desired_dtheta=[0.0] * n
            )
            write_motor_torques(torques)

            if reached and loop_count % 100 == 0:
                print(f"[{loop_count:6d}] Autonomous target reached.")

        # else: 'idle' — no output to motors

        # 4. Telemetry (every 100 ticks)
        if loop_count % 100 == 0:
            tip = robot.get_tip_position()
            print(f"[{loop_count:6d}] mode={mode:6s} | tip={[round(x, 4) for x in tip]}")

        # 5. Rate control — sleep for the remainder of the tick
        elapsed = time.monotonic() - t_start
        sleep_time = DT - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
        elif loop_count % 100 == 0:
            print(f"[WARN] Loop overrun by {-sleep_time*1000:.2f} ms")

        loop_count += 1

    print("\nControl loop stopped. Shutting down.")
    write_motor_torques([0.0] * n)   # zero torques on exit


if __name__ == "__main__":
    main()
