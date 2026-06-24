"""
examples/ros2_node.py
=====================
Template for integrating BASHAR into a ROS 2 node.

This node:
  - Subscribes to /joint_states to feed encoder readings into BASHAR
  - Publishes trajectory commands to /joint_trajectory_controller/joint_trajectory
  - Exposes a simple service to trigger autonomous motion to a target XYZ

Dependencies (in addition to BASHAR):
    sudo apt install ros-humble-control-msgs

Adapt the topic names and message types to match your robot's controller manager.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Duration

import numpy as np
from bashar.api import BasharSystem, Trajectory

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROFILE_PATH  = "config/profiles/my_robot.json"
JOINT_NAMES   = [                          # must match your URDF joint names, active joints only
    "insertion_joint",
    "proximal_yaw_joint",
    "proximal_pitch_joint",
    "mid_yaw_joint",
    "mid_pitch_joint",
    "distal_yaw_joint",
    "distal_roll_joint",
]
CONTROL_TOPIC = "/joint_trajectory_controller/joint_trajectory"
STATE_TOPIC   = "/joint_states"
LOOP_HZ       = 50   # control loop frequency


class BasharROS2Node(Node):

    def __init__(self):
        super().__init__("bashar_node")

        # Boot the BASHAR spinal cord
        self.robot = BasharSystem(PROFILE_PATH)
        self.get_logger().info(f"BASHAR online: {self.robot.robot_name}")

        # Subscriber: read joint encoder feedback
        self.joint_state_sub = self.create_subscription(
            JointState, STATE_TOPIC, self._joint_state_callback, 10
        )

        # Publisher: send trajectory commands
        self.traj_pub = self.create_publisher(JointTrajectory, CONTROL_TOPIC, 10)

        # State
        self._current_positions = [0.0] * self.robot.state.num_joints
        self._obstacles         = []     # Brain updates this externally
        self._target_xyz        = None   # Set to trigger autonomous mode

        # Control loop timer
        self.timer = self.create_timer(1.0 / LOOP_HZ, self._control_loop)

    # -----------------------------------------------------------------------
    # Subscriber callbacks
    # -----------------------------------------------------------------------
    def _joint_state_callback(self, msg: JointState):
        """Receives encoder readings and updates BASHAR's internal state."""
        # Re-order positions to match our JOINT_NAMES order (ROS doesn't guarantee order)
        pos_map = dict(zip(msg.name, msg.position))
        ordered = [pos_map.get(name, 0.0) for name in JOINT_NAMES]
        self._current_positions = ordered

        try:
            self.robot.update_state(ordered)
        except ValueError as e:
            self.get_logger().warn(str(e))

    # -----------------------------------------------------------------------
    # Control loop
    # -----------------------------------------------------------------------
    def _control_loop(self):
        """
        Runs at LOOP_HZ. Chooses between autonomous and idle mode.
        Extend this to add manual mode by subscribing to a cmd_vel-style topic.
        """
        if self._target_xyz is not None:
            # Autonomous: step toward the target
            joints, reached = self.robot.auto_step(
                target_xyz=self._target_xyz,
                obstacles=self._obstacles
            )
            self._publish_trajectory(joints, duration_sec=1.0 / LOOP_HZ)

            if reached:
                self.get_logger().info("Target reached.")
                self._target_xyz = None

    # -----------------------------------------------------------------------
    # Publisher helpers
    # -----------------------------------------------------------------------
    def _publish_trajectory(self, joint_positions: list, duration_sec: float):
        """Publishes a single-point trajectory command."""
        msg = JointTrajectory()
        msg.joint_names = JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = joint_positions
        secs    = int(duration_sec)
        nanosec = int((duration_sec - secs) * 1e9)
        point.time_from_start = Duration(sec=secs, nanosec=nanosec)

        msg.points = [point]
        self.traj_pub.publish(msg)

    def send_planned_trajectory(self,
                                theta_start: list,
                                theta_end: list,
                                Tf: float = 3.0,
                                N: int = 100):
        """
        Plans a smooth quintic trajectory and publishes all N waypoints at once.
        Call this when you want the controller to handle interpolation itself.
        """
        path = Trajectory.joint_trajectory(theta_start, theta_end, Tf=Tf, N=N)

        msg = JointTrajectory()
        msg.joint_names = JOINT_NAMES

        for i, q in enumerate(path):
            point = JointTrajectoryPoint()
            point.positions = q.tolist()
            t_secs = Tf * i / (N - 1)
            point.time_from_start = Duration(
                sec=int(t_secs),
                nanosec=int((t_secs % 1) * 1e9)
            )
            msg.points.append(point)

        self.traj_pub.publish(msg)
        self.get_logger().info(f"Published trajectory with {N} waypoints over {Tf}s")

    # -----------------------------------------------------------------------
    # Public methods — call these from your Brain node via a service or topic
    # -----------------------------------------------------------------------
    def set_target(self, x: float, y: float, z: float):
        """Triggers autonomous navigation to [x, y, z] in the robot's base frame."""
        self._target_xyz = [x, y, z]
        self.get_logger().info(f"New target set: {self._target_xyz}")

    def update_obstacles(self, obstacles: list):
        """
        Update the obstacle list from the Brain.
        obstacles: list of [x, y, z] coordinates in the robot's base frame.
        """
        self._obstacles = obstacles


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(args=None):
    rclpy.init(args=args)
    node = BasharROS2Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
