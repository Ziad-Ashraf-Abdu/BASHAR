"""
core/trajectory.py
==================
Time-scaling functions and path generation for joint-space and task-space trajectories.

All trajectory functions follow the same pattern:
    Input:  start config, end config, total time Tf, number of waypoints N
    Output: list of N configurations (joint angle arrays or SE(3) transforms)

The separation between time-scaling and trajectory functions is intentional:
    - Time-scaling functions are pure math, useful for tuning the motion profile
      independently of the path geometry.
    - Trajectory functions compose time-scaling with interpolation over a path.

Usage:
    from bashar.api import Trajectory

    # Generate 50 waypoints from home to a target pose over 3 seconds
    path = Trajectory.joint_trajectory(
        theta_start=[0.0] * 7,
        theta_end=[0.1, 0.0, 0.5, 0.0, -0.3, 0.0, 0.0],
        Tf=3.0, N=50, method='quintic'
    )

    # Feed into the torque controller
    for theta, theta_dot in Trajectory.joint_trajectory_velocities(...):
        torques = robot.calculate_motor_torques(theta_dot, theta, theta_dot)
"""

import numpy as np
import math


class Trajectory:

    # =========================================================================
    # Time Scaling Functions
    # Each returns (s, s_dot) — the normalized position [0,1] and velocity [1/s]
    # at time t within a motion of total duration Tf.
    # =========================================================================

    @staticmethod
    def cubic_time_scaling(Tf: float, t: float) -> tuple:
        """
        3rd-order polynomial time scaling.

        Boundary conditions:
            s(0) = 0,  s(Tf) = 1
            ṡ(0) = 0,  ṡ(Tf) = 0  (zero velocity at start and end)

        Returns: (s, s_dot) at time t.
        """
        t = float(np.clip(t, 0.0, Tf))
        r = t / Tf
        s     = 3 * r**2 - 2 * r**3
        s_dot = (6 * t / Tf**2) - (6 * t**2 / Tf**3)
        return s, s_dot

    @staticmethod
    def quintic_time_scaling(Tf: float, t: float) -> tuple:
        """
        5th-order polynomial time scaling.

        Boundary conditions:
            s(0) = s(Tf) = 0 → 1
            ṡ(0) = ṡ(Tf) = 0  (zero velocity)
            s̈(0) = s̈(Tf) = 0  (zero acceleration — smoother than cubic)

        Preferred over cubic when the robot is carrying a payload or operating
        close to its torque limits, as the smoother profile reduces peak torques.

        Returns: (s, s_dot) at time t.
        """
        t = float(np.clip(t, 0.0, Tf))
        r = t / Tf
        s     = 10 * r**3 - 15 * r**4 + 6 * r**5
        s_dot = (30 * t**2 / Tf**3) - (60 * t**3 / Tf**4) + (30 * t**4 / Tf**5)
        return s, s_dot

    @staticmethod
    def trapezoidal_time_scaling(Tf: float, t: float, accel_ratio: float = 0.3) -> tuple:
        """
        Bang-coast-bang (trapezoidal velocity) profile.

        The motion has three phases:
            1. Constant acceleration for (accel_ratio * Tf) seconds
            2. Constant velocity for the middle portion
            3. Constant deceleration (mirror of phase 1)

        Useful when you want maximum average velocity (fastest point-to-point)
        without infinite jerk. Cubic/quintic profiles have smoother starts but
        are slower on average.

        Args:
            accel_ratio: Fraction of Tf spent accelerating. Must be < 0.5.
                         Default 0.3 means 30% accel, 40% coast, 30% decel.

        Returns: (s, s_dot) at time t.
        """
        if accel_ratio >= 0.5:
            raise ValueError("accel_ratio must be < 0.5 (acceleration + deceleration must leave room for a coast phase).")

        t  = float(np.clip(t, 0.0, Tf))
        ta = accel_ratio * Tf   # duration of acceleration phase
        v_peak = 1.0 / (Tf - ta)  # peak velocity so total displacement = 1

        if t <= ta:
            # Acceleration phase
            s     = 0.5 * v_peak * t**2 / ta
            s_dot = v_peak * t / ta
        elif t <= Tf - ta:
            # Constant velocity phase
            s     = 0.5 * v_peak * ta + v_peak * (t - ta)
            s_dot = v_peak
        else:
            # Deceleration phase
            dt    = t - (Tf - ta)
            s     = (0.5 * v_peak * ta
                     + v_peak * (Tf - 2 * ta)
                     + v_peak * dt
                     - 0.5 * v_peak * dt**2 / ta)
            s_dot = v_peak * (1.0 - dt / ta)

        return s, s_dot

    # =========================================================================
    # Joint-Space Trajectory Generation
    # =========================================================================

    @classmethod
    def joint_trajectory(cls,
                         theta_start: list,
                         theta_end: list,
                         Tf: float,
                         N: int,
                         method: str = 'quintic') -> list:
        """
        Generates N joint configurations linearly interpolated from theta_start
        to theta_end with a chosen time-scaling profile.

        Args:
            theta_start: Starting joint angles/positions (list of floats, length n_joints).
            theta_end:   Target joint angles/positions  (list of floats, length n_joints).
            Tf:          Total motion duration in seconds.
            N:           Number of waypoints (including start and end).
            method:      Time-scaling profile — 'cubic', 'quintic', or 'trapezoidal'.

        Returns:
            List of N numpy arrays, each of shape (n_joints,).

        Example:
            path = Trajectory.joint_trajectory([0]*7, [0.5, 0, 0.3, 0, 0, 0, 0], Tf=2.0, N=100)
            for q in path:
                robot.update_state(q.tolist())
        """
        _METHODS = {'cubic', 'quintic', 'trapezoidal'}
        if method not in _METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from: {_METHODS}")

        theta_start = np.array(theta_start, dtype=float)
        theta_end   = np.array(theta_end,   dtype=float)
        delta       = theta_end - theta_start
        scale_fn    = cls._get_scale_fn(method)

        configs = []
        for t in np.linspace(0.0, Tf, N):
            s, _ = scale_fn(Tf, t)
            configs.append(theta_start + s * delta)

        return configs

    @classmethod
    def joint_trajectory_velocities(cls,
                                    theta_start: list,
                                    theta_end: list,
                                    Tf: float,
                                    N: int,
                                    method: str = 'quintic') -> list:
        """
        Like joint_trajectory() but also returns joint velocities at each waypoint.
        Useful for feeding into calculate_motor_torques() where desired_dtheta is needed.

        Returns:
            List of N tuples: [(theta_array, theta_dot_array), ...]
        """
        _METHODS = {'cubic', 'quintic', 'trapezoidal'}
        if method not in _METHODS:
            raise ValueError(f"Unknown method '{method}'. Choose from: {_METHODS}")

        theta_start = np.array(theta_start, dtype=float)
        theta_end   = np.array(theta_end,   dtype=float)
        delta       = theta_end - theta_start
        scale_fn    = cls._get_scale_fn(method)

        result = []
        for t in np.linspace(0.0, Tf, N):
            s, s_dot = scale_fn(Tf, t)
            theta     = theta_start + s     * delta
            theta_dot = s_dot * delta
            result.append((theta, theta_dot))

        return result

    @classmethod
    def via_point_trajectory(cls,
                             via_points: list,
                             segment_times: list,
                             N_per_segment: int = 50,
                             method: str = 'quintic') -> list:
        """
        Piecewise trajectory through a list of via points.
        Each segment uses independent time-scaling (the robot stops at each via point).

        Args:
            via_points:      List of joint configurations [[q1...], [q2...], ...].
                             Must have at least 2 points.
            segment_times:   Duration of each segment in seconds.
                             Length must be len(via_points) - 1.
            N_per_segment:   Waypoints per segment.
            method:          Time-scaling profile.

        Returns:
            Concatenated list of joint configurations along the full path.
        """
        if len(via_points) < 2:
            raise ValueError("Need at least 2 via points.")
        if len(segment_times) != len(via_points) - 1:
            raise ValueError("segment_times must have exactly len(via_points) - 1 entries.")

        full_path = []
        for i in range(len(via_points) - 1):
            segment = cls.joint_trajectory(
                via_points[i], via_points[i + 1],
                Tf=segment_times[i], N=N_per_segment, method=method
            )
            # Drop the last point of each segment to avoid duplicates at junction
            if i < len(via_points) - 2:
                segment = segment[:-1]
            full_path.extend(segment)

        return full_path

    # =========================================================================
    # Task-Space (Screw) Trajectory Generation
    # =========================================================================

    @classmethod
    def screw_trajectory(cls,
                         T_start: np.ndarray,
                         T_end: np.ndarray,
                         Tf: float,
                         N: int,
                         method: str = 'quintic') -> list:
        """
        Generates N SE(3) transforms along a constant-screw (straight task-space) path
        from T_start to T_end.

        A screw trajectory is the shortest path in SE(3): the end-effector moves
        in a straight line in position while simultaneously rotating about a fixed axis.
        Contrast with Cartesian trajectories where rotation and translation are decoupled.

        Args:
            T_start: 4×4 SE(3) starting transform.
            T_end:   4×4 SE(3) target transform.
            Tf:      Total duration in seconds.
            N:       Number of waypoints.
            method:  Time-scaling profile.

        Returns:
            List of N 4×4 numpy arrays.

        Example:
            T_home   = robot.kinematics.forward_kinematics_space([0]*7)
            T_target = ... # desired end-effector pose
            path_SE3 = Trajectory.screw_trajectory(T_home, T_target, Tf=2.0, N=50)
            for T in path_SE3:
                theta, success = robot.kinematics.ik_body(T, current_theta)
        """
        scale_fn = cls._get_scale_fn(method)

        # Relative transform: T_start → T_end
        T_rel     = cls._trans_inv(T_start) @ T_end
        S_theta   = cls._matrix_log_6(T_rel)   # Screw axis * angle in se(3)

        transforms = []
        for t in np.linspace(0.0, Tf, N):
            s, _ = scale_fn(Tf, t)
            T_i  = T_start @ cls._matrix_exp_6(S_theta, s)
            transforms.append(T_i)

        return transforms

    # =========================================================================
    # Private Helpers — SE(3) Math (duplicated here to keep trajectory.py
    # self-contained without requiring a BasharKinematics instance)
    # =========================================================================

    @staticmethod
    def _get_scale_fn(method: str):
        return {
            'cubic':       Trajectory.cubic_time_scaling,
            'quintic':     Trajectory.quintic_time_scaling,
            'trapezoidal': Trajectory.trapezoidal_time_scaling,
        }[method]

    @staticmethod
    def _trans_inv(T: np.ndarray) -> np.ndarray:
        R, p = T[:3, :3], T[:3, 3]
        T_inv = np.eye(4)
        T_inv[:3, :3] = R.T
        T_inv[:3,  3] = -R.T @ p
        return T_inv

    @staticmethod
    def _skew(v: np.ndarray) -> np.ndarray:
        return np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])

    @classmethod
    def _matrix_exp_6(cls, S_theta: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """Matrix exponential of a scaled se(3) vector."""
        V     = S_theta * scale
        omega = V[:3]
        v     = V[3:]
        T     = np.eye(4)

        if np.linalg.norm(omega) < 1e-6:
            T[:3, 3] = v
            return T

        omega_skew = cls._skew(omega)
        theta_sq   = np.dot(omega, omega)
        theta      = math.sqrt(theta_sq)
        R = (np.eye(3)
             + math.sin(theta) / theta * omega_skew
             + (1 - math.cos(theta)) / theta_sq * omega_skew @ omega_skew)
        G = (np.eye(3)
             + (1 - math.cos(theta)) / theta_sq * omega_skew
             + (theta - math.sin(theta)) / (theta_sq * theta) * omega_skew @ omega_skew)
        T[:3, :3] = R
        T[:3,  3] = G @ v
        return T

    @classmethod
    def _matrix_log_6(cls, T: np.ndarray) -> np.ndarray:
        """Matrix logarithm of an SE(3) transform → se(3) vector."""
        R = T[:3, :3]
        p = T[:3,  3]

        acos_in = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        theta   = math.acos(acos_in)

        if abs(theta) < 1e-6:
            omega_skew = np.zeros((3, 3))
        elif abs(theta - math.pi) < 1e-6:
            # Special case: 180° rotation
            diag = np.diag(R)
            idx  = int(np.argmax(diag))
            col  = (R[:, idx] + np.eye(3)[:, idx]) / math.sqrt(2 * (1 + R[idx, idx]))
            omega_skew = cls._skew(col * theta)
        else:
            omega_skew = theta / (2 * math.sin(theta)) * (R - R.T)

        if np.allclose(omega_skew, 0):
            v_theta = p
        else:
            omega = np.array([omega_skew[2, 1], omega_skew[0, 2], omega_skew[1, 0]])
            G_inv  = (np.eye(3)
                      - 0.5 * omega_skew
                      + (1.0 / theta - 0.5 / math.tan(theta / 2.0))
                        * (omega_skew @ omega_skew) / theta)
            v_theta = G_inv @ p
            omega   = np.array([omega_skew[2, 1], omega_skew[0, 2], omega_skew[1, 0]])
            return np.concatenate((omega, v_theta))

        return np.concatenate((np.zeros(3), v_theta))
