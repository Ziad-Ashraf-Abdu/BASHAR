import numpy as np
import math

class BasharKinematics:

    def __init__(self, profile_dict: dict):
        """
        Initializes the kinematic chain based on the compiled JSON profile.
        """
        self.robot_name = profile_dict.get('robot_name', 'bashar_robot')
        self.joints = profile_dict['kinematic_tree']['joints']
        self.num_joints = len(self.joints)
        
        # Core PoE matrices to be populated dynamically
        self.M = np.eye(4)      # Home configuration matrix (End-effector position at theta=0)
        self.S_list = []        # Screw axes in the space frame
        
        # Spatial inertia matrices and gravity for dynamics
        self.Glist = []
        self.Mlist = []
        self.gravity = np.array([0, 0, -9.81])

        # Build the kinematics tree from the profile
        self._build_poe_from_profile()

    @staticmethod
    def _rpy_to_rotation(rpy):
        """Converts Roll-Pitch-Yaw (XYZ extrinsic) to a 3x3 Rotation Matrix."""
        r, p, y_yaw = rpy
        
        # Roll (X-axis)
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(r), -math.sin(r)],
            [0, math.sin(r), math.cos(r)]
        ])
        
        # Pitch (Y-axis)
        Ry = np.array([
            [math.cos(p), 0, math.sin(p)],
            [0, 1, 0],
            [-math.sin(p), 0, math.cos(p)]
        ])
        
        # Yaw (Z-axis)
        Rz = np.array([
            [math.cos(y_yaw), -math.sin(y_yaw), 0],
            [math.sin(y_yaw), math.cos(y_yaw), 0],
            [0, 0, 1]
        ])
        
        return Rz @ Ry @ Rx  # Standard URDF rotation multiplication order
    
    def _build_poe_from_profile(self):
        """
        Translates sequential URDF-style kinematics into Global PoE Screw Axes.
        Constructs the M matrix, S_list, and dynamic properties (Glist, Mlist).
        """
        T_accum = np.eye(4)
        active_joints_count = 0
        
        for joint in self.joints:
            # 1. Parse the local transform from the parent to this joint
            xyz = np.array(joint['origin']['xyz'])
            rpy = joint['origin']['rpy']
            
            R_local = self._rpy_to_rotation(rpy)
            T_local = np.eye(4)
            T_local[:3, :3] = R_local
            T_local[:3,  3] = xyz
            
            # 2. Update our global position
            T_accum = T_accum @ T_local
            
            R_global = T_accum[:3, :3]
            q_global = T_accum[:3,  3]
            
            # 3. Calculate the Screw Axis (S) and dynamic properties for active joints
            j_type = joint['type']
            if j_type in ['revolute', 'continuous', 'prismatic']:
                axis_local = np.array(joint['axis'])
                
                if j_type in ['revolute', 'continuous']:
                    omega = R_global @ axis_local
                    v = np.cross(q_global, omega)
                    S = np.concatenate((omega, v))
                elif j_type == 'prismatic':
                    omega = np.zeros(3)
                    v = R_global @ axis_local
                    S = np.concatenate((omega, v))
                    
                self.S_list.append(S)
                
                # --- NEW: Extract and build dynamic properties (Mlist & Glist) ---
                inertial = joint.get('inertial', {})
                mass = inertial.get('mass', 0.0)
                I_dict = inertial.get('inertia', {"ixx":0, "ixy":0, "ixz":0, "iyy":0, "iyz":0, "izz":0})
                
                # Build the 6x6 Spatial Inertia Matrix (G)
                I_tensor = np.array([
                    [I_dict['ixx'], I_dict['ixy'], I_dict['ixz']],
                    [I_dict['ixy'], I_dict['iyy'], I_dict['iyz']],
                    [I_dict['ixz'], I_dict['iyz'], I_dict['izz']]
                ])
                G = np.zeros((6, 6))
                G[:3, :3] = I_tensor
                G[3:, 3:] = np.eye(3) * mass
                self.Glist.append(G)
                
                # Build the Center of Mass global frame (M_i)
                com_xyz = np.array(inertial.get('origin', {}).get('xyz', [0, 0, 0]))
                com_rpy = inertial.get('origin', {}).get('rpy', [0, 0, 0])
                
                R_com = self._rpy_to_rotation(com_rpy)
                T_com_local = np.eye(4)
                T_com_local[:3, :3] = R_com
                T_com_local[:3,  3] = com_xyz
                
                # The COM frame is offset from the joint's frame
                T_com_global = T_accum @ T_com_local
                self.Mlist.append(T_com_global)
                # -----------------------------------------------------------------

                active_joints_count += 1
                
        # 4. The final accumulated transform is the Home Configuration (M) of the end-effector
        self.M = T_accum
        self.num_joints = active_joints_count

    # =========================================================================
    # Rigid-Body Motions
    # =========================================================================

    @staticmethod
    def skew_symmetric(vec3):
        return np.array([
            [0,        -vec3[2],  vec3[1]],
            [vec3[2],   0,       -vec3[0]],
            [-vec3[1],  vec3[0],  0      ],
        ])

    def trans_inv(self, T):
        R = T[:3, :3]
        p = T[:3,  3]
        T_inv = np.eye(4)
        T_inv[:3, :3] = R.T
        T_inv[:3,  3] = -R.T @ p
        return T_inv

    def matrix_exp_6(self, S, theta):
        omega = S[:3]
        v     = S[3:]
        T     = np.eye(4)

        if np.linalg.norm(omega) < 1e-6: 
            T[:3, 3] = v * theta
            return T

        omega_skew = self.skew_symmetric(omega)
        R = (np.eye(3)
             + np.sin(theta) * omega_skew
             + (1 - np.cos(theta)) * omega_skew @ omega_skew)
        G = (np.eye(3) * theta
             + (1 - np.cos(theta)) * omega_skew
             + (theta - np.sin(theta)) * omega_skew @ omega_skew)

        T[:3, :3] = R
        T[:3,  3] = G @ v
        return T

    def adjoint_matrix(self, T):
        R = T[:3, :3]
        p = T[:3,  3]
        p_skew = self.skew_symmetric(p)

        AdT = np.zeros((6, 6))
        AdT[:3, :3] = R
        AdT[3:, :3] = p_skew @ R
        AdT[3:, 3:] = R
        return AdT

    def matrix_log_3(self, R):
        acos_input = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        theta = np.arccos(acos_input)

        if np.isclose(theta, 0.0): return np.zeros((3, 3))
        if np.isclose(theta, np.pi):
            if not np.isclose(1 + R[2, 2], 0.0): omega = (1.0 / np.sqrt(2 * (1 + R[2, 2]))) * np.array([R[0, 2], R[1, 2], 1 + R[2, 2]])
            elif not np.isclose(1 + R[1, 1], 0.0): omega = (1.0 / np.sqrt(2 * (1 + R[1, 1]))) * np.array([R[0, 1], 1 + R[1, 1], R[2, 1]])
            else: omega = (1.0 / np.sqrt(2 * (1 + R[0, 0]))) * np.array([1 + R[0, 0], R[1, 0], R[2, 0]])
            return self.skew_symmetric(omega) * np.pi

        return (theta / (2 * np.sin(theta))) * (R - R.T)

    def matrix_log_6(self, T):
        R = T[:3, :3]
        p = T[:3,  3]
        omega_skew = self.matrix_log_3(R)

        if np.allclose(omega_skew, 0): v_theta = p
        else:
            theta = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
            G_inv = (np.eye(3) - 0.5 * omega_skew + (1.0 / theta - 0.5 / np.tan(theta / 2.0)) * (omega_skew @ omega_skew) / theta)
            v_theta = G_inv @ p

        omega_theta = np.array([omega_skew[2, 1], omega_skew[0, 2], omega_skew[1, 0]])
        return np.concatenate((omega_theta, v_theta))

    # =========================================================================
    # Forward Kinematics 
    # =========================================================================

    def forward_kinematics_space(self, theta_list):
        T = np.eye(4)
        for S, theta in zip(self.S_list, theta_list):
            T = T @ self.matrix_exp_6(S, theta)
        return T @ self.M

    # =========================================================================
    # Velocity Kinematics & Statics
    # =========================================================================

    def jacobian_space(self, theta_list):
        """
        Dynamically calculates the Space Jacobian for n joints.
        Js(θ) = [S1, Ad_T1(S2), Ad_T12(S3), ... ]
        """
        n = len(theta_list)
        Js = np.zeros((6, n))
        T = np.eye(4)
        
        for i in range(n):
            if i == 0:
                Js[:, i] = self.S_list[i]
            else:
                T = T @ self.matrix_exp_6(self.S_list[i-1], theta_list[i-1])
                Js[:, i] = self.adjoint_matrix(T) @ self.S_list[i]
                
        return Js

    def jacobian_body(self, Js, T_final):
        return self.adjoint_matrix(self.trans_inv(T_final)) @ Js

    def ellipsoid_analysis(self, J):
        J_omega = J[:3, :]
        J_v     = J[3:, :]

        def calculate_mu(A_mat):
            eigs = np.clip(np.linalg.eigvalsh(A_mat), 0.0, None)
            eigs = np.sort(eigs)
            lmin, lmax = eigs[0], eigs[-1]
            mu3 = np.sqrt(max(0.0, np.linalg.det(A_mat)))
            if np.isclose(lmin, 0.0, atol=1e-6): return float('inf'), float('inf'), mu3
            return np.sqrt(lmax / lmin), lmax / lmin, mu3

        mu1w, mu2w, mu3w = calculate_mu(J_omega @ J_omega.T)
        mu1v, mu2v, mu3v = calculate_mu(J_v     @ J_v.T)

        return {
            'angular': {'mu1': mu1w, 'mu2': mu2w, 'mu3': mu3w},
            'linear':  {'mu1': mu1v, 'mu2': mu2v, 'mu3': mu3v},
        }

    # =========================================================================
    # Inverse Kinematics
    # =========================================================================

    def ik_body(self, T_sd, theta_guess, e_omega=0.001, e_v=0.0001, max_iter=50):
        theta = np.array(theta_guess, dtype=float)

        for _ in range(max_iter):
            T_sb = self.forward_kinematics_space(theta)
            T_bd = self.trans_inv(T_sb) @ T_sd
            V_b  = self.matrix_log_6(T_bd)

            if np.linalg.norm(V_b[:3]) < e_omega and np.linalg.norm(V_b[3:]) < e_v:
                return theta, True

            Js   = self.jacobian_space(theta)
            Jb   = self.jacobian_body(Js, T_sb)
            
            # np.linalg.pinv automatically handles the redundancy (7 DOF > 6 spatial dims)
            theta += np.linalg.pinv(Jb) @ V_b

        return theta, False