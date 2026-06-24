import numpy as np
from bashar.core.kinematics import BasharKinematics

class BasharDynamics:
    """
    Generalized Dynamics Engine using Recursive Newton-Euler formulations.
    Calculates forces, torques, and accelerations for an N-DOF kinematic chain.
    """
    def __init__(self, kin: BasharKinematics):
        self.kin = kin

    def ad(self, V):
        """
        Calculates the 6x6 adjoint representation (Lie Bracket) of a spatial velocity V.
        Used to calculate Coriolis and centripetal forces.
        """
        ad_matrix = np.zeros((6, 6))
        omega_skew = self.kin.skew_symmetric(V[:3])
        v_skew = self.kin.skew_symmetric(V[3:])
        
        ad_matrix[:3, :3] = omega_skew
        ad_matrix[3:, :3] = v_skew
        ad_matrix[3:, 3:] = omega_skew
        return ad_matrix

    def inverse_dynamics(self, theta, dtheta, ddtheta, Ftip=None):
        """
        Recursive Newton-Euler Algorithm (RNEA).
        Calculates the joint torques (tau) required to achieve a specific acceleration.
        """
        n = self.kin.num_joints
        if Ftip is None:
            Ftip = np.zeros(6)

        V = np.zeros((6, n + 1))
        Vdot = np.zeros((6, n + 1))
        
        # Base frame acceleration simulates gravity pulling the entire system down
        Vdot[3:, 0] = -np.array(self.kin.gravity)

        A = np.zeros((6, n))
        M_rel = []

        # --- 1. PRE-COMPUTE PASS ---
        T_accum = np.eye(4)
        for i in range(n):
            T_home = self.kin.Mlist[i] 
            
            # The relative home transform M_{i-1, i}
            M_rel.append(self.kin.trans_inv(T_accum) @ T_home)
            T_accum = T_home
            
            # Extract the body-frame screw axis
            A[:, i] = self.kin.adjoint_matrix(self.kin.trans_inv(T_home)) @ self.kin.S_list[i]

        # --- 2. FORWARD PASS (Velocity & Acceleration) ---
        Ad_T_inv_list = []  # We cache these to drastically speed up the backward pass
        
        for i in range(1, n + 1):
            # CORRECT MATH: T_{i, i-1} = exp(-A * theta) * M_{i-1, i}^{-1}
            T_i_to_i_minus_1 = self.kin.matrix_exp_6(A[:, i - 1], -theta[i - 1]) @ self.kin.trans_inv(M_rel[i - 1])
            Ad_T_inv = self.kin.adjoint_matrix(T_i_to_i_minus_1)
            Ad_T_inv_list.append(Ad_T_inv)
            
            V[:, i] = Ad_T_inv @ V[:, i - 1] + A[:, i - 1] * dtheta[i - 1]
            
            ad_V_A = self.ad(V[:, i]) @ A[:, i - 1]
            Vdot[:, i] = Ad_T_inv @ Vdot[:, i - 1] + A[:, i - 1] * ddtheta[i - 1] + ad_V_A * dtheta[i - 1]

        # --- 3. BACKWARD PASS (Forces & Torques) ---
        tau = np.zeros(n)
        Wrenches = np.zeros((6, n + 1))
        Wrenches[:, n] = Ftip

        for i in range(n, 0, -1):
            G_i = self.kin.Glist[i - 1]
            
            F_inertial = G_i @ Vdot[:, i] - self.ad(V[:, i]).T @ (G_i @ V[:, i])
            
            if i < n:
                # Pull the wrench backwards from the child link (i+1) using the cached Adjoint transpose
                Wrenches[:, i] = F_inertial + Ad_T_inv_list[i].T @ Wrenches[:, i + 1]
            else:
                Wrenches[:, i] = F_inertial + Ftip
                
            tau[i - 1] = Wrenches[:, i].T @ A[:, i - 1]

        return tau

    def mass_matrix(self, theta):
        """Calculates the Mass/Inertia matrix M(theta)."""
        n = self.kin.num_joints
        M = np.zeros((n, n))
        for i in range(n):
            ddtheta = np.zeros(n)
            ddtheta[i] = 1.0
            M[:, i] = self.inverse_dynamics(theta, np.zeros(n), ddtheta)
        return M

    def velocity_quadratic_forces(self, theta, dtheta):
        """Calculates the Coriolis and centripetal forces c(theta, dtheta)."""
        g_actual = np.copy(self.kin.gravity)
        self.kin.gravity = np.zeros(3)
        c = self.inverse_dynamics(theta, dtheta, np.zeros(self.kin.num_joints))
        self.kin.gravity = g_actual
        return c

    def gravity_forces(self, theta):
        """Calculates the forces required purely to hold the robot against gravity g(theta)."""
        n = self.kin.num_joints
        return self.inverse_dynamics(theta, np.zeros(n), np.zeros(n))

    def forward_dynamics(self, theta, dtheta, tau, Ftip=None):
        """
        Calculates the resulting joint accelerations (ddtheta) given the applied torques (tau).
        """
        M = self.mass_matrix(theta)
        c = self.velocity_quadratic_forces(theta, dtheta)
        g = self.gravity_forces(theta)
        
        tau_ext = np.zeros(self.kin.num_joints)
        if Ftip is not None:
            Jb = self.kin.jacobian_body(self.kin.jacobian_space(theta), self.kin.forward_kinematics_space(theta))
            tau_ext = Jb.T @ Ftip
            
        # M * ddtheta + c + g = tau - J^T * Ftip  =>  ddtheta = M^-1 * (tau - c - g - tau_ext)
        rhs = tau - c - g - tau_ext
        ddtheta = np.linalg.inv(M) @ rhs
        return ddtheta