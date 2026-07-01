import pytest
import numpy as np
from bashar.core.kinematics import BasharKinematics
from bashar.core.dynamics import BasharDynamics
from bashar.core.controller import RobotState

# --- FIXTURE: IN-MEMORY ROBOT PROFILE ---
@pytest.fixture
def mock_kinematics():
    profile = {
        "robot_name": "test_bot",
        "kinematic_tree": {
            "base_frame": "base_link",
            "joints": [
                {
                    "name": "j1", "type": "revolute", "parent": "base_link", "child": "link1",
                    "origin": {"xyz": [0, 0, 0.5], "rpy": [0, 0, 0]},
                    "axis": [0, 0, 1],
                    "limits": {"lower": -1.5, "upper": 1.5, "velocity": 1.0, "effort": 10.0},
                    "inertial": {
                        "mass": 2.0, 
                        "origin": {"xyz": [0, 0.1, 0], "rpy": [0, 0, 0]}, 
                        "inertia": {"ixx": 0.1, "ixy": 0, "ixz": 0, "iyy": 0.1, "iyz": 0, "izz": 0.1}
                    }
                }
            ]
        }
    }
    return BasharKinematics(profile), profile

# --- TEST SUITE 1: LIE ALGEBRA & KINEMATICS ---
def test_skew_symmetric(mock_kinematics):
    kin, _ = mock_kinematics
    v = np.array([1, 2, 3])
    skew = kin.skew_symmetric(v)
    
    # Skew-symmetric matrix transpose must equal its negative
    assert np.allclose(skew.T, -skew)
    assert skew[0, 1] == -3
    assert skew[0, 2] == 2

def test_forward_kinematics_identity(mock_kinematics):
    kin, _ = mock_kinematics
    # At zero angles, FK should exactly equal the Home Configuration (M)
    T_zero = kin.forward_kinematics_space([0.0])
    assert np.allclose(T_zero, kin.M)

def test_jacobian_shape(mock_kinematics):
    kin, _ = mock_kinematics
    Js = kin.jacobian_space([0.0])
    # Spatial Jacobian must always be 6 x N (where N is active joints)
    assert Js.shape == (6, kin.num_joints)

# --- TEST SUITE 2: DYNAMICS ---
def test_mass_matrix_properties(mock_kinematics):
    kin, _ = mock_kinematics
    dyn = BasharDynamics(kin)
    
    M = dyn.mass_matrix([0.0])
    
    # Mass matrix must be square (N x N)
    assert M.shape == (kin.num_joints, kin.num_joints)
    
    # Mass matrix must be symmetric (M = M^T)
    assert np.allclose(M, M.T)
    
    # Mass matrix must be positive definite (all eigenvalues > 0)
    eigenvalues = np.linalg.eigvals(M)
    assert np.all(eigenvalues > 0)

# --- TEST SUITE 3: CONTROLLER SAFETY ---
def test_robot_state_clipping(mock_kinematics):
    _, profile = mock_kinematics
    
    # Attempt to initialize state outside the [-1.5, 1.5] limits
    state = RobotState(profile, initial_positions=[2.0])
    
    # The state should automatically clip to 1.5
    assert state.positions[0] == 1.5
    
    # Apply a delta that pushes it further out
    state.apply_delta([1.0])
    
    # Must remain clamped at 1.5
    assert state.positions[0] == 1.5