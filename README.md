# BASHAR
**Bimodal Autonomous System for Handling and Articulated Robotics**

A hardware-agnostic Python library for articulated robot control. You give it a URDF/XACRO file, it handles the kinematics, dynamics, and safety. The goal is that swapping from one robot to another only requires dropping in a new config file — no changes to the math or the control logic.

---

## Background

This started as a port of the control stack from the SEAS (Smart Endoscopy) prototype. The original code was hardcoded for a 7-DOF bronchoscope — specific joint names, specific link lengths, specific hardware limits. The idea here was to pull the core math (Product of Exponentials kinematics, Newton-Euler dynamics, APF collision avoidance) out of that context and wrap it in a proper library so the same "brain stem" can drive any serial-chain robot.

The terminology we settled on: BASHAR is the **Spinal Cord**. It does the physics. Whatever makes the decisions — an ML model, a vision script, a person at a keyboard — is the **Brain**. BASHAR doesn't know what a camera is and doesn't need to.

---

## What's been built so far

### Directory layout

```
BASHAR/
├── config/
│   └── profiles/              # Where compiled robot JSON files go
├── src/
│   └── bashar/
│       ├── api.py             # The only file users import
│       ├── core/
│       │   ├── kinematics.py  # PoE kinematics engine
│       │   ├── dynamics.py    # Newton-Euler dynamics
│       │   └── controller.py  # State management, collision guard, autopilot
│       └── utils/
│           ├── model_compiler.py  # URDF/XACRO → JSON compiler
│           ├── config_parser.py   # JSON loader and validator
│           └── logger.py
├── tests/
├── test_integration.py
├── setup.py
└── package.xml
```

---

### `utils/model_compiler.py` — Robot Profile Compiler

Takes a `.urdf` or `.xacro` file path and outputs a validated JSON profile in `config/profiles/`.

What it does:
- For XACRO files, it calls the system `xacro` command via subprocess to flatten everything first, so multi-file xacro setups (like the BASHAR prototype's split arm/lift files) work out of the box without needing to pre-compile
- Walks the joint tree and extracts: joint name, type, parent/child links, origin (xyz + rpy), axis, and limits
- Handles `planar` and `floating` joints by decomposing them into virtual 1-DOF chains — this keeps the downstream PoE math clean since it only handles one degree of freedom at a time
- Runs three validation checks before writing anything: axis normalization (catches things like `<axis xyz="0 0 2"/>`), limit sanity (lower < upper, no negative velocity/effort), and closed-loop detection (PoE breaks on parallel linkages)
- Optional `verbose` flag prints an ASCII tree of the joint hierarchy to the terminal

```python
from bashar.api import compile_profile

compile_profile("my_robot.urdf", "my_robot", verbose=True)
# → config/profiles/my_robot.json
```

---

### `core/kinematics.py` — PoE Kinematics Engine

This is where the math lives. It reads the compiled JSON and builds the Product of Exponentials representation on the fly.

The key part is `_build_poe_from_profile()`. URDF defines joints relative to their parent (local offsets). PoE needs everything in the global base frame. So this method walks the chain, accumulates the global transform as it goes, and computes the correct space-frame screw axis `S = [ω, v]` for each active joint. Fixed joints get absorbed silently — they contribute to the home configuration `M` but don't add a DOF to the state vector.

What's implemented:
- Forward kinematics (space frame): `T(θ) = e^[S₁]θ₁ · ... · e^[Sₙ]θₙ · M`
- Space and body Jacobians
- Manipulability ellipsoid analysis
- Numerical IK via Newton-Raphson on the body Jacobian (handles redundant manipulators with pseudo-inverse)
- All the rigid-body math helpers: `skew_symmetric`, `matrix_exp_6`, `matrix_log_6`, `adjoint_matrix`, `trans_inv`

---

### `core/dynamics.py` — Newton-Euler Dynamics

Implements the Recursive Newton-Euler Algorithm (RNEA) for computing joint torques given a motion command.

- Forward pass propagates velocities and accelerations link-by-link
- Backward pass accumulates wrenches back to get torques
- Built on top of this: `mass_matrix(θ)`, `velocity_quadratic_forces(θ, θ̇)` (Coriolis + centripetal), `gravity_forces(θ)`, and `forward_dynamics(θ, θ̇, τ)`

The inertial pipeline is fully connected end-to-end. The model compiler's `_extract_links()` method parses every `<link>` tag in the URDF and extracts the `mass`, `origin` (COM offset), and the full 6-element inertia tensor (`ixx`, `ixy`, `ixz`, `iyy`, `iyz`, `izz`). That data gets stored in each joint's `inertial` block in the JSON profile. When the kinematics engine loads the profile, `_build_poe_from_profile()` reads those values and builds the correct 6×6 spatial inertia matrix `G` and the COM transform `M_i` for every active joint. The torque outputs from the dynamics engine reflect the actual masses and inertia tensors defined in your URDF.

---

### `core/controller.py` — State, Safety, and Navigation

Three classes:

**`RobotState`** — manages the joint position array. Reads hardware limits directly from the JSON profile so it auto-clips to whatever the robot's actual range of motion is. Works for any N-DOF configuration.

**`CollisionGuard`** — manual mode safety layer. The Brain passes a `desired_dtheta` and a list of obstacle coordinates (in the robot's base frame). The guard computes a repulsive wrench using Artificial Potential Fields, maps it back to joint space, and adds it to the command before applying hardware limits.

A few things worth noting:
- Obstacles are inflated by a `body_radius` parameter — so the system accounts for the physical thickness of the arm, not just the center-point of the end-effector
- The input sanitizer handles malformed obstacle arrays: a single `[x, y, z]` gets wrapped, a 2D `[x, y]` gets padded with the current Z height (treating it as an infinite vertical cylinder), and anything unparseable gets logged and skipped without crashing the loop
- Jacobian inversion uses **Damped Least Squares (DLS)** (`J⁺ = Jᵀ(JJᵀ + λ²I)⁻¹`) instead of plain pseudoinverse. This keeps the mapping stable when the arm is near a singularity. The damping factor `λ` defaults to 0.05 and is tunable on the constructor.

**`AutoPilot`** — autonomous mode. Brain gives a target `[x, y, z]` and an obstacle list. The autopilot computes an attractive velocity toward the goal, maps it to joint space via DLS, then passes it through the `CollisionGuard`. The guard handles obstacle avoidance automatically — the autopilot doesn't need to know about the specifics.

**`ComputedTorqueController`** — implements the full feedforward + feedback control law:
```
τ = M(θ) · [θ̈_des + Kp·e + Kd·ė] + c(θ, θ̇) + g(θ)
```
Predicts the physics to cancel it out, then applies a PD correction on top.

---

### `api.py` — The Public Interface

The only file a user should ever import.

```python
from bashar.api import compile_profile, BasharSystem, Trajectory

# Step 1: compile once
compile_profile("robot.urdf", "my_robot", verbose=True)

# Step 2: boot
robot = BasharSystem("config/profiles/my_robot.json")

# Step 3: use
robot.update_state(encoder_readings)
tip = robot.get_tip_position()

safe_joints = robot.manual_step(dtheta, obstacles=[...])
new_joints, done = robot.auto_step(target_xyz=[0.4, 0.0, 0.3], obstacles=[...])
torques = robot.calculate_motor_torques(current_dtheta, desired_theta, desired_dtheta)

# Trajectory planning
path = Trajectory.joint_trajectory(start, end, Tf=3.0, N=100, method='quintic')
path_with_vel = Trajectory.joint_trajectory_velocities(start, end, Tf=3.0, N=100)
```

`update_state()` validates the length of the input against the robot's actual DOF count and raises a clear error if they don't match — we fixed a bug here where a silent `IndexError` was bubbling up from inside `clip()`.

---

### `core/trajectory.py` — Trajectory Generation

Time-scaling functions and path generators for joint-space and task-space motion.

Three time-scaling profiles:
- **Cubic** — 3rd-order polynomial. Zero velocity at start/end.
- **Quintic** — 5th-order polynomial. Zero velocity *and* zero acceleration at start/end. Better for payload safety and torque limits.
- **Trapezoidal** — Bang-coast-bang velocity profile. Fastest average speed, useful when you want maximum throughput without caring about smoothness.

Path generation:
- `joint_trajectory(start, end, Tf, N)` → list of N joint configs
- `joint_trajectory_velocities(start, end, Tf, N)` → list of (θ, θ̇) tuples (needed for computed torque control)
- `via_point_trajectory(points, times, N_per_segment)` → concatenated path through multiple waypoints
- `screw_trajectory(T_start, T_end, Tf, N)` → list of N SE(3) transforms along the shortest task-space path

---

### Integration test

`test_integration.py` runs through the full pipeline end-to-end using the bronchoscope URDF:
1. Compiles the profile
2. Boots the system
3. Forward kinematics check
4. Computed torque calculation
5. Collision guard filtering

All passing. Run it with:

```bash
source .venv/bin/activate
python3 test_integration.py
```

---

### `examples/`

Three scripts in the `examples/` directory:

- **`basic_usage.py`** — standalone walkthrough of the full pipeline. Start here. Covers profile compilation, FK, trajectory planning, torque control, manual mode, and autonomous stepping.
- **`ros2_node.py`** — ROS 2 node template. Subscribes to `/joint_states`, publishes to `/joint_trajectory_controller/joint_trajectory`, and exposes `set_target()` / `update_obstacles()` for the Brain to call.
- **`hardware_loop.py`** — bare-metal control loop. No ROS dependency. Shows the encode→update→control→write pattern with stubbed hardware functions you replace with your actual driver calls.

---

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Requires Python 3.10+. Only dependency is `numpy`. XACRO support requires ROS 2 to be sourced in the terminal (uses the `xacro` command); plain URDF files work without ROS.

---

## What's still missing

The main remaining gap is tests:

**Unit tests** — `tests/` is empty. Need math validation tests for at minimum: forward kinematics round-trip, Jacobian finite-difference check against numerical differentiation, DLS stability check at a known singular config, and joint limit clamping.

**`example_arm.json`** — A hand-annotated template profile so someone can use the library without a URDF file (useful for simple test setups or quick prototyping). This would let a teammate try the examples without having a URDF ready.

---

## Notes for the team

- The obstacle coordinate frame requirement is strict: **everything must be in the robot's base frame**. If the Brain is working in camera frame or world frame, the transformation has to happen before the call to `manual_step` or `auto_step`. This is a Brain responsibility, not BASHAR's.
- The `body_radius` parameter in `CollisionGuard` defaults to 8cm. Adjust this to match the physical arm thickness of whatever robot you're testing with.
- The `.venv/` directory is gitignored. Each person needs to run `pip install -e .` after cloning.
- Generated profiles in `config/profiles/` are also gitignored — they get compiled from the URDF at runtime.
