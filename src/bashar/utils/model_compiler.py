import os
import json
import math
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

class ModelCompiler:
    """
    Crawls URDF/XACRO files to extract the pure kinematic skeleton, 
    validates the mathematics, and compiles a BASHAR-compatible JSON profile.
    """

    def __init__(self):

        self.supported_types = ['revolute', 'prismatic', 'continuous', 'fixed', 'planar', 'floating']


    def compile(self, model_path: str, output_name: str, output_dir: str, verbose: bool = False) -> bool:
        """Main pipeline execution handler for model compilation."""
        # Check if the user provided a valid path
        if not os.path.exists(model_path):
            print(f"[ERROR] Source model not found: {model_path}")
            return False

        print(f"--> Compiling kinematic profile for: {output_name}")
        
        # Step 1: Inflate XACRO macros or read URDF directly into a flat string
        xml_string = self._resolve_model_file(model_path)
        if not xml_string:
            return False

        # Step 2: Strip visual data, parse multi-DOF joints, and extract the pure kinematic tree
        tree_dict = self._extract_kinematics(xml_string, model_path)
        if not tree_dict:
            return False

        # Step 3: Validate mathematical boundaries and check for closed kinematic loops
        if not self._validate_tree(tree_dict):
            print("[ERROR] Mathematical validation failed. Aborting compilation.")
            return False

        # Step 4: Optional terminal visualization (great for debugging)
        if verbose:
            self._print_tree(tree_dict)

        # Step 5: Commit the certified mathematical skeleton to disk as JSON
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{output_name}.json")
        
        with open(output_path, 'w') as f:
            json.dump(tree_dict, f, indent=2)
            
        print(f"[SUCCESS] Profile compiled and saved to: {output_path}")
        return True
    
    def _resolve_model_file(self, file_path: str) -> str:
        """Parses URDF or evaluates XACRO macros into a flat XML string."""
        if file_path.endswith('.xacro'):
            try:
                # Calls the system 'xacro' command to expand all macros, properties, and includes
                result = subprocess.run(
                    ['xacro', file_path], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE, 
                    text=True, 
                    check=True
                )
                return result.stdout
            except subprocess.CalledProcessError as e:
                # The xacro file has a syntax error or a missing include
                print(f"[ERROR] XACRO evaluation failed:\n{e.stderr}")
                return ""
            except FileNotFoundError:
                # The script can't find the xacro command line tool
                print("[ERROR] 'xacro' command not found. Ensure ROS 2 is sourced in your terminal.")
                return ""
        else:
            # If it's a .urdf, we just read the raw text
            try:
                with open(file_path, 'r') as f:
                    return f.read()
            except Exception as e:
                print(f"[ERROR] Failed to read URDF file: {e}")
                return ""
            

    def _extract_kinematics(self, xml_string: str, source_path: str) -> dict:
        """Strips visual/collision data, decomposes multi-DOF joints, and extracts the kinematic tree."""
        try:
            root = ET.fromstring(xml_string)
        except ET.ParseError as e:
            print(f"[ERROR] XML Parsing failed: {e}")
            return {}

        robot_name = root.attrib.get('name', 'unknown_robot')
        
        # PRE-FETCH ALL LINK MASSES AND INERTIAS
        links_dict = self._extract_links(root)
        joints = []

        for joint in root.findall('joint'):
            j_type = joint.attrib.get('type', 'fixed')
            if j_type not in self.supported_types:
                continue

            j_name = joint.attrib.get('name')
            parent = joint.find('parent').attrib.get('link') if joint.find('parent') is not None else None
            child = joint.find('child').attrib.get('link') if joint.find('child') is not None else None

            if not parent or not child:
                continue

            # Default Origin
            origin_xyz, origin_rpy = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
            origin_node = joint.find('origin')
            if origin_node is not None:
                if 'xyz' in origin_node.attrib: origin_xyz = [float(x) for x in origin_node.attrib['xyz'].split()]
                if 'rpy' in origin_node.attrib: origin_rpy = [float(x) for x in origin_node.attrib['rpy'].split()]

            # Default Axis
            axis_xyz = [1.0, 0.0, 0.0]
            axis_node = joint.find('axis')
            if axis_node is not None and 'xyz' in axis_node.attrib: axis_xyz = [float(x) for x in axis_node.attrib['xyz'].split()]

            # Default Limits
            limits = {"lower": 0.0, "upper": 0.0, "velocity": 0.0, "effort": 0.0}
            limit_node = joint.find('limit')
            if limit_node is not None:
                limits["lower"] = float(limit_node.attrib.get('lower', 0.0))
                limits["upper"] = float(limit_node.attrib.get('upper', 0.0))
                limits["velocity"] = float(limit_node.attrib.get('velocity', 0.0))
                limits["effort"] = float(limit_node.attrib.get('effort', 0.0))

            # Fetch the actual physical weight/inertia of the limb connected to this joint
            child_inertial = links_dict.get(child, self._default_inertial())

            # Virtual Decomposition Routing
            if j_type == 'planar':
                joints.extend(self._decompose_planar(j_name, parent, child, origin_xyz, origin_rpy, child_inertial))
            elif j_type == 'floating':
                joints.extend(self._decompose_floating(j_name, parent, child, origin_xyz, origin_rpy, child_inertial))
            else:
                joints.append({
                    "name": j_name,
                    "type": j_type,
                    "parent": parent,
                    "child": child,
                    "origin": {"xyz": origin_xyz, "rpy": origin_rpy},
                    "axis": axis_xyz,
                    "limits": limits,
                    "inertial": child_inertial  # <--- INJECTED HERE
                })

        # Dynamically find the root base link
        child_links = {j['child'] for j in joints}
        parent_links = {j['parent'] for j in joints}
        base_frames = list(parent_links - child_links)
        base_frame = base_frames[0] if base_frames else "base_link"

        return {
            "robot_name": robot_name,
            "meta": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source_directory": os.path.abspath(source_path)
            },
            "kinematic_tree": {
                "base_frame": base_frame,
                "joints": joints
            }
        }

    def _create_virtual_joint(self, name, j_type, parent, child, xyz, rpy, axis, inertial):
        """Helper to generate a clean 1-DOF joint dictionary for decomposition."""
        return {
            "name": name,
            "type": j_type,
            "parent": parent,
            "child": child,
            "origin": {"xyz": xyz, "rpy": rpy},
            "axis": axis,
            "limits": {"lower": -100.0, "upper": 100.0, "velocity": 10.0, "effort": 100.0},
            "inertial": inertial
        }

    def _decompose_planar(self, name, parent, child, origin_xyz, origin_rpy, final_inertial):
        """Decomposes a 3-DOF planar joint. The intermediate links weigh 0kg. The final link gets the real mass."""
        v1, v2 = f"{name}_v_x", f"{name}_v_y"
        return [
            self._create_virtual_joint(f"{name}_x", "prismatic", parent, v1, origin_xyz, origin_rpy, [1.0, 0.0, 0.0], self._default_inertial()),
            self._create_virtual_joint(f"{name}_y", "prismatic", v1, v2, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], self._default_inertial()),
            self._create_virtual_joint(f"{name}_theta", "continuous", v2, child, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], final_inertial)
        ]

    def _decompose_floating(self, name, parent, child, origin_xyz, origin_rpy, final_inertial):
        """Decomposes a 6-DOF floating joint. The final rotational link gets the real mass."""
        v1, v2, v3, v4, v5 = [f"{name}_v_{i}" for i in range(1, 6)]
        return [
            self._create_virtual_joint(f"{name}_x", "prismatic", parent, v1, origin_xyz, origin_rpy, [1.0, 0.0, 0.0], self._default_inertial()),
            self._create_virtual_joint(f"{name}_y", "prismatic", v1, v2, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], self._default_inertial()),
            self._create_virtual_joint(f"{name}_z", "prismatic", v2, v3, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], self._default_inertial()),
            self._create_virtual_joint(f"{name}_roll", "continuous", v3, v4, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], self._default_inertial()),
            self._create_virtual_joint(f"{name}_pitch", "continuous", v4, v5, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0], self._default_inertial()),
            self._create_virtual_joint(f"{name}_yaw", "continuous", v5, child, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], final_inertial)
        ]
    def _validate_tree(self, tree: dict) -> bool:
        """Executes mathematical and logic checks on the extracted joints."""
        joints = tree['kinematic_tree']['joints']
        
        # 1. Axis Normalization Check
        for j in joints:
            if j['type'] in ['revolute', 'continuous', 'prismatic']:
                ax = j['axis']
                magnitude = math.sqrt(sum(x**2 for x in ax))
                if magnitude == 0:
                    print(f"[WARN] Joint '{j['name']}' has a zero-vector axis. Defaulting to [1, 0, 0].")
                    j['axis'] = [1.0, 0.0, 0.0]
                elif not math.isclose(magnitude, 1.0, rel_tol=1e-5):
                    # Normalize the vector automatically so the downstream math doesn't break
                    j['axis'] = [x / magnitude for x in ax]

        # 2. Limit Sanity Check
        for j in joints:
            if j['type'] in ['revolute', 'prismatic']:
                lims = j['limits']
                if lims['lower'] > lims['upper']:
                    print(f"[ERROR] Joint '{j['name']}' has inverted limits (lower > upper).")
                    return False
                if lims['velocity'] < 0 or lims['effort'] < 0:
                    print(f"[ERROR] Joint '{j['name']}' has negative safety limits (velocity/effort).")
                    return False

        # 3. Closed-Loop Detection (DAG check)
        # PoE requires an open kinematic chain. If the user accidentally looped a child back to a parent, block it.
        graph = {}
        for j in joints:
            if j['parent'] not in graph:
                graph[j['parent']] = []
            graph[j['parent']].append(j['child'])

        visited = set()
        rec_stack = set()

        def is_cyclic(node):
            visited.add(node)
            rec_stack.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if is_cyclic(neighbor): return True
                elif neighbor in rec_stack:
                    print(f"[ERROR] Closed kinematic loop detected at link: {neighbor}")
                    return True
            rec_stack.remove(node)
            return False

        base = tree['kinematic_tree']['base_frame']
        if is_cyclic(base):
            return False

        return True

    def _print_tree(self, tree: dict):
        """Displays an ASCII representation of the compiled kinematic tree for debugging."""
        print(f"\n--- BASHAR KINEMATIC TREE: {tree['robot_name']} ---")
        print(f"[BASE] {tree['kinematic_tree']['base_frame']}")
        
        joints = tree['kinematic_tree']['joints']
        graph = {}
        for j in joints:
            if j['parent'] not in graph:
                graph[j['parent']] = []
            graph[j['parent']].append(j)

        def print_children(parent_link, indent="  "):
            for j in graph.get(parent_link, []):
                axis_str = f" axis={j['axis']}" if j['type'] != 'fixed' else ""
                print(f"{indent}└─ [{j['type'].upper()}] {j['name']} -> {j['child']}{axis_str}")
                print_children(j['child'], indent + "    ")

        print_children(tree['kinematic_tree']['base_frame'])
        print("----------------------------------------\n")

    def _default_inertial(self):
        """Returns a zero-mass inertial dictionary for virtual or empty links."""
        return {
            "mass": 0.0,
            "origin": {"xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
            "inertia": {"ixx": 0.0, "ixy": 0.0, "ixz": 0.0, "iyy": 0.0, "iyz": 0.0, "izz": 0.0}
        }

    def _extract_links(self, root: ET.Element) -> dict:
        """Crawls all <link> tags and extracts their physical mass and inertia tensors."""
        links = {}
        for link in root.findall('link'):
            name = link.attrib.get('name')
            if not name: continue
            
            inertial_node = link.find('inertial')
            if inertial_node is not None:
                mass_node = inertial_node.find('mass')
                mass = float(mass_node.attrib.get('value', 0.0)) if mass_node is not None else 0.0
                
                origin_node = inertial_node.find('origin')
                xyz, rpy = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
                if origin_node is not None:
                    if 'xyz' in origin_node.attrib: xyz = [float(x) for x in origin_node.attrib['xyz'].split()]
                    if 'rpy' in origin_node.attrib: rpy = [float(x) for x in origin_node.attrib['rpy'].split()]
                
                inertia_node = inertial_node.find('inertia')
                inertia = {"ixx": 0.0, "ixy": 0.0, "ixz": 0.0, "iyy": 0.0, "iyz": 0.0, "izz": 0.0}
                if inertia_node is not None:
                    for key in inertia.keys():
                        inertia[key] = float(inertia_node.attrib.get(key, 0.0))
                        
                links[name] = {"mass": mass, "origin": {"xyz": xyz, "rpy": rpy}, "inertia": inertia}
            else:
                links[name] = self._default_inertial()
        return links