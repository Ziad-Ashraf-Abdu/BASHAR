import json
import os
from bashar.utils.logger import get_bashar_logger

logger = get_bashar_logger()

class ProfileParser:
    """
    Safely loads and validates the compiled JSON kinematics profile.
    Acts as the gatekeeper before the math engines boot up.
    """
    @staticmethod
    def load(profile_path: str) -> dict:
        if not os.path.exists(profile_path):
            logger.error(f"Could not find robot profile at {profile_path}")
            raise FileNotFoundError(f"Missing profile: {profile_path}")
        
        with open(profile_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"JSON format is corrupted in {profile_path}:\n{e}")
                raise ValueError("Corrupted JSON Profile.")
        
        # Security Checks: Ensure the core mathematical structures exist
        if 'kinematic_tree' not in data:
            raise KeyError("Profile is missing 'kinematic_tree' block.")
            
        if 'joints' not in data['kinematic_tree']:
            raise KeyError("Profile is missing 'joints' list.")
            
        if len(data['kinematic_tree']['joints']) == 0:
            logger.warning("The loaded profile has 0 active joints. Is this intended?")
            
        logger.info(f"Successfully verified profile: {data.get('robot_name', 'Unknown')}")
        return data