import numpy as np
from dm_control import mjcf
import os

class FrankaFK:
    def __init__(self, model_path=None):
        if model_path is None:
            # Use local path relative to this file or package root
            # This file is in serl_robot_infra/franka_env/gello_teleop/franka_fk.py
            # third_party is in realRL-main/third_party
            
            # Get path to realRL-main root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up 3 levels: gello_teleop -> franka_env -> serl_robot_infra -> realRL-main
            # root_dir = os.path.abspath(os.path.join(current_dir))
            
            model_path = os.path.join(
                current_dir, 
                "franka_fr3/fr3.xml"
            )
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Franka FR3 model not found at: {model_path}")
            
        self.model = mjcf.from_path(model_path)
        self.physics = mjcf.Physics.from_mjcf_model(self.model)
        
        # Attachment site name in the FR3 model (usually "attachment_site" or similar)
        # We need to verify this name from the XML or by inspecting the model
        self.attachment_site = "attachment_site"
        
    def get_fk(self, joint_angles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Forward Kinematics for Franka FR3.
        
        Args:
            joint_angles: 7-element numpy array of joint angles in radians.
            
        Returns:
            tuple: (position, quaternion)
                - position: 3-element numpy array [x, y, z]
                - quaternion: 4-element numpy array [w, x, y, z] (scalar-first) or [x, y, z, w]?
                  dm_control uses [w, x, y, z] (scalar-first)
        """
        if len(joint_angles) != 7:
            raise ValueError(f"Expected 7 joint angles, got {len(joint_angles)}")
            
        # Set joint positions (first 7 qpos are for the arm)
        # We need to ensure we are setting the correct joints.
        # The FR3 model usually has joints named "fr3_joint1", "fr3_joint2", etc.
        # Or we can just set physics.data.qpos[:7] if the arm is the first thing.
        
        self.physics.data.qpos[:7] = joint_angles
        self.physics.forward()
        
        # Get attachment site pose
        # site_xpos is position
        # site_xmat is rotation matrix (3x3)
        try:
            pos = self.physics.named.data.site_xpos[self.attachment_site].copy()
            rot_mat = self.physics.named.data.site_xmat[self.attachment_site].copy().reshape(3, 3)
        except KeyError:
             # Fallback if attachment_site is named differently, try "flange" or "ee"
             # But "attachment_site" is standard in menagerie models
             raise ValueError(f"Site '{self.attachment_site}' not found in model.")

        # Convert rotation matrix to quaternion
        # We can use scipy.spatial.transform.Rotation
        from scipy.spatial.transform import Rotation
        quat = Rotation.from_matrix(rot_mat).as_quat() # returns [x, y, z, w]
        
        return pos, quat
