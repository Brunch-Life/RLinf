import numpy as np
import time
import sys
import os
from typing import Optional, Tuple, Dict

try:
    from gello.agents.gello_agent import GelloAgent, DynamixelRobotConfig, PORT_CONFIG_MAP

    _GELLO_AVAILABLE = True
except ImportError:
    GelloAgent = None
    DynamixelRobotConfig = None
    PORT_CONFIG_MAP = {}
    _GELLO_AVAILABLE = False

class GelloTeleopAgent:
    def __init__(self, port: str = None, joint_ids: Optional[Tuple[int]] = None):
        """
        Initialize GelloTeleopAgent.
        
        Args:
            port: Serial port for Dynamixel. If None, tries to find FR3 config in PORT_CONFIG_MAP.
            joint_ids: Optional override for joint IDs.
        """
        self.agent = None
        
        # Default FR3 port from gello config if not provided
        # "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAJEDPC-if00-port0"
        if port is None:
            # Try to find a port that looks like the FR3 one or use the first available
            for p, config in PORT_CONFIG_MAP.items():
                if len(config.joint_ids) == 7: # FR3 has 7 joints
                    port = p
                    break
        
        if port is None:
             raise ValueError("No suitable gello port found. Please specify port.")
             
        self.port = port
        print(f"Initializing GelloTeleopAgent on port: {self.port}")
        
        # Initialize GelloAgent
        try:
            if not _GELLO_AVAILABLE and not os.environ.get("MOCK_GELLO"):
                raise ImportError(
                    "GelloTeleopAgent requires the 'gello' package. "
                    "Install it (e.g. pip install gello) or set MOCK_GELLO=1 for testing."
                )
            # Check if port exists
            if not os.path.exists(self.port) and not os.environ.get("MOCK_GELLO"):
                 raise FileNotFoundError(f"Port {self.port} not found.")
            # Use gello's GelloAgent
            self.agent = GelloAgent(port=self.port)
            
            # Verify we have 7 joints for FR3
            # GelloAgent.act returns joint state
            # We can't easily inspect the internal robot object without private access
            # But we can try to read once
            pass
        except Exception as e:
            if os.environ.get("MOCK_GELLO"):
                print("Using Mock Gello Agent")
                self.agent = MockGelloAgent()
            else:
                raise e

    def get_action(self) -> Tuple[np.ndarray, float]:
        """
        Get current joint angles and gripper state from Gello.
        
        Returns:
            tuple: (joint_angles, gripper_state)
                - joint_angles: 7-element numpy array (radians)
                - gripper_state: float (0.0 to 1.0, 0=closed, 1=open)? 
                  Or mapped to [-1, 1]?
                  Let's check gello implementation.
        """
        # GelloAgent.act returns dict with "joint_state" or just array depending on implementation
        # Looking at gello_agent.py: act returns self._robot.get_joint_state()
        # DynamixelRobot.get_joint_state() returns numpy array of positions.
        # If gripper is configured, it's the last element.
        
        obs = {} # Dummy obs
        if self.agent is None:
             return np.zeros(7), 0.0
             
        state = self.agent.act(obs)
        
        # Expecting 8 elements for FR3 (7 arm + 1 gripper) if gripper configured
        if len(state) == 8:
            joint_angles = state[:7]
            gripper_val = state[7]
        elif len(state) == 7:
            joint_angles = state
            gripper_val = 0.0 # Default
        else:
            raise ValueError(f"Unexpected state length from gello: {len(state)}")
            
        return joint_angles, gripper_val

class MockGelloAgent:
    def __init__(self):
        self.num_joints = 8 # 7 arm + 1 gripper
        
    def act(self, obs):
        return np.zeros(self.num_joints)
