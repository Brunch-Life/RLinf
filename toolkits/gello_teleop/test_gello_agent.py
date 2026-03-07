import unittest
import numpy as np
import os
import sys

# Set Mock environment variable
os.environ["MOCK_GELLO"] = "1"

# Ensure the module can be imported
sys.path.append("/home/franka/xyf_ws/realRL")

from serl_robot_infra.franka_env.gello_teleop.gello_teleop_agent import GelloTeleopAgent

class TestGelloTeleopAgent(unittest.TestCase):
    def setUp(self):
        # Provide a dummy port, the Mock agent will ignore it
        self.agent = GelloTeleopAgent(port="/dev/null")

    def test_get_action(self):
        joint_angles, gripper_val = self.agent.get_action()
        
        print(f"\nMock Action: joints={joint_angles}, gripper={gripper_val}")
        
        self.assertEqual(joint_angles.shape, (7,))
        self.assertIsInstance(gripper_val, float)
        # Mock returns zeros
        np.testing.assert_array_equal(joint_angles, np.zeros(7))

if __name__ == '__main__':
    unittest.main()
