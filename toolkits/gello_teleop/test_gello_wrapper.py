import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import sys
import os
import gymnasium as gym

# Set Mock environment variable
os.environ["MOCK_GELLO"] = "1"

# Ensure the module can be imported
sys.path.append("/home/franka/xyf_ws/realRL")

# Mock modules that might not be available
sys.modules["franka_env.spacemouse.spacemouse_expert"] = MagicMock()
sys.modules["franka_env.envs.franka_env"] = MagicMock()
sys.modules["serl_launcher.wrappers.progress"] = MagicMock()
sys.modules["serl_launcher.wrappers.force_reset"] = MagicMock()
sys.modules["experiments.config"] = MagicMock()

# Import the wrapper to test
from serl_robot_infra.franka_env.envs.wrappers import GelloIntervention

class MockEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(7,))
        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,)),
            "joint_positions": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(7,))
        })
        self.currpos = np.array([0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0]) # [x, y, z, qx, qy, qz, qw]
        self.action_scale = (0.05, 0.25, 1.0)
        
    def step(self, action):
        return {}, 0, False, False, {}
    
    def reset(self, **kwargs):
        return {}, {}
    
    def _get_obs(self):
        return {
            "joint_positions": np.zeros(7) # Robot at zero
        }

class TestGelloIntervention(unittest.TestCase):
    def setUp(self):
        self.env = MockEnv()
        
    @patch("serl_robot_infra.franka_env.gello_teleop.gello_teleop_agent.GelloTeleopAgent")
    @patch("serl_robot_infra.franka_env.gello_teleop.franka_fk.FrankaFK")
    def test_safety_check_pass(self, MockFK, MockAgent):
        # Setup mocks
        agent_instance = MockAgent.return_value
        # Robot is at 0 (from MockEnv._get_obs), Gello at 0.1
        agent_instance.get_action.return_value = (np.ones(7) * 0.1, 0.0)
        
        # Threshold is 0.8, diff is 0.1 -> should pass
        wrapper = GelloIntervention(self.env, start_joint_threshold=0.8)
        
    @patch("serl_robot_infra.franka_env.gello_teleop.gello_teleop_agent.GelloTeleopAgent")
    @patch("serl_robot_infra.franka_env.gello_teleop.franka_fk.FrankaFK")
    def test_safety_check_fail(self, MockFK, MockAgent):
        # Setup mocks
        agent_instance = MockAgent.return_value
        # Robot is at 0, Gello at 1.0
        agent_instance.get_action.return_value = (np.ones(7) * 1.0, 0.0)
        
        # Threshold is 0.8, diff is 1.0 -> should fail
        with self.assertRaises(RuntimeError) as cm:
            wrapper = GelloIntervention(self.env, start_joint_threshold=0.8)
        self.assertIn("Gello safety check failed", str(cm.exception))

    @patch("serl_robot_infra.franka_env.gello_teleop.gello_teleop_agent.GelloTeleopAgent")
    @patch("serl_robot_infra.franka_env.gello_teleop.franka_fk.FrankaFK")
    def test_action_computation(self, MockFK, MockAgent):
        # Setup mocks
        agent_instance = MockAgent.return_value
        fk_instance = MockFK.return_value
        
        # Gello joints
        agent_instance.get_action.return_value = (np.zeros(7), 1.0) # Open gripper (1.0)
        
        # Target Pose from FK (matches current pose, so delta should be 0)
        # Current pose: [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0] (identity quat)
        fk_instance.get_fk.return_value = (
            np.array([0.5, 0.0, 0.5]), 
            np.array([0.0, 0.0, 0.0, 1.0])
        )
        
        wrapper = GelloIntervention(self.env, start_joint_threshold=10.0)
        
        # Test action
        dummy_policy_action = np.zeros(7)
        action, intervened = wrapper.action(dummy_policy_action)
        
        self.assertTrue(intervened)
        # Delta position should be 0
        np.testing.assert_array_almost_equal(action[:3], np.zeros(3))
        # Gripper: Gello 1.0 -> Mapped to 1.0 (1.0*2 - 1 = 1.0)
        self.assertAlmostEqual(action[6], 1.0)

if __name__ == '__main__':
    unittest.main()
