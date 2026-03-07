import unittest
import numpy as np
import sys
import os

# Ensure the module can be imported
sys.path.append("/home/franka/xyf_ws/realRL")

from serl_robot_infra.franka_env.gello_teleop.franka_fk import FrankaFK

class TestFrankaFK(unittest.TestCase):
    def setUp(self):
        try:
            self.fk = FrankaFK()
        except FileNotFoundError:
            self.skipTest("Franka FR3 model file not found")
        except Exception as e:
            self.skipTest(f"Failed to initialize FrankaFK: {e}")

    def test_initial_pose(self):
        # Test with home position (all zeros)
        joint_angles = np.zeros(7)
        pos, quat = self.fk.get_fk(joint_angles)
        
        print(f"\nZero joints pose: pos={pos}, quat={quat}")
        
        # Check output shapes
        self.assertEqual(pos.shape, (3,))
        self.assertEqual(quat.shape, (4,))
        
        # Basic sanity check (robot should be pointing up roughly)
        # In standard Franka home, z should be positive
        self.assertGreater(pos[2], 0.0)

    def test_known_pose(self):
        # Test with a known configuration if possible, 
        # or just verify deterministic output
        joint_angles = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
        pos1, quat1 = self.fk.get_fk(joint_angles)
        pos2, quat2 = self.fk.get_fk(joint_angles)
        
        np.testing.assert_array_almost_equal(pos1, pos2)
        np.testing.assert_array_almost_equal(quat1, quat2)
        print(f"\nTest pose: pos={pos1}, quat={quat1}")

if __name__ == '__main__':
    unittest.main()
