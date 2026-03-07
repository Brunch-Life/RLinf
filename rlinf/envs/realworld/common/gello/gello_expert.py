# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading

import numpy as np


class GelloExpert:
    # """
    # This class provides an interface to the SpaceMouse.
    # It continuously reads the SpaceMouse state and provide
    # a "get_action" method to get the latest action and button state.
    # """ TODO

    def __init__(self, port):
        from toolkits.gello_teleop.gello_teleop_agent import GelloTeleopAgent
        from toolkits.gello_teleop.franka_fk import FrankaFK
        self.agent = GelloTeleopAgent(port=port)
        self.fk = FrankaFK()


        self.state_lock = threading.Lock()
        self.latest_data = {"target_pos": np.zeros(3),"target_quat": np.zeros(4), "gripper": np.zeros(1)}
        # Start a thread to continuously read the SpaceMouse state
        self.thread = threading.Thread(target=self._read_gello)
        self.thread.daemon = True
        self.thread.start()

    def _read_gello(self):
        # import pyspacemouse

        while True:
            # state = pyspacemouse.read()
            gello_joints, gello_gripper = self.agent.get_action()

            gello_gripper = np.array([gello_gripper])

            target_pos, target_quat = self.fk.get_fk(gello_joints)


            
            with self.state_lock:
                self.latest_data["target_pos"] = target_pos
                self.latest_data["target_quat"] = target_quat
                self.latest_data["gripper"] = gello_gripper



    def get_action(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the latest action and button state of the SpaceMouse."""
        with self.state_lock:
            return self.latest_data["target_pos"], self.latest_data["target_quat"], self.latest_data["gripper"]


if __name__ == "__main__":
    import time

    def test_gello():
        # """Test the SpaceMouseExpert class.

        # This interactive test prints the action and buttons of the spacemouse at a rate of 10Hz.
        # The user is expected to move the spacemouse and press its buttons while the test is running.
        # It keeps running until the user stops it.

        # """
        # TODO
        port = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0"
        # port = None
        gello = GelloExpert(port=port)
        with np.printoptions(precision=3, suppress=True):
            while True:
                target_pos, target_quat, gripper = gello.get_action()
                # print(f"Gello target_pos: {target_pos}, Gello target_quat: {target_quat}, Gello gripper: {gripper}")
                time.sleep(0.1)

    test_gello()
