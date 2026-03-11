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


import os
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.scheduler import Cluster
from rlinf.envs.realworld.franka.franka_controller import FrankaController


def main():
    robot_ip = os.environ.get("FRANKA_ROBOT_IP", "172.16.0.2")
    assert robot_ip is not None, "Please set the FRANKA_ROBOT_IP environment variable."
    cluster_cfg = {
        "num_nodes": 1,
        "component_placement": {"actor, rollout, env": {"node_group": "franka", "placement": 0}},
        "node_groups": [{"label": "franka", "node_ranks": 0, "hardware": {"type": "Franka", "configs": [{"robot_ip": robot_ip, "node_rank": 0}]}}]
    }
    cluster = Cluster(cluster_cfg=cluster_cfg)
    controller = FrankaController.launch_controller(robot_ip=robot_ip)

    start_time = time.time()
    while not controller.is_robot_up().wait()[0]:
        time.sleep(0.5)
        if time.time() - start_time > 30:
            print(
                f"Waited {time.time() - start_time} seconds for Franka robot to be ready."
            )
    while True:
        try:
            cmd_str = input("Please input cmd:")
            if cmd_str == "q":
                break
            elif cmd_str == "getpos":
                print(controller.get_state().wait()[0].tcp_pose)
            elif cmd_str == "getpos_euler":
                tcp_pose = controller.get_state().wait()[0].tcp_pose
                r = R.from_quat(tcp_pose[3:].copy())
                euler = r.as_euler("xyz")
                print(np.concatenate([tcp_pose[:3], euler]))
            else:
                print(f"Unknown cmd: {cmd_str}")
        except KeyboardInterrupt:
            break
        time.sleep(1.0)


if __name__ == "__main__":
    main()
