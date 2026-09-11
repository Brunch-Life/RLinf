# Copyright 2026 The RLinf Authors.
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

"""Task-owned Panda with the insertion peg integrated into its hand body."""

import numpy as np
import sapien
from mani_skill import format_path
from mani_skill.agents.robots.panda import PandaWristCam
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs import Link, Pose
from sapien.wrapper.articulation_builder import LinkBuilder


class _PegHandBuilder(LinkBuilder):
    """Preserve parsed hand geometry and filter the tool before scene insertion."""

    def __init__(self, hand: LinkBuilder):
        super().__init__(hand.index, hand.parent)
        self.name = hand.name
        self.scene = hand.scene
        self.joint_record = hand.joint_record
        self.collision_records = list(hand.collision_records)
        self.visual_records = list(hand.visual_records)
        self.collision_groups = list(hand.collision_groups)
        self.use_density = hand.use_density
        self.initial_pose = hand.initial_pose
        if hand._auto_inertial:
            raise ValueError("PegSlot requires the Panda hand's explicit URDF inertia")
        self.set_mass_and_inertia(
            hand._mass, hand._cmass_local_pose, hand._inertia.copy()
        )
        self.peg_collision_groups: list[int] = []
        self.peg_local_pose = sapien.Pose()
        self.peg_half_size = np.zeros(3)

    def build_physx_component(self, link_parent=None):
        body = super().build_physx_component(link_parent)
        shape = body.get_collision_shapes()[-1]
        if not isinstance(shape, sapien.physx.PhysxCollisionShapeBox):
            raise RuntimeError("The appended peg collision shape was not built")
        if not np.allclose(shape.half_size, self.peg_half_size, atol=1e-8):
            raise RuntimeError("Unexpected peg collision dimensions")
        if not np.allclose(shape.local_pose.p, self.peg_local_pose.p, atol=1e-7):
            raise RuntimeError("Unexpected peg collision position")
        if not np.allclose(shape.local_pose.q, self.peg_local_pose.q, atol=1e-7):
            raise RuntimeError("Unexpected peg collision orientation")
        # This must precede adding the link to its entity/scene. Changing a
        # constructed articulation's self-collision masks is not reliable.
        shape.set_collision_groups(self.peg_collision_groups)
        return body


class RigidPeg:
    """Read-only tool pose on the hand; not an independent actor or saved state.

    Contact queries through this view refer to the whole hand body, including
    its peg geometry. Only the robot articulation can move the peg.
    """

    name = "peg"

    def __init__(self, hand: Link, local_pose: sapien.Pose):
        self.hand = hand
        self.local_pose = local_pose
        self._bodies = hand._bodies

    @property
    def pose(self) -> Pose:
        return self.hand.pose * self.local_pose


class PegSlotPanda(PandaWristCam):
    """Panda wrist-camera robot carrying a rigid peg and fixed finger opening.

    The inherited uid keeps existing Panda controller/scene initialization
    compatible. This class is instantiated only by PegSlotEnv; it does not
    replace the globally registered Panda. The tool inherits the robot's GPU
    gravity compensation, and cannot slip or fall out of the fingers.
    """

    PEG_HALF_SIZE = (0.009, 0.009, 0.060)
    PEG_DENSITY = 1000.0
    GRIPPER_HOLD = 0.008

    def _load_articulation(self, initial_pose=None):
        if self.build_separate:
            raise ValueError("PegSlotPanda does not support build_separate=True")
        loader = self.scene.create_urdf_loader()
        loader.name = self.uid
        if self._agent_idx is not None:
            loader.name = f"{self.uid}-agent-{self._agent_idx}"
        loader.fix_root_link = self.fix_root_link
        loader.load_multiple_collisions_from_file = self.load_multiple_collisions
        loader.disable_self_collisions = self.disable_self_collisions
        config = sapien_utils.parse_urdf_config(self.urdf_config)
        sapien_utils.check_urdf_config(config)
        sapien_utils.apply_urdf_config(loader, config)
        parsed = loader.parse(format_path(str(self.urdf_path)))
        if len(parsed["articulation_builders"]) != 1:
            raise ValueError("PegSlotPanda requires a single Panda articulation")
        builder = parsed["articulation_builders"][0]
        self._attach_peg(builder)
        builder.initial_pose = initial_pose
        # ManiSkill's builder registers the articulation and its saved state.
        self.robot = builder.build()
        self.robot_link_names = [link.name for link in self.robot.get_links()]

    def _attach_peg(self, builder):
        links = {link.name: link for link in builder.link_builders}
        original_hand = links["panda_hand"]
        tcp = links[self.ee_link_name]
        if tcp.parent is not original_hand or tcp.joint_record.joint_type != "fixed":
            raise ValueError("PegSlot requires a fixed TCP directly on panda_hand")
        hand = _PegHandBuilder(original_hand)
        builder.link_builders[hand.index] = hand
        for link in builder.link_builders:
            if link.parent is original_hand:
                link.parent = hand
        links["panda_hand"] = hand

        tcp_in_hand = (
            tcp.joint_record.pose_in_parent * tcp.joint_record.pose_in_child.inv()
        )
        peg_in_tcp = sapien.Pose(p=[0.0, 0.0, self.PEG_HALF_SIZE[2] - 0.006])
        peg_in_hand = tcp_in_hand * peg_in_tcp
        half = np.asarray(self.PEG_HALF_SIZE, dtype=np.float64)
        peg_mass = float(np.prod(2 * half) * self.PEG_DENSITY)
        box_inertia = (
            peg_mass
            / 3
            * np.array(
                [
                    half[1] ** 2 + half[2] ** 2,
                    half[0] ** 2 + half[2] ** 2,
                    half[0] ** 2 + half[1] ** 2,
                ]
            )
        )
        hand_mass = float(hand._mass)
        hand_com = np.asarray(hand._cmass_local_pose.p, dtype=np.float64)
        hand_rotation = hand._cmass_local_pose.to_transformation_matrix()[
            :3, :3
        ].astype(np.float64)
        peg_rotation = peg_in_hand.to_transformation_matrix()[:3, :3].astype(np.float64)
        peg_com = np.asarray(peg_in_hand.p, dtype=np.float64)
        mass = hand_mass + peg_mass
        com = (hand_mass * hand_com + peg_mass * peg_com) / mass
        inertia = (
            hand_rotation @ np.diag(hand._inertia) @ hand_rotation.T
            + peg_rotation @ np.diag(box_inertia) @ peg_rotation.T
        )
        for part_mass, part_com in ((hand_mass, hand_com), (peg_mass, peg_com)):
            offset = part_com - com
            inertia += part_mass * (
                np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset)
            )
        moments, axes = np.linalg.eigh(inertia)
        if np.any(moments <= 0):
            raise ValueError("The combined hand and peg inertia must be positive")
        if np.linalg.det(axes) < 0:
            axes[:, 2] *= -1
        com_transform = np.eye(4)
        com_transform[:3, :3] = axes
        com_transform[:3, 3] = com
        hand.set_mass_and_inertia(mass, sapien.Pose(com_transform), moments)
        hand.add_box_collision(
            pose=peg_in_hand, half_size=half, density=self.PEG_DENSITY
        )
        hand.add_box_visual(
            pose=peg_in_hand,
            half_size=half,
            material=sapien.render.RenderMaterial(
                base_color=sapien_utils.hex2rgba("#F26B38"), roughness=0.45
            ),
        )

        occupied = 0
        for link in links.values():
            occupied |= link.collision_groups[2]
        available = [bit for bit in range(28, -1, -1) if not occupied & (1 << bit)]
        if not available:
            raise ValueError("No collision-filter bit is available for the peg")
        peg_group = 1 << available[0]
        for name in ("panda_leftfinger", "panda_rightfinger"):
            finger = links[name]
            finger.collision_groups[2] |= peg_group
            finger.joint_record.limits = [[self.GRIPPER_HOLD, self.GRIPPER_HOLD]]
        hand.peg_collision_groups = [1, 1, peg_group, hand.collision_groups[3]]
        hand.peg_local_pose = peg_in_hand
        hand.peg_half_size = half
        self.peg_local_pose = peg_in_hand
        self.peg_metadata = {
            "attachment": "panda_hand_rigid_body",
            "peg_half_size_m": half.tolist(),
            "peg_center_in_tcp_m": peg_in_tcp.p.tolist(),
            "peg_center_in_hand_m": peg_in_hand.p.tolist(),
            "peg_mass_kg": peg_mass,
            "combined_hand_mass_kg": mass,
            "combined_com_in_hand_m": com.tolist(),
            "combined_inertia_in_hand_kg_m2": inertia.tolist(),
            "gripper_hold_m": self.GRIPPER_HOLD,
            "inherits_robot_gravity_compensation": True,
        }

    def _after_init(self):
        super()._after_init()
        self.peg = RigidPeg(self.robot.links_map["panda_hand"], self.peg_local_pose)
