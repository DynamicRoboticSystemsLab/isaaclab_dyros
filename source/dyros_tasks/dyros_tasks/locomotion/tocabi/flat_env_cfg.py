# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from isaaclab.managers import RewardTermCfg as RewTerm
from .rough_env_cfg import TocabiRoughEnvCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import dyros_tasks.locomotion.mdp as mdp
from isaaclab.managers import SceneEntityCfg
import isaaclab.sim as sim_utils
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.assets import RigidObjectCfg

@configclass
class TocabiFlatEnvCfg(TocabiRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        self.scene.cube = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(0.2, 0.2, 0.1),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    kinematic_enabled=True,
                    disable_gravity=True,
                    enable_gyroscopic_forces=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=0,
                    max_depenetration_velocity=1.0,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.005,
                    rest_offset=0.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0), metallic=0.2),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
        )
        self.events.reset_cube = EventTerm( func=mdp.reset_root_state_uniform, mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("cube"),
                "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            }
        )

class TocabiFlatEnvCfg_PLAY(TocabiFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.episode_length_s = 10.0
        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        
        
        # self.viewer.resolution = (2560, 1440)
        # # self.viewer.resolution = (1920, 1080)
        # self.viewer.eye = (-3.0, 3.0, 1.3)
        # self.viewer.lookat = (0.0, 0.0, 0.0)
        # self.viewer.origin_type = "asset_root"
        # self.viewer.asset_name = "robot"
        # self.viewer.env_index = -1

        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        # self.events.push_robot = None
        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)

        # self.rewards.contact_force_l = RewTerm(
        #     func=mdp.contact_force,
        #     weight=1.0,
        #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="L_AnkleRoll.*")},
        # )
        # self.rewards.contact_force_r = RewTerm(
        #     func=mdp.contact_force,
        #     weight=1.0,
        #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="R_AnkleRoll.*")},
        # )




