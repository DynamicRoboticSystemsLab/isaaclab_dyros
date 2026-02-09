# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import math
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import dyros_tasks.locomotion.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, ObservationsCfg

from isaaclab.sensors import ContactSensorCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
# from isaaclab.managers import ObservationGroupCfg as ObsGroup
from dyros.managers import DyrosObservationGroupCfg as ObsGroup
# from isaaclab.managers import ObservationTermCfg as ObsTerm
from dyros.managers import DyrosObservationTermCfg as ObsTerm

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from dyros.terrains.config.rough import ROUGH_TERRAINS_CUSTOMCFG

from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

##
# Pre-defined configs
##
from dyros_assets import Tocabi_CFG  # isort: skip

@configclass
class TocabiActionsCfg:
    joint_pos = mdp.LowerBodyActionsCfg(
        asset_name="robot", 
        clip = {".*": (-1.0, 1.0)},
        lower_joint_names=["L_HipYaw_Joint", "L_HipRoll_Joint", "L_HipPitch_Joint", "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
                           "R_HipYaw_Joint", "R_HipRoll_Joint", "R_HipPitch_Joint", "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint"],
        upper_joint_names=["Waist1_Joint", "Waist2_Joint", "Upperbody_Joint",
                           "L_Shoulder1_Joint", "L_Shoulder2_Joint", "L_Shoulder3_Joint", "L_Elbow_Joint", "L_Armlink_Joint", "L_Forearm_Joint", "L_Wrist1_Joint", "L_Wrist2_Joint",
                           "Neck_Joint", "Head_Joint",
                           "R_Shoulder1_Joint", "R_Shoulder2_Joint", "R_Shoulder3_Joint", "R_Elbow_Joint", "R_Armlink_Joint", "R_Forearm_Joint", "R_Wrist1_Joint", "R_Wrist2_Joint"],
        pd_control=True,

        p_gains = [2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0, 
                   2000.0, 5000.0, 4000.0, 3700.0, 3200.0, 3200.0, 
                   6000.0, 10000.0, 10000.0, 
                   400.0, 1000.0, 400.0, 400.0, 400.0, 400.0, 100.0, 100.0, 
                   100.0, 100.0, 
                   400.0, 1000.0, 400.0, 400.0, 400.0, 400.0, 100.0, 100.0],

        d_gains = [15.0, 50.0, 20.0, 25.0, 24.0, 24.0, 
                   15.0, 50.0, 20.0, 25.0, 24.0, 24.0, 
                   200.0, 100.0, 100.0, 
                   10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0, 
                   3.0, 3.0, 
                   10.0, 28.0, 10.0, 10.0, 10.0, 10.0, 3.0, 3.0],

        torque_limits= [333, 232, 263, 289, 222, 166,
                        333, 232, 263, 289, 222, 166,
                        303, 303, 303,
                        64, 64, 64, 64, 23, 23, 10, 10,
                        10, 10,
                        64, 64, 64, 64, 23, 23, 10, 10],

        joint_pos_limits = [(-0.3, 0.3), (-0.5, 0.5), (-1.0, 0.5), (-0.3, 1.2), (-0.8, 0.5), (-0.6, 0.6), 
                            (-0.3, 0.3), (-0.5, 0.5), (-1.0, 0.5), (-0.3, 1.2), (-0.8, 0.5), (-0.6, 0.6)],

        rand_motor_scale_range = (0.8, 1.2)
    )

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.05,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.8, 0.8), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-0.8, 0.8)
        ),
    )
    phase_time = mdp.WalkingPhaseCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        ranges=mdp.WalkingPhaseCommandCfg.Ranges(
            # phase_time=(1.0, 2.0)
            phase_time=(1.2, 1.2)
        ),
    )

@configclass
class TocabiObservations(ObservationsCfg):
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.1, n_max=0.1))
        clock_input = ObsTerm(func=mdp.clock_input, params={"command_name": "phase_time"})
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_ordered_rel, 
                            noise=Unoise(n_min=-0.1, n_max=0.1),
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["L_HipYaw_Joint", "L_HipRoll_Joint", "L_HipPitch_Joint", "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
                                                                                     "R_HipYaw_Joint", "R_HipRoll_Joint", "R_HipPitch_Joint", "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint"])})
        joint_vel = ObsTerm(func=mdp.joint_vel_ordered, 
                            noise=Unoise(n_min=-0.1, n_max=0.1),
                            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["L_HipYaw_Joint", "L_HipRoll_Joint", "L_HipPitch_Joint", "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
                                                                                     "R_HipYaw_Joint", "R_HipRoll_Joint", "R_HipPitch_Joint", "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint"])})
        actions = ObsTerm(func=mdp.last_processed_action, params={"action_name": "joint_pos"})
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5
            self.skip_history_tick = 2
    policy: PolicyCfg = PolicyCfg()


@configclass
class Tocabi_WordlModelRewards:
    lin_vel_xy_tracking = RewTerm(func=mdp.track_lin_vel_xy_base_frame_exp, weight=1.0, params={"command_name": "base_velocity", "omega": 5.0})
    ang_vel_z_tracking = RewTerm(func=mdp.track_ang_vel_z_world_exp_custom, weight=1.0, params={"command_name": "base_velocity", "omega": 7.0})
    orientation_tracking = RewTerm(func=mdp.orientation_tracking, weight=1.0, params={"omega": 5.0})
    base_height_tracking = RewTerm(func=mdp.base_height_tracking, weight=0.5, params={"height": 0.928, "omega": 10.0})
    # ----------------------------------------------------------------------------------------------------------------------------------------------
    periodic_force = RewTerm(func=mdp.periodic_force, weight=1.0, params={
                        "scale": 1300, 
                        "command_name": "phase_time",
                        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["L_AnkleRoll_Link", "R_AnkleRoll_Link"]),
                        "left_foot": "L_AnkleRoll_Link", "right_foot": "R_AnkleRoll_Link"
    })
    # ----------------------------------------------------------------------------------------------------------------------------------------------
    feet_height_tracking = RewTerm(func=mdp.feet_height_tracking, weight=1.0, params={
                        # "omega": 5.0, "foot_height": 0.1, "kappa": 4.0, "offset": 0.1585,
                        "omega": 5.0, "foot_height": 0.1, "kappa": 2.0, "offset": -0.77,
                        "command_name": "phase_time",
                        "asset_cfg": SceneEntityCfg("robot", body_names=["L_AnkleRoll_Link", "R_AnkleRoll_Link"]),
                        "left_foot": "L_AnkleRoll_Link", "right_foot": "R_AnkleRoll_Link"})
    feet_velocity_z_tracking = RewTerm(func=mdp.feet_velocity_z_tracking, weight=0.5, params={
                        # "omega": 5.0, "foot_height": 0.1, "kappa": 4.0, "offset": 0.1585,
                        "omega": 3.0, "foot_height": 0.1, "kappa": 2.0,
                        "command_name": "phase_time",
                        "asset_cfg": SceneEntityCfg("robot", body_names=["L_AnkleRoll_Link", "R_AnkleRoll_Link"]),
                        "left_foot": "L_AnkleRoll_Link", "right_foot": "R_AnkleRoll_Link"})
    # ----------------------------------------------------------------------------------------------------------------------------------------------
    large_contact = RewTerm(func=mdp.large_contact_force, weight=-0.005, params={"threshold": 100.0,
                        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["L_AnkleRoll_Link", "R_AnkleRoll_Link"]),
                        "asset_cfg": SceneEntityCfg("robot")})
    default_joint = RewTerm(func=mdp.default_joint_pos, weight=0.2, params={"omega": 2.0, 
                        "asset_cfg": SceneEntityCfg("robot",
                        joint_names=["L_HipYaw_Joint", "L_HipRoll_Joint", "L_HipPitch_Joint", "L_Knee_Joint", "L_AnklePitch_Joint", "L_AnkleRoll_Joint",
                                        "R_HipYaw_Joint", "R_HipRoll_Joint", "R_HipPitch_Joint", "R_Knee_Joint", "R_AnklePitch_Joint", "R_AnkleRoll_Joint"])})
    # ----------------------------------------------------------------------------------------------------------------------------------------------
    # joint_deviation_hip = RewTerm(func=mdp.joint_deviation_l1, weight=-0.05, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_HipRoll_Joint"])})
    # joint_deviation_ankle = RewTerm(func=mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_AnkleRoll_Joint"])})
    # feet_flat = RewTerm(func=mdp.flat_orientation_l2, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", body_names=["L_AnkleRoll_Link", "R_AnkleRoll_Link"])})
    feet_flat = RewTerm(func=mdp.feet_flat, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", body_names=["L_AnkleRoll_Link", "R_AnkleRoll_Link"])})

@configclass
class TocabiTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="Pelvis_Link"), "threshold": 1.0},
    )
    root_bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.79, "asset_cfg": SceneEntityCfg("robot")},
    )
    
@configclass
class TocabiEventCfg:
    physics_material = EventTerm( func=mdp.randomize_rigid_body_material, mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (1.0, 1.0),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    randomize_rigid_body_com = EventTerm( func=mdp.randomize_rigid_body_com, mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    randomize_rigid_body_mass = EventTerm( func=mdp.randomize_rigid_body_mass, mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    reset_init_root_state = EventTerm( func=mdp.reset_root_state_uniform, mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0), "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
        }
    )
    reset_joints_by_offset = EventTerm( func=mdp.reset_joints_by_offset, mode="reset",
        params={
            "position_range": (-0.0, 0.0), "velocity_range": (0.0, 0.0),
        },
    )
    randomize_armature = EventTerm( func=mdp.randomize_joint_parameters, mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "friction_distribution_params": (0.0, 0.0),
            "armature_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )
    randomize_damping = EventTerm( func=mdp.randomize_actuator_gains, mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.0, 0.0),
            "damping_distribution_params": (-0.9, 2.0),
            "operation": "add",
        },
    )


@configclass
class TocabiRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: Tocabi_WordlModelRewards = Tocabi_WordlModelRewards()
    observations: TocabiObservations = TocabiObservations()
    actions: TocabiActionsCfg = TocabiActionsCfg()
    terminations: TocabiTerminations = TocabiTerminations()
    events: TocabiEventCfg = TocabiEventCfg()
    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = Tocabi_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if self.scene.height_scanner:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/Pelvis_Link"
        

        self.decimation = 2
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        # self.sim.dt = 0.002


@configclass
class TocabiRoughEnvCfg_PLAY(TocabiRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        # self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator = ROUGH_TERRAINS_CUSTOMCFG

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
