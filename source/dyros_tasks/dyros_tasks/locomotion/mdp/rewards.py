# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2025, Jaeyong Shin (jasonshin0537@snu.ac.kr).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import math
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


"""Custom rewards and penalties"""
def track_lin_vel_xy_base_frame_exp(
    env, omega: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the base frame using exponential kernel."""
    asset = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1
    )
    return torch.exp(-omega*lin_vel_error)

def track_ang_vel_z_world_exp_custom(
    env, command_name: str, omega: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-omega*ang_vel_error)

def different_step_times(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize different step times between the two feet.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the penalty
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    penalty = 2 * (torch.std(last_contact_time, dim=1) + torch.std(last_air_time, dim=1))
    # no penalty for zero command
    penalty *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.05
    return penalty


def different_air_contact_times(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize huge difference between air and contact times.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the penalty
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    air_contact_time_abs_diff = torch.abs(last_air_time - last_contact_time)
    penalty = torch.sum(air_contact_time_abs_diff, dim=1)
    # no penalty for zero command
    penalty *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.05
    return penalty


# from isaaclab_tasks.manager_based.locomotion.velocity.config.tocabi.motions.motion_loader import MotionLoader
# # -----------------------------------------------------------------------------
# # Deep Mimic motion cache (avoid reloading file each simulation step)
# # -----------------------------------------------------------------------------
# _DEEP_MIMIC_MOTION_CACHE: dict[str, MotionLoader] = {}

# def _get_motion_loader(motion_file: str, device: torch.device | str):
#     """Return cached MotionLoader, loading if necessary."""
#     if isinstance(device, str):
#         device = torch.device(device)
#     if motion_file not in _DEEP_MIMIC_MOTION_CACHE:
#         _DEEP_MIMIC_MOTION_CACHE[motion_file] = MotionLoader(motion_file, device=device)
#     return _DEEP_MIMIC_MOTION_CACHE[motion_file]


# def deep_mimic_rewards(
#     env,
#     step_time: float,
#     alpha: float,
#     motion_file: str,
#     asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
# ) -> torch.Tensor:
#     """Reward tracking of whole-body motion from motion capture data.

#     Args:
#         env: RL 환경.
#         step_time: 모션이 한 주기를 도는 시간(s).
#         motion_file: NPZ 또는 TXT 등 모션 파일 경로 (문자열).
#         asset_cfg: 로봇 SceneEntity 설정.
#     """
#     # Lazy-load & cache motion
#     motion = _get_motion_loader(motion_file, device=env.device)

#     # 현재 phase 계산 (0~duration)
#     # 5600 ~ 9201 is the range of the motion data (2.8s ~ 4.6s)
#     phase = (env.episode_length_buf.unsqueeze(1) * env.step_dt % step_time) + 2.8
#     joint_position, _, _, _, _, _, _ = motion.sample(1, times=phase.cpu().numpy().reshape(-1))
#     # rearange the joint position to the order of the asset_cfg
#     asset = env.scene[asset_cfg.name]
#     # get the reference joint order in motion that matches the asset's joint ids
#     joint_names_selected = [asset.data.joint_names[i] for i in asset_cfg.joint_ids]
#     selected_joints = motion.get_dof_index(joint_names_selected)

#     # reference and current joint positions
#     joint_position_ref = joint_position[:, selected_joints]
#     current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]

#     # compute reward with exponential kernel
#     reward = torch.exp(-alpha * torch.norm(joint_position_ref - current_joint_pos, dim=1) ** 2)
#     return reward


# -----------------------------------------------------------------------------
# World Model Rewards
# -----------------------------------------------------------------------------
def orientation_tracking(env, omega: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ maintain base orientation(alpha, beta) flat in base frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    return torch.exp(-omega*(torch.square(roll) + torch.square(pitch)))

def feet_flat(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ give penalty if the projected gravity vector is not flat"""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    reward = torch.zeros(env.num_envs, device=env.device)
    for ids in asset_cfg.body_ids:
        link_quat_w = asset.data.body_link_quat_w[:, ids, :].squeeze(1)
        gravity = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
        projected_gravity = quat_apply_inverse(link_quat_w, gravity)
        reward += torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
    return reward

def base_height_tracking(env, height: float, omega: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """ maintain base height(z) flat in base frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    return torch.exp(-omega*torch.square(asset.data.root_pos_w[:, 2] - height))

def periodic_force(
        env, 
        scale: float, 
        command_name: str, 
        left_foot: str, 
        right_foot: str, 
        sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward periodic force on the robot's feet. reasonable foot forces during the stance phase of locomotion."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    phase_time = env.command_manager.get_command(command_name)

    cycle_time = env.episode_length_buf.unsqueeze(1) * env.step_dt % phase_time
    left_foot_indicator = torch.where(cycle_time < phase_time/2, 1.0, 0.0) # left foot
    right_foot_indicator = torch.where(cycle_time >= phase_time/2, 1.0, 0.0) # right foot

    # left_foot_indicator = torch.where(~is_rfoot, first_foot_indicator, second_foot_indicator)
    # right_foot_indicator = torch.where(is_rfoot, first_foot_indicator, second_foot_indicator)

    left_foot_idx = [sensor_cfg.body_ids[i] for i in range(len(sensor_cfg.body_ids)) if left_foot in sensor_cfg.body_names[i]]
    right_foot_idx = [sensor_cfg.body_ids[i] for i in range(len(sensor_cfg.body_ids)) if right_foot in sensor_cfg.body_names[i]]

    left_foot_force = contact_sensor.data.net_forces_w_history[:, :, left_foot_idx, :].norm(dim=-1).max(dim=1)[0]
    right_foot_force = contact_sensor.data.net_forces_w_history[:, :, right_foot_idx, :].norm(dim=-1).max(dim=1)[0]
    reward = left_foot_indicator * left_foot_force / scale + right_foot_indicator * right_foot_force / scale
    reward = torch.clip(reward, 0.0, 1.0)
    return reward.squeeze()

def periodic_velocity(
        env, 
        scale: float, 
        command_name: str, 
        left_foot: str, 
        right_foot: str, 
        asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward periodic velocity on the robot's feet. reasonable foot velocity during the stance phase of locomotion."""
    asset = env.scene[asset_cfg.name]
    phase_time = env.command_manager.get_command(command_name)

    cycle_time = env.episode_length_buf.unsqueeze(1) * env.step_dt % phase_time
    left_foot_indicator = torch.where(cycle_time < phase_time/2, 1.0, 0.0) # left foot
    right_foot_indicator = torch.where(cycle_time >= phase_time/2, 1.0, 0.0) # right foot

    # left_foot_indicator = torch.where(~is_rfoot, first_foot_indicator, second_foot_indicator)
    # right_foot_indicator = torch.where(is_rfoot, first_foot_indicator, second_foot_indicator)

    left_foot_idx = [asset_cfg.body_ids[i] for i in range(len(asset_cfg.body_ids)) if left_foot in asset_cfg.body_names[i]]
    right_foot_idx = [asset_cfg.body_ids[i] for i in range(len(asset_cfg.body_ids)) if right_foot in asset_cfg.body_names[i]]

    left_foot_vel = torch.norm(asset.data.body_lin_vel_w[:, left_foot_idx, :], dim=-1)
    right_foot_vel = torch.norm(asset.data.body_lin_vel_w[:, right_foot_idx, :], dim=-1)

    reward = (1 - left_foot_indicator) * left_foot_vel * scale + (1 - right_foot_indicator) * right_foot_vel * scale
    reward = torch.clip(reward, 0.0, 1.0)
    return reward.squeeze()

def feet_height_tracking(
        env, 
        omega: float, 
        foot_height: float, 
        kappa: float, 
        offset: float, 
        command_name: str, 
        left_foot: str, 
        right_foot: str, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward tracking of feet height trajectory. generated by Von Mises distribution."""
    asset = env.scene[asset_cfg.name]
    phase_time = env.command_manager.get_command(command_name)

    B = torch.exp(-torch.tensor(kappa))
    norm = torch.exp(torch.tensor(kappa))-torch.exp(-torch.tensor(kappa))
    theta1 = 2 * math.pi * (env.episode_length_buf.unsqueeze(1) * env.step_dt % phase_time) / phase_time - math.pi/2 #right foot
    theta2 = 2 * math.pi * (env.episode_length_buf.unsqueeze(1) * env.step_dt - phase_time/2) % phase_time / phase_time - math.pi/2 #left foot
    gx1 = torch.exp(torch.tensor(kappa) * torch.cos(theta1))
    gx2 = torch.exp(torch.tensor(kappa) * torch.cos(theta2))
    right_foot_traj = foot_height * (gx1-B) / norm + offset
    left_foot_traj = foot_height * (gx2-B) / norm + offset

    # left_foot_traj = torch.where(~is_rfoot, first_foot_traj, second_foot_traj)
    # right_foot_traj = torch.where(is_rfoot, first_foot_traj, second_foot_traj)

    left_foot_idx = [asset_cfg.body_ids[i] for i in range(len(asset_cfg.body_ids)) if left_foot in asset_cfg.body_names[i]]
    right_foot_idx = [asset_cfg.body_ids[i] for i in range(len(asset_cfg.body_ids)) if right_foot in asset_cfg.body_names[i]]

    # change to tracking based on base frame 
    l_foot_pos_b = quat_apply_inverse(asset.data.root_quat_w, asset.data.body_pos_w[:, left_foot_idx[0], :3] - asset.data.root_pos_w[:, :3])
    r_foot_pos_b = quat_apply_inverse(asset.data.root_quat_w, asset.data.body_pos_w[:, right_foot_idx[0], :3] - asset.data.root_pos_w[:, :3])

    # left_foot_pos_z = asset.data.body_pos_w[:, left_foot_idx, 2] - left_foot_traj
    # right_foot_pos_z = asset.data.body_pos_w[:, right_foot_idx, 2] - right_foot_traj
    left_foot_pos_z = l_foot_pos_b[:, 2].unsqueeze(1) - left_foot_traj
    right_foot_pos_z = r_foot_pos_b[:, 2].unsqueeze(1) - right_foot_traj
    foot_tracking_error = torch.sum(torch.square(left_foot_pos_z) + torch.square(right_foot_pos_z), dim=1).squeeze()
    return torch.exp(-omega*foot_tracking_error)

def feet_velocity_z_tracking(
        env, 
        omega: float, 
        foot_height: float, 
        kappa: float, 
        command_name: str, 
        left_foot: str, 
        right_foot: str, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward tracking of feet velocity trajectory. generated by differentiated Von Mises distribution."""
    asset = env.scene[asset_cfg.name]
    phase_time = env.command_manager.get_command(command_name)

    B = torch.exp(-torch.tensor(kappa))
    norm = torch.exp(torch.tensor(kappa))-torch.exp(-torch.tensor(kappa))
    theta_right = 2 * math.pi * (env.episode_length_buf.unsqueeze(1) * env.step_dt % phase_time) / phase_time - math.pi/2
    theta_left = 2 * math.pi * (env.episode_length_buf.unsqueeze(1) * env.step_dt - phase_time/2) % phase_time / phase_time - math.pi/2
    gx_right = torch.exp(torch.tensor(kappa) * torch.cos(theta_right))
    gx_left = torch.exp(torch.tensor(kappa) * torch.cos(theta_left))
    dtheta = 2 * math.pi / phase_time
    d_gx_right = -kappa * torch.sin(theta_right) * gx_right * dtheta
    d_gx_left = -kappa * torch.sin(theta_left) * gx_left * dtheta

    foot_traj_right = foot_height * (d_gx_right) / norm #right foot
    foot_traj_left = foot_height * (d_gx_left) / norm #left foot

    # foot_traj_left = torch.where(~is_rfoot, foot_traj_first, foot_traj_second)
    # foot_traj_right = torch.where(is_rfoot, foot_traj_first, foot_traj_second)

    left_foot_idx = [asset_cfg.body_ids[i] for i in range(len(asset_cfg.body_ids)) if left_foot in asset_cfg.body_names[i]]
    right_foot_idx = [asset_cfg.body_ids[i] for i in range(len(asset_cfg.body_ids)) if right_foot in asset_cfg.body_names[i]]

    left_foot_vel_z_error = asset.data.body_lin_vel_w[:, left_foot_idx, 2] - foot_traj_left
    right_foot_vel_z_error = asset.data.body_lin_vel_w[:, right_foot_idx, 2] - foot_traj_right

    # left_foot_vel_y_error = asset.data.body_lin_vel_w[:, left_foot_idx, 1]
    # right_foot_vel_y_error = asset.data.body_lin_vel_w[:, right_foot_idx, 1]

    foot_vel_tracking_error = torch.sum(torch.square(left_foot_vel_z_error) + torch.square(right_foot_vel_z_error), dim=1).squeeze()
    return torch.exp(-omega*foot_vel_tracking_error)

def feet_height_poly_tracking(
        env, 
        omega: float, 
        foot_height: float, 
        phase_time: float, 
        kappa: float, 
        offset: float, 
        left_foot: str, 
        right_foot: str, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward tracking of feet height trajectory. generated by polynomial function."""
    asset = env.scene[asset_cfg.name]

    B = torch.exp(-torch.tensor(kappa))
    norm = torch.exp(torch.tensor(kappa))-torch.exp(-torch.tensor(kappa))
    theta1 = 2 * math.pi * (env.episode_length_buf.unsqueeze(1) * env.step_dt % phase_time) / phase_time - math.pi/2 #right foot
    theta2 = 2 * math.pi * (env.episode_length_buf.unsqueeze(1) * env.step_dt - phase_time/2) % phase_time / phase_time - math.pi/2 #left foot
    gx1 = torch.exp(torch.tensor(kappa) * torch.cos(theta1))
    gx2 = torch.exp(torch.tensor(kappa) * torch.cos(theta2))
    
    right_foot_traj = foot_height * (gx1-B) / norm + offset
    left_foot_traj = foot_height * (gx2-B) / norm + offset


    left_foot_idx = [asset_cfg.body_ids[i] for i in range(len(asset_cfg.body_ids)) if left_foot in asset_cfg.body_names[i]]
    right_foot_idx = [asset_cfg.body_ids[i] for i in range(len(asset_cfg.body_ids)) if right_foot in asset_cfg.body_names[i]]

    l_foot_pos_b = quat_apply_inverse(asset.data.root_quat_w, asset.data.body_pos_w[:, left_foot_idx[0], :] - asset.data.root_pos_w[:, :3])
    r_foot_pos_b = quat_apply_inverse(asset.data.root_quat_w, asset.data.body_pos_w[:, right_foot_idx[0], :] - asset.data.root_pos_w[:, :3])

    # left_foot_pos_z = asset.data.body_pos_w[:, left_foot_idx, 2] - left_foot_traj
    # right_foot_pos_z = asset.data.body_pos_w[:, right_foot_idx, 2] - right_foot_traj
    left_foot_pos_z = l_foot_pos_b[:, 2].unsqueeze(1) - left_foot_traj
    right_foot_pos_z = r_foot_pos_b[:, 2].unsqueeze(1) - right_foot_traj

    foot_tracking_error = torch.sum(torch.square(left_foot_pos_z) + torch.square(right_foot_pos_z), dim=1).squeeze()
    return torch.exp(-omega*foot_tracking_error)

def large_contact_force(env, threshold: float, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize large contact force on the robot's feet."""
    asset = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    max_force = 1.3 * 9.81 * asset.data.default_mass[:, asset_cfg.body_ids].sum(dim=1).unsqueeze(1).to(contact_forces.device)
    clipped_contact_forces = torch.clip(contact_forces-max_force, 0.0, threshold)
    reward = torch.sum(clipped_contact_forces, dim=1)
    return reward

def default_joint_pos(env, omega: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward default joint positions."""
    asset = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    error = torch.norm(default_joint_pos - current_joint_pos, dim=1).squeeze()
    return torch.exp(-omega*error)
