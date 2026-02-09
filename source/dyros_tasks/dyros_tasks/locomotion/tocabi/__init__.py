# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Tocabi-Rough",
    entry_point="dyros.envs:DyrosManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:TocabiRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TocabiRoughPPORunnerCfg",
    },
)


gym.register(
    id="Tocabi-Rough-Play",
    entry_point="dyros.envs:DyrosManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:TocabiRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TocabiRoughPPORunnerCfg",
    },
)


gym.register(
    id="Tocabi-Flat",
    entry_point="dyros.envs:DyrosManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:TocabiFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TocabiFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tocabi-Flat-Play",
    entry_point="dyros.envs:DyrosManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:TocabiFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TocabiFlatPPORunnerCfg",
    },
)

# gym.register(
#     id="Tocabi-DeepMimic",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.flat_env_cfg:TocabiMimicEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TocabiMimicPPORunnerCfg",
#     },
# )


# gym.register(
#     id="Tocabi-DeepMimic-Play",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.flat_env_cfg:TocabiMimicEnvCfg_PLAY",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TocabiMimicPPORunnerCfg",
#     },
# )

# gym.register(
#     id="AMP-Tocabi-Flat",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.amp_env_cfg:TocabiAMPFlatEnvCfg",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TocabiAMPPPORunnerCfg",
#     },
# )

# gym.register(
#     id="AMP-Tocabi-Flat-Play",
#     entry_point="isaaclab.envs:ManagerBasedRLEnv",
#     disable_env_checker=True,
#     kwargs={
#         "env_cfg_entry_point": f"{__name__}.amp_env_cfg:TocabiAMPFlatEnvCfg_PLAY",
#         "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:TocabiAMPPPORunnerCfg",
#     },
# )
