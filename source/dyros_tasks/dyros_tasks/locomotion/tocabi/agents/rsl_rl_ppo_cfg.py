# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlPpoActorCriticCfg,
)

from dyros_rl.rsl_rl import (
    DyrosRslRlPpoRunnerCfg,
    DyrosRslRlPpoAlgorithmCfg,
    RslRlBoundLossCfg,
    RslRlLcpLossCfg,
)

@configclass
class TocabiRoughPPORunnerCfg(DyrosRslRlPpoRunnerCfg):
    num_steps_per_env = 64
    max_iterations = 5000
    save_interval = 50
    experiment_name = "tocabi_rough"
    empirical_normalization = True
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = DyrosRslRlPpoAlgorithmCfg(
        value_loss_coef=5,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=5,
        num_mini_batches=2,
        learning_rate=1.0e-5,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.004,
        max_grad_norm=1.0,
        lcp_loss_cfg=RslRlLcpLossCfg(
            gradient_penalty_coef=0.002,
            is_lcp=True,
        ),
        bound_loss_cfg=RslRlBoundLossCfg(
            bound_loss_coef=10,
            bound_range=1.1,
        ),
    )

@configclass
class TocabiFlatPPORunnerCfg(TocabiRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "tocabi_flat"


