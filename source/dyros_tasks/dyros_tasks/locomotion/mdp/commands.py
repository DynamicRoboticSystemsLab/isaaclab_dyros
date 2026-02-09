from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING
from dataclasses import MISSING

import omni.log

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

class WalkingPhaseCommand(CommandTerm):
    """Command generator that generates a walking phase command."""

    cfg: WalkingPhaseCommandCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: WalkingPhaseCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.phase_time_cmd = torch.zeros(self.num_envs, 1, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.phase_time_cmd
    
    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        # sample phase time command
        r = torch.empty(len(env_ids), device=self.device)
        self.phase_time_cmd[env_ids, 0] = r.uniform_(*self.cfg.ranges.phase_time)

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        pass



@configclass
class WalkingPhaseCommandCfg(CommandTermCfg):
    """Configuration for the walking phase command generator."""

    class_type: type = WalkingPhaseCommand

    asset_name: str = MISSING
    """Name of the asset in the environment for which the commands are generated."""

    @configclass
    class Ranges:
        """Uniform distribution ranges for the walking phase commands."""

        phase_time: tuple[float, float] = MISSING
        """Range for the phase time (in s)."""

    ranges: Ranges = MISSING
    """Distribution ranges for the walking phase commands."""