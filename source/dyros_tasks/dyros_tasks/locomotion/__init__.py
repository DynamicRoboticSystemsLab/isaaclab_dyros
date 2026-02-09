# Copyright (c) 2025, Dyros Lab.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locomotion tasks for Dyros Lab.

Importing this package will register the environments under `dyros_tasks.locomotion.*`.
"""

# Register environments by importing subpackages with gym.register(...) side-effects.
from . import tocabi  # noqa: F401


