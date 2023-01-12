# Copyright (c) 2021-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Dexterity: schema for task class configurations.

Used by Hydra. Defines template for task class YAML files. Not enforced.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Sim:
    use_gpu_pipeline: bool  # use GPU pipeline
    up_axis: str  # up-down axis {x, y, z}
    dt: float  # timestep size
    gravity: list[float]  # gravity vector

    disable_gravity: bool  # disable gravity for all actors


@dataclass
class Env:
    numObservations: int  # number of observations per env; camel case required by VecTask
    numActions: int  # number of actions per env; camel case required by VecTask
    numEnvs: int  # number of envs; camel case required by VecTask


@dataclass
class Randomize:
    franka_arm_initial_dof_pos: list[float]  # initial Franka arm DOF position (7)


@dataclass
class RL:
    max_episode_length: int  # max number of timesteps in each episode


@dataclass
class DexteritySchemaConfigTask:
    name: str
    physics_engine: str
    sim: Sim
    env: Env
    rl: RL
