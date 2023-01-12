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

"""Dexterity: schema for base class configuration.

Used by Hydra. Defines template for base class YAML file.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import *


@dataclass
class Mode:
    export_scene: bool  # export scene to USD
    export_states: bool  # export states to NPY


@dataclass
class Sim:
    dt: float  # timestep size (default = 1.0 / 60.0)
    num_substeps: int  # number of substeps (default = 2)
    num_pos_iters: int  # number of position iterations for PhysX TGS solver (default = 4)
    num_vel_iters: int  # number of velocity iterations for PhysX TGS solver (default = 1)
    gravity_mag: float  # magnitude of gravitational acceleration
    add_damping: bool  # add damping to stabilize gripper-object interactions


@dataclass
class Env:
    env_spacing: float  # lateral offset between envs
    franka_depth: float  # depth offset of Franka base relative to env origin
    table_height: float  # height of table
    franka_friction: float  # coefficient of friction associated with Franka
    table_friction: float  # coefficient of friction associated with table


@dataclass
class Debug:
    verbose: bool  # print verbose information
    visualize: List[str]  # list of visualizations to be called


@dataclass
class All:
    jacobian_type: str  # map between joint space and task space via geometric or analytic Jacobian {geometric, analytic}
    #residual_prop_gains: list[float]  # proportional gains on DoF position of joints after the ik_body
    #residual_deriv_gains: list[float]  # derivative gains on DoF position of joints after the ik_body


@dataclass
class GymDefault:
    ik_method: str
    #ik_prop_gains: list[int]  # proportional gains on DoF positions of joints used to position ik_body
    #ik_deriv_gains: list[int]  # derivative gains on DoF positions of joints used to position ik_body
    #residual_prop_gains: list[float]  # proportional gains on DoF position of joints after the ik_body
    #residual_deriv_gains: list[float]  # derivative gains on DoF position of joints after the ik_body


@dataclass
class JointSpaceID:
    ik_method: str
    #joint_prop_gains: list[int]
    #joint_deriv_gains: list[int]


@dataclass
class Ctrl:
    ctrl_type: str  # {gym_default, joint_space_id}
    ik_body: str  # {eef_body, tracker}  (which body to be positioned via ik)
    add_pose_actions_to: str  # {target, pose}
    gym_default: GymDefault
    joint_space_id: JointSpaceID

    #pos_action_scale: list[float]  # scale on pos displacement targets (3), to convert [-1, 1] to +- x m
    #rot_action_scale: list[float]  # scale on rot displacement targets (3), to convert [-1, 1] to +- x rad
    #force_action_scale: list[float]  # scale on force targets (3), to convert [-1, 1] to +- x N
    #torque_action_scale: list[float]  # scale on torque targets (3), to convert [-1, 1] to +- x Nm
    clamp_rot: bool  # clamp small values of rotation actions to zero
    clamp_rot_thresh: float  # smallest acceptable value


@dataclass
class DexteritySchemaConfigBase:
    mode: Mode
    sim: Sim
    env: Env
    ctrl: Ctrl
