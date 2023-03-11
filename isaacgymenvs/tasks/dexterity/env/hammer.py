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

"""Dexterity: class for hammer env.

Inherits base class and abstract environment class. Inherited by hammer task
classes. Not directly executed.

Configuration defined in DexterityEnvHammer.yaml.
"""

import hydra
import numpy as np
import os
import torch
from typing import *

from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.dexterity.env.tool_use import DexterityEnvToolUse
from isaacgymenvs.tasks.dexterity.env.schema_config_env import \
    DexteritySchemaConfigEnv
from isaacgymenvs.tasks.dexterity.env.object import randomize_rotation


class DexterityEnvHammer(DexterityEnvToolUse):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless,
                 virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize environment superclass.
        Acquire tensors."""

        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render,
                         tool_category="hammer")

    def create_envs(self):
        """Set env options. Import assets. Create actors."""

        lower = gymapi.Vec3(-self.cfg_base.env.env_spacing,
                            -self.cfg_base.env.env_spacing, 0.0)
        upper = gymapi.Vec3(self.cfg_base.env.env_spacing,
                            self.cfg_base.env.env_spacing,
                            self.cfg_base.env.env_spacing)
        num_per_row = int(np.sqrt(self.num_envs))

        robot_asset, table_asset = self.import_robot_assets()
        source_hammer, target_hammers = self._import_tool_assets()

        nail_asset = self._import_nail_asset()

        self._create_actors(lower, upper, num_per_row, robot_asset, table_asset,
                            source_hammer, target_hammers, nail_asset)

    def _import_env_assets(self):
        """Set object assets options. Import objects."""
        hammer_assets = self._import_hammer_assets()
        nail_asset = self._import_nail_asset()
        return hammer_assets, nail_asset

    def _import_nail_asset(self):
        asset_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '../..', '..', '..',
                         'assets', 'dexterity', 'tools', 'hammer'))
        asset_file = 'nail.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True

        nail_asset = self.gym.load_asset(self.sim, asset_root, asset_file,
                                         asset_options)
        return nail_asset

    def _create_actors(self, lower: gymapi.Vec3, upper: gymapi.Vec3,
                       num_per_row: int, robot_asset, table_asset,
                       source_hammer, target_hammers, nail_asset) -> None:
        robot_pose = gymapi.Transform()
        robot_pose.p.x = -self.cfg_base.env.robot_depth
        robot_pose.p.y = 0.0
        robot_pose.p.z = 0.0
        robot_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        table_pose = gymapi.Transform()
        table_pose.p.x = 0.0
        table_pose.p.y = 0.0
        table_pose.p.z = self.cfg_base.env.table_height * 0.5
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        nail_pose = gymapi.Transform()
        nail_pose.p.x = 0.5
        nail_pose.p.z = 0.1
        hammer_pose = gymapi.Transform()

        self.env_ptrs = []
        self.nail_handles = []
        self.hammer_handles = []
        self.nail_actor_ids_sim = []  # within-sim indices
        self.hammer_actor_ids_sim = []  # within-sim indices

        hammers_list = [source_hammer, ] + target_hammers

        actor_count = 0
        hammer_count = len(hammers_list)
        for i in range(self.num_envs):
            # Create new env
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Loop through all used hammers
            hammer_idx = i % hammer_count
            used_hammer = hammers_list[hammer_idx]

            # Aggregate all actors
            if self.cfg_base.sim.aggregate_mode > 1:
                max_rigid_bodies = self.base_rigid_bodies + \
                    used_hammer.rigid_body_count + \
                    self.gym.get_asset_rigid_body_count(nail_asset)
                max_rigid_shapes = self.base_rigid_shapes + \
                    used_hammer.rigid_shape_count + \
                    self.gym.get_asset_rigid_shape_count(nail_asset)
                self.gym.begin_aggregate(env_ptr, max_rigid_bodies,
                                         max_rigid_shapes, True)

            # Create common actors (robot, table, cameras)
            actor_count = self.create_base_actors(
                env_ptr, i, actor_count, robot_asset, robot_pose,
                table_asset, table_pose)

            # Aggregate task-specific actors (hammers and nails)
            if self.cfg_base.sim.aggregate_mode == 1:
                max_rigid_bodies = \
                    used_hammer.rigid_body_count + \
                    self.gym.get_asset_rigid_body_count(nail_asset)
                max_rigid_shapes = \
                    used_hammer.rigid_shape_count + \
                    self.gym.get_asset_rigid_shape_count(nail_asset)
                self.gym.begin_aggregate(env_ptr, max_rigid_bodies,
                                         max_rigid_shapes, True)

            # Create nail actor
            nail_handle = self.gym.create_actor(
                env_ptr, nail_asset, nail_pose, 'nail', i, 0, 2)
            self.nail_actor_ids_sim.append(actor_count)
            self.nail_handles.append(nail_handle)
            self.gym.set_rigid_body_color(
                env_ptr, nail_handle, 1, gymapi.MESH_VISUAL,
                gymapi.Vec3(*self.cfg_env['env']['nail_color']))
            self.gym.set_rigid_body_color(
                env_ptr, nail_handle, 2, gymapi.MESH_VISUAL,
                gymapi.Vec3(*self.cfg_env['env']['nail_color']))
            actor_count += 1

            # Create hammer actor
            hammer_handle = self.gym.create_actor(
                env_ptr, used_hammer.asset, hammer_pose, 'hammer', i, 0, 2)
            self.hammer_actor_ids_sim.append(actor_count)
            self.hammer_handles.append(hammer_handle)
            actor_count += 1

            # Finish aggregation group
            if self.cfg_base.sim.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.env_ptrs.append(env_ptr)

        self.num_actors = int(actor_count / self.num_envs)  # per env
        self.num_bodies = self.gym.get_env_rigid_body_count(env_ptr)  # per env
        self.num_dofs = self.gym.get_env_dof_count(env_ptr)  # per env

        # For setting targets
        self.robot_actor_ids_sim = torch.tensor(
            self.robot_actor_ids_sim, dtype=torch.int32, device=self.device)
        self.hammer_actor_ids_sim = torch.tensor(
            self.hammer_actor_ids_sim, dtype=torch.int32, device=self.device)
        self.nail_actor_ids_sim = torch.tensor(
            self.nail_actor_ids_sim, dtype=torch.int32, device=self.device)

        # For extracting root pos/quat
        self.hammer_actor_id_env = self.gym.find_actor_index(
            env_ptr, 'hammer', gymapi.DOMAIN_ENV)
        self.nail_actor_id_env = self.gym.find_actor_index(
            env_ptr, 'nail', gymapi.DOMAIN_ENV)

    def _acquire_env_tensors(self):
        """Acquire and wrap tensors. Create views."""
        super()._acquire_env_tensors()

        self.nail_pos = self.root_pos[:, self.nail_actor_id_env, 0:3]
        self.nail_quat = self.root_quat[:, self.nail_actor_id_env, 0:4]

        self.to_nail_pos = self.nail_pos - self.hammer_pos
        self.to_nail_quat = quat_mul(
            self.nail_quat, quat_conjugate(self.hammer_quat))

    def refresh_env_tensors(self):
        """Refresh tensors."""
        # NOTE: Tensor refresh functions should be called once per step, before
        # setters.
        # NOTE: Since the drill_pos, drill_quat, etc. are obtained from the
        # root state tensor through regular slicing, they are views rather than
        # separate tensors and hence don't have to be updated separately.
        self.to_nail_pos[:] = self.nail_pos - self.hammer_pos
        self.to_nail_quat[:] = quat_mul(
            self.nail_quat, quat_conjugate(self.hammer_quat))

    def visualize_hammer_pose(self, env_id: int, axis_length: float = 0.3
                              ) -> None:
        self.visualize_body_pose("hammer", env_id, axis_length)

    def visualize_nail_pose(self, env_id: int, axis_length: float = 0.3
                            ) -> None:
        self.visualize_body_pose("nail", env_id, axis_length)
