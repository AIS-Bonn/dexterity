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

"""Dexterity: class for mug env.

Inherits base class and abstract environment class. Inherited by mug task
classes. Not directly executed.

Configuration defined in DexterityEnvMug.yaml.
"""

import hydra
import numpy as np
import os

import pycpd.rigid_registration
from isaacgym.torch_utils import *
from typing import *

from isaacgym import gymapi
from isaacgymenvs.tasks.dexterity.env.tool_use import DexterityEnvToolUse
from isaacgymenvs.tasks.dexterity.env.schema_config_env import \
    DexteritySchemaConfigEnv
from .object import randomize_rotation


class DexterityEnvMug(DexterityEnvToolUse):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless,
                 virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize environment superclass.
        Acquire tensors."""

        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render,
                         tool_category="mug")

    def create_envs(self):
        """Set env options. Import assets. Create actors."""

        lower = gymapi.Vec3(-self.cfg_base.env.env_spacing,
                            -self.cfg_base.env.env_spacing, 0.0)
        upper = gymapi.Vec3(self.cfg_base.env.env_spacing,
                            self.cfg_base.env.env_spacing,
                            self.cfg_base.env.env_spacing)
        num_per_row = int(np.sqrt(self.num_envs))

        robot_asset, table_asset = self.import_robot_assets()
        source_mug, target_mugs = self._import_tool_assets()
        shelf_asset = self._import_shelf_asset()

        self._create_actors(lower, upper, num_per_row, robot_asset, table_asset,
                            source_mug, target_mugs, shelf_asset)

    def _acquire_env_tensors(self):
        super()._acquire_env_tensors()

        # Acquire shelf pose
        self.shelf_pos = self.root_pos[:, self.shelf_actor_id_env, 0:3]
        self.shelf_quat = self.root_quat[:, self.shelf_actor_id_env, 0:4]

        self.to_shelf_pos = self.shelf_pos - self.mug_pos
        self.to_shelf_quat = quat_mul(
            self.shelf_quat, quat_conjugate(self.mug_quat))

        # Buffer for success metric.
        self.on_shelf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool)

    def refresh_env_tensors(self):
        """Refresh tensors."""
        # NOTE: Tensor refresh functions should be called once per step, before
        # setters.
        # NOTE: Since the drill_pos, drill_quat, etc. are obtained from the
        # root state tensor through regular slicing, they are views rather than
        # separate tensors and hence don't have to be updated separately.
        self.to_shelf_pos[:] = self.shelf_pos - self.mug_pos
        self.to_shelf_quat[:] = quat_mul(
            self.shelf_quat, quat_conjugate(self.mug_quat))

    def _import_shelf_asset(self):
        asset_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '../..', '..', '..',
                         'assets', 'dexterity', 'tools', 'shelf'))
        asset_file = 'rigid.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.use_mesh_materials = True
        mug_rack_asset = self.gym.load_asset(self.sim, asset_root, asset_file,
                                             asset_options)
        return mug_rack_asset

    def _create_actors(self, lower: gymapi.Vec3, upper: gymapi.Vec3,
                       num_per_row: int, robot_asset, table_asset,
                       source_mug, target_mugs, shelf_asset) -> None:

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

        mug_pose = gymapi.Transform()
        shelf_pose = gymapi.Transform(p=gymapi.Vec3(0.125, 0, 0.25))

        self.env_ptrs = []
        self.mug_handles = []
        self.mug_actor_ids_sim = []  # within-sim indices
        self.mug_site_actor_ids_sim = []  # within-sim indices

        mugs_list = [source_mug, ] + target_mugs

        actor_count = 0
        mug_count = len(mugs_list)
        for i in range(self.num_envs):
            # Create new env
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Loop through all used mugs
            mug_idx = i % mug_count
            used_mug = mugs_list[mug_idx]

            # Aggregate all actors
            if self.cfg_base.sim.aggregate_mode > 1:
                max_rigid_bodies = self.base_rigid_bodies + \
                    used_mug.rigid_body_count + \
                    self.gym.get_asset_rigid_body_count(shelf_asset)
                max_rigid_shapes = self.base_rigid_shapes + \
                    used_mug.rigid_shape_count + \
                    self.gym.get_asset_rigid_shape_count(shelf_asset)
                self.gym.begin_aggregate(env_ptr, max_rigid_bodies,
                                         max_rigid_shapes, True)

            # Create common actors (robot, table, cameras)
            actor_count = self.create_base_actors(
                env_ptr, i, actor_count, robot_asset, robot_pose,
                table_asset, table_pose)

            # Aggregate task-specific actors (mug and shelf)
            if self.cfg_base.sim.aggregate_mode == 1:
                max_rigid_bodies = \
                    used_mug.rigid_body_count + \
                    self.gym.get_asset_rigid_body_count(shelf_asset)
                max_rigid_shapes = \
                    used_mug.rigid_shape_count + \
                    self.gym.get_asset_rigid_shape_count(shelf_asset)
                self.gym.begin_aggregate(env_ptr, max_rigid_bodies,
                                         max_rigid_shapes, True)

            # Create mug actor
            mug_handle = self.gym.create_actor(
                env_ptr, used_mug.asset, mug_pose, 'mug', i, 0, 2)
            self.mug_actor_ids_sim.append(actor_count)
            self.mug_handles.append(mug_handle)
            actor_count += 1

            # Create mug_rack actor
            shelf_handle = self.gym.create_actor(
                env_ptr, shelf_asset, shelf_pose, 'shelf', i, 0, 3)
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
        self.mug_actor_ids_sim = torch.tensor(
            self.mug_actor_ids_sim, dtype=torch.int32, device=self.device)

        # For extracting root pos/quat
        self.mug_actor_id_env = self.gym.find_actor_index(
            env_ptr, 'mug', gymapi.DOMAIN_ENV)
        self.shelf_actor_id_env = self.gym.find_actor_index(
            env_ptr, 'shelf', gymapi.DOMAIN_ENV)

    def visualize_mug_pose(self, env_id: int, axis_length: float = 0.3
                           ) -> None:
        self.visualize_body_pose("mug", env_id, axis_length)

    def visualize_shelf_pose(self, env_id: int, axis_length: float = 0.3
                             ) -> None:
        self.visualize_body_pose("shelf", env_id, axis_length)
