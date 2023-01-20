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
from isaacgymenvs.tasks.dexterity.base.base import DexterityBase
from isaacgymenvs.tasks.dexterity.env.schema_class_env import DexterityABCEnv
from isaacgymenvs.tasks.dexterity.env.schema_config_env import \
    DexteritySchemaConfigEnv
from isaacgymenvs.tasks.dexterity.env.object import randomize_rotation


class DexterityEnvHammer(DexterityBase, DexterityABCEnv):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless,
                 virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize environment superclass.
        Acquire tensors."""

        self._get_env_yaml_params()
        self.parse_camera_spec(cfg)

        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

        self.acquire_base_tensors()  # defined in superclass
        self._acquire_env_tensors()
        self.refresh_base_tensors()  # defined in superclass
        self.refresh_env_tensors()

    def _get_env_yaml_params(self):
        """Initialize instance variables from YAML files."""
        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='dexterity_schema_config_env',
                 node=DexteritySchemaConfigEnv)

        config_path = 'task/DexterityEnvHammer.yaml'  # relative to cfg dir
        self.cfg_env = hydra.compose(config_name=config_path)
        self.cfg_env = self.cfg_env['task']  # strip superfluous nesting

    def create_envs(self):
        """Set env options. Import assets. Create actors."""

        lower = gymapi.Vec3(-self.cfg_base.env.env_spacing,
                            -self.cfg_base.env.env_spacing, 0.0)
        upper = gymapi.Vec3(self.cfg_base.env.env_spacing,
                            self.cfg_base.env.env_spacing,
                            self.cfg_base.env.env_spacing)
        num_per_row = int(np.sqrt(self.num_envs))

        robot_asset, table_asset = self.import_robot_assets()
        hammer_assets, nail_asset = self._import_env_assets()

        self._create_actors(lower, upper, num_per_row, robot_asset, table_asset,
                            hammer_assets, nail_asset)

    def _import_env_assets(self):
        """Set object assets options. Import objects."""
        hammer_assets = self._import_hammer_assets()
        nail_asset = self._import_nail_asset()
        return hammer_assets, nail_asset

    def _import_hammer_assets(self):
        asset_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '../..', '..', '..',
                         'assets', 'dexterity', 'tools', 'hammers'))
        asset_files = self.cfg_env['env']['hammers']
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False

        # While I have not specified asset inertia manually yet
        asset_options.override_com = True
        asset_options.override_inertia = True

        asset_options.vhacd_enabled = True  # Enable convex decomposition
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 1000000

        hammer_assets = []
        for asset_file in asset_files:
            hammer_asset = self.gym.load_asset(self.sim, asset_root, asset_file,
                                               asset_options)
            hammer_assets.append(hammer_asset)
        return hammer_assets

    def _import_nail_asset(self):
        asset_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '../..', '..', '..',
                         'assets', 'dexterity', 'tools', 'hammers'))
        asset_file = 'nail.urdf'
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True

        nail_asset = self.gym.load_asset(self.sim, asset_root, asset_file,
                                         asset_options)
        return nail_asset

    def _create_actors(self, lower: gymapi.Vec3, upper: gymapi.Vec3,
                       num_per_row: int, robot_asset, table_asset,
                       hammer_assets, nail_asset) -> None:
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
        self.robot_handles = []
        self.table_handles = []
        self.nail_handles = []
        self.hammer_handles = []
        self.shape_ids = []
        self.robot_actor_ids_sim = []  # within-sim indices
        self.table_actor_ids_sim = []  # within-sim indices
        self.nail_actor_ids_sim = []  # within-sim indices
        self.hammer_actor_ids_sim = []  # within-sim indices
        actor_count = 0

        hammer_count = len(self.cfg_env['env']['hammers'])

        for i in range(self.num_envs):
            # Create new env
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Loop through all used hammers
            hammer_idx = i % hammer_count
            used_hammer = hammer_assets[hammer_idx]

            # Aggregate all actors
            if self.cfg_base.sim.aggregate_mode > 1:
                max_rigid_bodies = \
                    self.gym.get_asset_rigid_body_count(used_hammer) + \
                    self.gym.get_asset_rigid_body_count(nail_asset) + \
                    self.robot.rigid_body_count + \
                    int(self.cfg_base.env.has_table) + \
                    self.camera_rigid_body_count
                max_rigid_shapes = \
                    self.gym.get_asset_rigid_shape_count(used_hammer) + \
                    self.gym.get_asset_rigid_shape_count(nail_asset) + \
                    self.robot.rigid_shape_count + \
                    int(self.cfg_base.env.has_table) + \
                    self.camera_rigid_shape_count
                self.gym.begin_aggregate(env_ptr, max_rigid_bodies,
                                         max_rigid_shapes, True)

            # Create robot actor
            # collision_filter=-1 to use asset collision filters in XML model
            robot_handle = self.gym.create_actor(
                env_ptr, robot_asset, robot_pose, 'robot', i, -1, 0)
            self.robot_actor_ids_sim.append(actor_count)
            self.robot_handles.append(robot_handle)
            # Enable force sensors for robot
            self.gym.enable_actor_dof_force_sensors(env_ptr, robot_handle)
            actor_count += 1

            # Create table actor
            if self.cfg_base.env.has_table:
                table_handle = self.gym.create_actor(
                    env_ptr, table_asset, table_pose, 'table', i, 0, 1)
                self.table_actor_ids_sim.append(actor_count)
                actor_count += 1

                # Set table shape properties
                table_shape_props = self.gym.get_actor_rigid_shape_properties(
                    env_ptr, table_handle)
                table_shape_props[0].friction = self.cfg_base.env.table_friction
                table_shape_props[0].rolling_friction = 0.0  # default = 0.0
                table_shape_props[0].torsion_friction = 0.0  # default = 0.0
                table_shape_props[0].restitution = 0.0  # default = 0.0
                table_shape_props[0].compliance = 0.0  # default = 0.0
                table_shape_props[0].thickness = 0.0  # default = 0.0
                self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle,
                                                          table_shape_props)
                self.table_handles.append(table_handle)

            # Create camera actors
            if "cameras" in self.cfg_env.keys():
                self.create_camera_actors(env_ptr, i)
                actor_count += self.camera_count

            # Aggregate task-specific actors (objects)
            if self.cfg_base.sim.aggregate_mode == 1:
                max_rigid_bodies = \
                    self.gym.get_asset_rigid_body_count(used_hammer) + \
                    self.gym.get_asset_rigid_body_count(nail_asset)
                max_rigid_shapes = \
                    self.gym.get_asset_rigid_shape_count(used_hammer) + \
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
                env_ptr, used_hammer, hammer_pose, 'hammer', i, 0, 2)
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
        self.hammer_pos = self.root_pos[:, self.hammer_actor_id_env, 0:3]
        self.hammer_quat = self.root_quat[:, self.hammer_actor_id_env, 0:4]
        self.hammer_linvel = self.root_linvel[:, self.hammer_actor_id_env, 0:3]
        self.hammer_angvel = self.root_angvel[:, self.hammer_actor_id_env, 0:3]

        self.nail_pos = self.root_pos[:, self.nail_actor_id_env, 0:3]
        self.nail_quat = self.root_quat[:, self.nail_actor_id_env, 0:4]
        self.nail_linvel = self.root_linvel[:, self.nail_actor_id_env, 0:3]
        self.nail_angvel = self.root_angvel[:, self.nail_actor_id_env, 0:3]

    def refresh_env_tensors(self):
        """Refresh tensors."""
        # NOTE: Tensor refresh functions should be called once per step, before
        # setters.
        # NOTE: Since the drill_pos, drill_quat, etc. are obtained from the
        # root state tensor through regular slicing, they are views rather than
        # separate tensors and hence don't have to be updated separately.
        pass

    def _get_random_drop_pos(self, env_ids) -> torch.Tensor:
        hammer_pos_drop = torch.tensor(
            self.cfg_task.randomize.hammer_pos_drop, device=self.device
        ).unsqueeze(0).repeat(len(env_ids), 1)
        hammer_pos_drop_noise = \
            2 * (torch.rand((len(env_ids), 3), dtype=torch.float32,
                            device=self.device) - 0.5)  # [-1, 1]
        hammer_pos_drop_noise = hammer_pos_drop_noise @ torch.diag(torch.tensor(
            self.cfg_task.randomize.hammer_pos_drop_noise, device=self.device))
        hammer_pos_drop += hammer_pos_drop_noise
        return hammer_pos_drop

    def _get_random_drop_quat(self, env_ids) -> torch.Tensor:
        x_unit_tensor = to_torch(
            [1, 0, 0], dtype=torch.float, device=self.device).repeat(
            (len(env_ids), 1))
        y_unit_tensor = to_torch(
            [0, 1, 0], dtype=torch.float, device=self.device).repeat(
            (len(env_ids), 1))
        rand_floats = torch_rand_float(
            -1.0, 1.0, (len(env_ids), 2), device=self.device)
        hammer_quat_drop = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1], x_unit_tensor, y_unit_tensor)
        return hammer_quat_drop

    def _hammer_in_workspace(self, hammer_pos) -> torch.Tensor:
        x_lower = self.cfg_task.randomize.workspace_extent_xy[0][0]
        x_upper = self.cfg_task.randomize.workspace_extent_xy[1][0]
        y_lower = self.cfg_task.randomize.workspace_extent_xy[0][1]
        y_upper = self.cfg_task.randomize.workspace_extent_xy[1][1]
        hammer_in_workspace = x_lower <= hammer_pos[:, 0]
        hammer_in_workspace = torch.logical_and(
            hammer_in_workspace, hammer_pos[:, 0] <= x_upper)
        hammer_in_workspace = torch.logical_and(
            hammer_in_workspace, y_lower <= hammer_pos[:, 1])
        hammer_in_workspace = torch.logical_and(
            hammer_in_workspace, hammer_pos[:, 1] <= y_upper)
        return hammer_in_workspace

    def visualize_hammer_pose(self, env_id: int, axis_length: float = 0.3
                              ) -> None:
        self.visualize_body_pose("hammer", env_id, axis_length)

    def visualize_nail_pose(self, env_id: int, axis_length: float = 0.3
                            ) -> None:
        self.visualize_body_pose("nail", env_id, axis_length)

    def visualize_workspace_xy(self, env_id: int) -> None:
        # Set extent in z-direction to 0
        lower = self.cfg_task.randomize.workspace_extent_xy[0] + [0.]
        upper = self.cfg_task.randomize.workspace_extent_xy[1] + [0.]
        extent = torch.tensor([lower, upper])
        drop_pose = gymapi.Transform(
            p=gymapi.Vec3(self.cfg_task.randomize.hammer_pos_drop[0],
                          self.cfg_task.randomize.hammer_pos_drop[1],
                          0.))
        bbox = gymutil.WireframeBBoxGeometry(extent, pose=drop_pose,
                                             color=(0, 1, 1))
        gymutil.draw_lines(bbox, self.gym, self.viewer, self.env_ptrs[env_id],
                           pose=gymapi.Transform())
