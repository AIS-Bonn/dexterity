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

"""Dexterity: class for drill env.

Inherits base class and abstract environment class. Inherited by drill task
classes. Not directly executed.

Configuration defined in DexterityEnvDrill.yaml.
"""

import hydra
import numpy as np
import os
import torch
from typing import *

from isaacgym import gymapi
from isaacgymenvs.tasks.dexterity.base.base import DexterityBase
from isaacgymenvs.tasks.dexterity.env.schema_class_env import DexterityABCEnv
from isaacgymenvs.tasks.dexterity.env.schema_config_env import \
    DexteritySchemaConfigEnv


class DexterityEnvDrill(DexterityBase, DexterityABCEnv):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless,
                 virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize environment superclass.
        Acquire tensors."""

        self._get_env_yaml_params()

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

        config_path = 'task/DexterityEnvDrill.yaml'  # relative to cfg dir
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
        drill_assets, drill_site_assets = self._import_env_assets()

        self._create_actors(lower, upper, num_per_row, robot_asset, table_asset,
                            drill_assets, drill_site_assets)

    def _import_env_assets(self):
        """Set object assets options. Import objects."""
        return self._import_drill_assets()

    def _import_drill_assets(self):
        asset_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '../..', '..', '..',
                         'assets', 'dexterity', 'tools', 'drills'))
        asset_files = self.cfg_env['env']['drills']
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False

        # While I have not specified asset inertia manually yet
        asset_options.override_com = True
        asset_options.override_inertia = True

        asset_options.vhacd_enabled = True  # Enable convex decomposition
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 1000000

        site_options = gymapi.AssetOptions()
        site_options.fix_base_link = True

        drill_assets, drill_site_assets = [], []
        for asset_file in asset_files:
            drill_asset = self.gym.load_asset(self.sim, asset_root, asset_file,
                                              asset_options)
            drill_assets.append(drill_asset)
            drill_site_asset = self.gym.load_asset(
                self.sim, asset_root,
                os.path.join(os.path.dirname(asset_file), "site.urdf"),
                site_options)
            drill_site_assets.append(drill_site_asset)
        return drill_assets, drill_site_assets

    def _create_actors(self, lower: gymapi.Vec3, upper: gymapi.Vec3,
                       num_per_row: int, robot_asset, table_asset,
                       drill_assets, drill_site_assets) -> None:
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

        drill_pose = gymapi.Transform()
        drill_site_pose = gymapi.Transform()

        self.env_ptrs = []
        self.robot_handles = []
        self.table_handles = []
        self.drill_handles = []
        self.drill_site_handles = []
        self.shape_ids = []
        self.robot_actor_ids_sim = []  # within-sim indices
        self.table_actor_ids_sim = []  # within-sim indices
        self.drill_actor_ids_sim = []  # within-sim indices
        self.drill_site_actor_ids_sim = []  # within-sim indices
        actor_count = 0

        drill_count = len(self.cfg_env['env']['drills'])

        for i in range(self.num_envs):
            # Create new env
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Loop through all used drills
            drill_idx = i % drill_count
            used_drill = drill_assets[drill_idx]
            used_drill_site = drill_site_assets[drill_idx]

            # Aggregate all actors
            if self.cfg_base.sim.aggregate_mode > 1:
                max_rigid_bodies = \
                    self.gym.get_asset_rigid_body_count(used_drill) + \
                    self.gym.get_asset_rigid_body_count(used_drill_site) + \
                    self.robot.rigid_body_count + \
                    int(self.cfg_base.env.has_table)
                max_rigid_shapes = \
                    self.gym.get_asset_rigid_shape_count(used_drill) + \
                    self.gym.get_asset_rigid_shape_count(used_drill_site) + \
                    self.robot.rigid_shape_count + \
                    int(self.cfg_base.env.has_table)
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

            # Aggregate task-specific actors (objects)
            if self.cfg_base.sim.aggregate_mode == 1:
                max_rigid_bodies = \
                    self.gym.get_asset_rigid_body_count(used_drill) + \
                    self.gym.get_asset_rigid_body_count(used_drill_site)
                max_rigid_shapes = \
                    self.gym.get_asset_rigid_shape_count(used_drill) + \
                    self.gym.get_asset_rigid_shape_count(used_drill_site)
                self.gym.begin_aggregate(env_ptr, max_rigid_bodies,
                                         max_rigid_shapes, True)

            # Create drill actor
            drill_handle = self.gym.create_actor(
                env_ptr, used_drill, drill_pose, 'drill', i, 0, 2)
            self.drill_actor_ids_sim.append(actor_count)
            self.drill_handles.append(drill_handle)
            actor_count += 1

            # Create drill site actor (used to visualize target pose)
            drill_site_handle = self.gym.create_actor(
                env_ptr, used_drill_site, drill_site_pose, 'drill_site', i, 0,
                0)
            self.drill_site_actor_ids_sim.append(actor_count)
            self.drill_site_handles.append(drill_site_handle)
            for rigid_body_idx in range(
                    self.gym.get_asset_rigid_body_count(used_drill_site)):
                self.gym.set_rigid_body_color(
                    env_ptr, drill_site_handle, rigid_body_idx,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(*self.cfg_env['env']['drill_target_color']))
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
        self.drill_actor_ids_sim = torch.tensor(
            self.drill_actor_ids_sim, dtype=torch.int32, device=self.device)
        self.drill_site_actor_ids_sim = torch.tensor(
            self.drill_site_actor_ids_sim, dtype=torch.int32,
            device=self.device)

        # For extracting root pos/quat
        self.drill_actor_id_env = self.gym.find_actor_index(
            env_ptr, 'drill', gymapi.DOMAIN_ENV)
        self.drill_site_actor_id_env = self.gym.find_actor_index(
            env_ptr, 'drill_site', gymapi.DOMAIN_ENV)

    def _acquire_env_tensors(self):
        """Acquire and wrap tensors. Create views."""
        self.drill_pos = self.root_pos[:, self.drill_actor_id_env, 0:3]
        self.drill_quat = self.root_quat[:, self.drill_actor_id_env, 0:4]
        self.drill_linvel = self.root_linvel[:, self.drill_actor_id_env, 0:3]
        self.drill_angvel = self.root_angvel[:, self.drill_actor_id_env, 0:3]

        self.drill_target_pos = self.root_pos[:, self.drill_site_actor_id_env, 0:3]
        self.drill_target_quat = self.root_quat[:, self.drill_site_actor_id_env, 0:4]

    def refresh_env_tensors(self):
        """Refresh tensors."""
        # NOTE: Tensor refresh functions should be called once per step, before
        # setters.
        # NOTE: Since the drill_pos, drill_quat, etc. are obtained from the
        # root state tensor through regular slicing, they are views rather than
        # separate tensors and hence don't have to be updated separately.
        pass

    def _get_random_drop_pos(self, env_ids) -> torch.Tensor:
        drill_pos_drop = torch.tensor(
            self.cfg_task.randomize.drill_pos_drop, device=self.device
        ).unsqueeze(0).repeat(len(env_ids), 1)
        drill_pos_drop_noise = \
            2 * (torch.rand((len(env_ids), 3), dtype=torch.float32,
                            device=self.device) - 0.5)  # [-1, 1]
        drill_pos_drop_noise = drill_pos_drop_noise @ torch.diag(torch.tensor(
            self.cfg_task.randomize.drill_pos_drop_noise, device=self.device))
        drill_pos_drop += drill_pos_drop_noise
        return drill_pos_drop

    def _get_random_target_pos(self, env_ids) -> torch.Tensor:
        drill_pos_target = torch.tensor(
            self.cfg_task.randomize.drill_pos_target, device=self.device
        ).unsqueeze(0).repeat(len(env_ids), 1)
        drill_pos_target_noise = \
            2 * (torch.rand((len(env_ids), 3), dtype=torch.float32,
                            device=self.device) - 0.5)  # [-1, 1]
        drill_pos_target_noise = drill_pos_target_noise @ torch.diag(torch.tensor(
            self.cfg_task.randomize.drill_pos_target_noise, device=self.device))
        drill_pos_target += drill_pos_target_noise
        return drill_pos_target

    def _get_random_target_quat(self, env_ids) -> torch.Tensor:
        drill_quat_target = torch.tensor(
            [[0, 0, 0, 1]], dtype=torch.float,
            device=self.device).repeat(len(env_ids), 1)
        return drill_quat_target

    def visualize_drill_pose(self, env_id: int, axis_length: float = 0.3
                             ) -> None:
        self.visualize_body_pose("drill", env_id, axis_length)

    def visualize_drill_target_pose(self, env_id: int, axis_length: float = 0.3
                                    ) -> None:
        self.visualize_body_pose("drill_target", env_id, axis_length)
