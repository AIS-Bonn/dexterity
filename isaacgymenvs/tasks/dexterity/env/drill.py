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

import pycpd.rigid_registration
from isaacgym.torch_utils import *
from typing import *

from isaacgym import gymapi
from isaacgymenvs.tasks.dexterity.base.base import DexterityBase
from isaacgymenvs.tasks.dexterity.env.tool_use import DexterityEnvToolUse
from .object import randomize_rotation


class DexterityEnvDrill(DexterityEnvToolUse):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless,
                 virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize environment superclass.
        Acquire tensors."""

        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render,
                         tool_category="drill")

    def create_envs(self):
        """Set env options. Import assets. Create actors."""

        lower = gymapi.Vec3(-self.cfg_base.env.env_spacing,
                            -self.cfg_base.env.env_spacing, 0.0)
        upper = gymapi.Vec3(self.cfg_base.env.env_spacing,
                            self.cfg_base.env.env_spacing,
                            self.cfg_base.env.env_spacing)
        num_per_row = int(np.sqrt(self.num_envs))

        robot_asset, table_asset = self.import_robot_assets()
        source_drill, target_drills = self._import_tool_assets()
        source_drill_site_asset, target_drill_site_assets = self._import_drill_site_assets()

        self._create_actors(lower, upper, num_per_row, robot_asset, table_asset,
                            source_drill, target_drills, source_drill_site_asset, target_drill_site_assets)

    def _import_drill_site_assets(self):
        asset_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '../..', '..', '..',
                         'assets', 'dexterity', 'tools', 'drill'))
        source_asset_file = self.cfg_env['env']['canonical']
        target_asset_files = self.cfg_env['env'][self.tool_category + "s"]
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True

        source_asset_file += '/site.urdf'
        target_asset_files = [t + '/site.urdf' for t in target_asset_files]

        source_drill_site_asset = self.gym.load_asset(
                self.sim, asset_root, source_asset_file,asset_options)

        target_drill_site_assets = []
        for asset_file in target_asset_files:
            target_drill_site_asset = self.gym.load_asset(
                self.sim, asset_root, asset_file, asset_options)
            target_drill_site_assets.append(target_drill_site_asset)
        return source_drill_site_asset, target_drill_site_assets

    def _create_actors(self, lower: gymapi.Vec3, upper: gymapi.Vec3,
                       num_per_row: int, robot_asset, table_asset,
                       source_drill, target_drills,
                       source_drill_site_asset, target_drill_site_assets) -> None:
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
        self.drill_handles = []
        self.drill_site_handles = []
        self.drill_actor_ids_sim = []  # within-sim indices
        self.drill_site_actor_ids_sim = []  # within-sim indices

        drills_list = [source_drill, ] + target_drills
        drill_sites_list = [source_drill_site_asset, ] + target_drill_site_assets

        actor_count = 0
        drill_count = len(self.cfg_env['env']['drills']) + 1
        for i in range(self.num_envs):
            # Create new env
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Loop through all used drills
            drill_idx = i % drill_count
            used_drill = drills_list[drill_idx]
            used_drill_site = drill_sites_list[drill_idx]

            # Aggregate all actors
            if self.cfg_base.sim.aggregate_mode > 1:
                max_rigid_bodies = self.base_rigid_bodies + \
                    used_drill.rigid_body_count + \
                    self.gym.get_asset_rigid_body_count(used_drill_site)
                max_rigid_shapes = self.base_rigid_shapes + \
                    used_drill.rigid_shape_count + \
                    self.gym.get_asset_rigid_shape_count(used_drill_site)
                self.gym.begin_aggregate(env_ptr, max_rigid_bodies,
                                         max_rigid_shapes, True)

            # Create common actors (robot, table, cameras)
            actor_count = self.create_base_actors(
                env_ptr, i, actor_count, robot_asset, robot_pose,
                table_asset, table_pose)

            # Aggregate task-specific actors (drills and drill target sites)
            if self.cfg_base.sim.aggregate_mode == 1:
                max_rigid_bodies = \
                    used_drill.rigid_body_count + \
                    self.gym.get_asset_rigid_body_count(used_drill_site)
                max_rigid_shapes = \
                    used_drill.rigid_shape_count + \
                    self.gym.get_asset_rigid_shape_count(used_drill_site)
                self.gym.begin_aggregate(env_ptr, max_rigid_bodies,
                                         max_rigid_shapes, True)

            # Create drill actor
            drill_handle = self.gym.create_actor(
                env_ptr, used_drill.asset, drill_pose, 'drill', i, 0, 2)
            self.drill_actor_ids_sim.append(actor_count)
            self.drill_handles.append(drill_handle)
            actor_count += 1

            # Create drill site actor (used to visualize target pose)
            drill_site_handle = self.gym.create_actor(
                env_ptr, used_drill_site, drill_site_pose, 'drill_site', i, 0,
                2)
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

        self.to_drill_target_pos = self.drill_target_pos - self.drill_pos
        self.to_drill_target_quat = quat_mul(
            self.drill_quat, quat_conjugate(self.drill_target_quat))

    def refresh_env_tensors(self):
        """Refresh tensors."""
        # NOTE: Tensor refresh functions should be called once per step, before
        # setters.
        # NOTE: Since the drill_pos, drill_quat, etc. are obtained from the
        # root state tensor through regular slicing, they are views rather than
        # separate tensors and hence don't have to be updated separately.
        self.to_drill_target_pos[:] = self.drill_target_pos - self.drill_pos
        self.to_drill_target_quat[:] = quat_mul(
            self.drill_quat, quat_conjugate(self.drill_target_quat))

    def visualize_drill_pose(self, env_id: int, axis_length: float = 0.3
                             ) -> None:
        self.visualize_body_pose("drill", env_id, axis_length)

    def visualize_drill_target_pose(self, env_id: int, axis_length: float = 0.3
                                    ) -> None:
        self.visualize_body_pose("drill_target", env_id, axis_length)
