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

"""Dexterity: class for object bin.

Inherits base class and abstract environment class. Inherited by object task
classes. Not directly executed.

Configuration defined in DexterityEnvBin.yaml. Asset info defined in
asset_info_object_sets.yaml.
"""

import glob
import hydra
import numpy as np
import os
import torch
from typing import *
import random

from isaacgym import gymapi, gymtorch, torch_utils, gymutil
from isaacgymenvs.tasks.dexterity.env.schema_config_env import \
    DexteritySchemaConfigEnv


from isaacgymenvs.tasks.dexterity.env.object import DexterityObject, DexterityEnvObject


class DexterityEnvBin(DexterityEnvObject):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless,
                 virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize environment superclass.
        Acquire tensors."""
        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

    def _get_env_yaml_params(self):
        """Initialize instance variables from YAML files."""
        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='dexterity_schema_config_env',
                 node=DexteritySchemaConfigEnv)

        config_path = 'task/DexterityEnvBin.yaml'  # relative to cfg dir
        self.cfg_env = hydra.compose(config_name=config_path)
        self.cfg_env = self.cfg_env['task']  # strip superfluous nesting

        bin_info_path = '../../assets/dexterity/bin/bin_info.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.bin_info = hydra.compose(config_name=bin_info_path)
        self.bin_info = self.bin_info['']['']['']['']['']['']['assets']['dexterity']['bin']  # strip superfluous nesting

        self.object_sets_asset_root = os.path.normpath(os.path.join(
            os.path.dirname(__file__),
            self.cfg_env['env']['object_sets_asset_root']))

    def _import_env_assets(self):
        """Set object assets options. Import objects."""
        # Import objects assets like in the single object task
        object_assets = super()._import_env_assets()

        # Import bin asset
        bin_asset_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..', '..',
                         'assets', 'dexterity', 'bin'))
        bin_asset_file = 'bin.urdf'
        bin_options = gymapi.AssetOptions()
        bin_options.fix_base_link = True
        bin_options.use_mesh_materials = True
        bin_options.vhacd_enabled = True  # Enable convex decomposition
        bin_options.vhacd_params = gymapi.VhacdParams()
        bin_options.vhacd_params.resolution = 1000000
        bin_asset = self.gym.load_asset(self.sim, bin_asset_root,
                                        bin_asset_file, bin_options)
        return (object_assets, bin_asset)

    def _create_actors(self, lower: gymapi.Vec3, upper: gymapi.Vec3,
                       num_per_row: int, robot_asset, table_asset,
                       env_assets) -> None:
        object_assets, bin_asset = env_assets
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

        bin_pose = gymapi.Transform(
            p=gymapi.Vec3(*self.cfg_env['env']['bin_pos']),
            r=gymapi.Quat(*self.cfg_env['env']['bin_quat']))

        self.env_ptrs = []
        self.object_handles = [[] for _ in range(self.num_envs)]
        self.bin_handles = []
        self.object_actor_ids_sim = [[] for _ in range(self.num_envs)]  # within-sim indices
        self.bin_actor_ids_sim = []  # within-sim indices

        actor_count = 0
        for i in range(self.num_envs):
            # Create new env
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Select subset of size num_objects from all available objects
            assert self.object_count >= self.cfg_env['env']['num_objects'], \
                "Number of objects per bin cannot be larger that the total " \
                "number of objects used."
            objects_idx = random.sample(list(range(self.object_count)),
                                        self.cfg_env['env']['num_objects'])
            # Get rigid body and shape count of selected subset
            objects_rigid_body_count = sum(
                [o.rigid_body_count for o in [self.objects[i] for i in
                 objects_idx]])
            objects_rigid_shape_count = sum(
                [o.rigid_shape_count for o in [self.objects[i] for i in
                 objects_idx]])

            # Get rigid body and shape count for bin asset
            bin_rigid_body_count = self.gym.get_asset_rigid_body_count(
                bin_asset)
            bin_rigid_shape_count = self.gym.get_asset_rigid_shape_count(
                bin_asset)

            # Aggregate all actors
            if self.cfg_base.sim.aggregate_mode > 1:
                max_rigid_bodies = self.base_rigid_bodies + \
                    objects_rigid_body_count + \
                    bin_rigid_body_count
                max_rigid_shapes = self.base_rigid_shapes + \
                    objects_rigid_shape_count + \
                    bin_rigid_shape_count
                self.gym.begin_aggregate(env_ptr, max_rigid_bodies,
                                         max_rigid_shapes, True)

            # Create common actors (robot, table, cameras)
            actor_count = self.create_base_actors(
                env_ptr, i, actor_count, robot_asset, robot_pose,
                table_asset, table_pose)

            # Aggregate task-specific actors (objects and bin)
            if self.cfg_base.sim.aggregate_mode == 1:
                max_rigid_bodies = \
                    objects_rigid_shape_count + \
                    bin_rigid_shape_count
                max_rigid_shapes = \
                    objects_rigid_shape_count + \
                    bin_rigid_shape_count
                self.gym.begin_aggregate(env_ptr, max_rigid_bodies,
                                         max_rigid_shapes, True)

            # Create bin actor
            bin_handle = self.gym.create_actor(
                env_ptr, bin_asset, bin_pose, 'bin', i, 0, 2)
            self.bin_actor_ids_sim.append(actor_count)
            self.bin_handles.append(bin_handle)
            actor_count += 1

            # Create object actors
            for object_idx in objects_idx:
                used_object = self.objects[object_idx]
                object_handle = self.gym.create_actor(
                    env_ptr, used_object.asset, used_object.start_pose,
                    used_object.name, i, 0, 3)
                self.object_actor_ids_sim[i].append(actor_count)
                self.object_handles[i].append(object_handle)
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
        self.object_actor_ids_sim = torch.tensor(
            self.object_actor_ids_sim, dtype=torch.int32, device=self.device)

        # For extracting root pos/quat
        self.robot_actor_id_env = self.gym.find_actor_index(
            env_ptr, 'robot', gymapi.DOMAIN_ENV)
        self.object_actor_id_env = [self.gym.find_actor_index(
            env_ptr, o.name, gymapi.DOMAIN_ENV)
            for o in [self.objects[idx] for idx in objects_idx]]

        # For bookkeeping of target object
        self.target_object_id = torch.zeros(self.num_envs, dtype=torch.int64,
                                            device=self.device)
        self.target_object_actor_id_env = torch.zeros(
            self.num_envs, dtype=torch.int64, device=self.device)

    def _acquire_env_tensors(self):
        """Acquire and wrap tensors. Create views."""
        self.object_pos = self.root_pos[:, self.object_actor_id_env, 0:3]
        self.object_quat = self.root_quat[:, self.object_actor_id_env, 0:4]
        self.object_linvel = self.root_linvel[:, self.object_actor_id_env, 0:3]
        self.object_angvel = self.root_angvel[:, self.object_actor_id_env, 0:3]

        # Init tensors for initial object positions that will be overwritten
        # after objects have been dropped
        self.object_pos_initial = \
            self.root_pos[:, self.object_actor_id_env, 0:3].detach().clone()
        self.object_quat_initial = \
            self.root_quat[:, self.object_actor_id_env, 0:4].detach().clone()

    def refresh_env_tensors(self):
        """Refresh tensors."""
        # NOTE: Tensor refresh functions should be called once per step, before
        # setters.
        # NOTE: Since the object_pos, object_quat etc. are obtained from the
        # root state tensor through advanced slicing, they are separate tensors
        # and hence have to be updated separately.

        # Refresh pose and velocities of all objects
        self.object_pos[:] = self.root_pos[:, self.object_actor_id_env, 0:3]
        self.object_quat[:] = self.root_quat[:, self.object_actor_id_env, 0:4]
        self.object_linvel[:] = self.root_linvel[:, self.object_actor_id_env, 0:3]
        self.object_angvel[:] = self.root_angvel[:, self.object_actor_id_env, 0:3]

        # Refresh pose and velocities of target object
        self.target_object_pos = self.object_pos.gather(
            1, self.target_object_id.unsqueeze(
                1).unsqueeze(2).repeat(1, 1, 3)).squeeze(1)
        self.target_object_pos_initial = self.object_pos_initial.gather(
            1, self.target_object_id.unsqueeze(
                1).unsqueeze(2).repeat(1, 1, 3)).squeeze(1)
        self.target_object_quat = self.object_quat.gather(
            1, self.target_object_id.unsqueeze(
                1).unsqueeze(2).repeat(1, 1, 4)).squeeze(1)
        self.target_object_linvel = self.object_linvel.gather(
            1, self.target_object_id.unsqueeze(
                1).unsqueeze(2).repeat(1, 1, 3)).squeeze(1)
        self.target_object_angvel = self.object_angvel.gather(
            1, self.target_object_id.unsqueeze(
                1).unsqueeze(2).repeat(1, 1, 3)).squeeze(1)

    def _object_in_bin(self, object_pos) -> torch.Tensor:
        x_lower = self.bin_info['extent'][0][0]
        x_upper = self.bin_info['extent'][1][0]
        y_lower = self.bin_info['extent'][0][1]
        y_upper = self.bin_info['extent'][1][1]
        z_lower = self.bin_info['extent'][0][2]
        z_upper = self.bin_info['extent'][1][2]
        in_bin = x_lower <= object_pos[..., 0]
        in_bin = torch.logical_and(in_bin, object_pos[..., 0] <= x_upper)
        in_bin = torch.logical_and(in_bin, y_lower <= object_pos[..., 1])
        in_bin = torch.logical_and(in_bin, object_pos[..., 1] <= y_upper)
        in_bin = torch.logical_and(in_bin, z_lower <= object_pos[..., 2])
        in_bin = torch.logical_and(in_bin, object_pos[..., 2] <= z_upper)
        return in_bin

    def _disable_object_collisions(self, object_ids):
        self._set_object_collisions(object_ids, collision_filter=-1)

    def _enable_object_collisions(self, object_ids):
        self._set_object_collisions(object_ids, collision_filter=0)

    def _set_object_collisions(self, object_ids: List[int],
                               collision_filter: int) -> None:
        # No tensor API to set actor rigid shape props, so a loop is required
        for env_id in range(self.num_envs):
            for object_id in object_ids:
                object_shape_props = self.gym.get_actor_rigid_shape_properties(
                    self.env_ptrs[env_id],
                    self.object_handles[env_id][object_id])
                for shape_id in range(len(object_shape_props)):
                    object_shape_props[shape_id].filter = collision_filter
                self.gym.set_actor_rigid_shape_properties(
                    self.env_ptrs[env_id],
                    self.object_handles[env_id][object_id], object_shape_props)

    def _place_objects_before_bin(self):
        # Place the objects in front of the bin
        self.root_pos[:, self.object_actor_id_env, 0] = 0.5
        self.root_pos[:, self.object_actor_id_env, 1] = 0.0
        self.root_pos[:, self.object_actor_id_env, 2] = 0.5
        self.root_linvel[:, self.object_actor_id_env] = 0.0
        self.root_angvel[:, self.object_actor_id_env] = 0.0

        object_actor_ids_sim = self.object_actor_ids_sim.flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(object_actor_ids_sim),
            len(object_actor_ids_sim))

        for _ in range(1):
            self.gym.simulate(self.sim)
            self.render()

    def visualize_object_pose(self, env_id: int, axis_length: float = 0.3
                              ) -> None:
        """Visualizes the poses of all objects (called by adding 'object_pose'
        to the visualizations in DexterityBase.yaml."""
        for i in range(self.cfg_env['env']['num_objects']):
            self.visualize_body_pose("object", env_id, axis_length, i)

    def visualize_target_object_pose(self, env_id: int, axis_length: float = 0.3
                                     ) -> None:
        """Visualizes the pose of the target objects (called by adding
        'target_object_pose' to the visualizations in DexterityBase.yaml."""
        self.visualize_body_pose("target_object", env_id, axis_length)

    #def visualize_target_object(self, env_id: int) -> None:
    #    """"Highlights the target object by setting the mesh color (called by
    #    adding 'target_object' to the visualizations in DexterityBase.yaml."""
    #    target_object_handle = self.object_handles[env_id][self.target_object_id[env_id]]

    def visualize_bin_extent(self, env_id) -> None:
        extent = torch.tensor(self.bin_info['extent'])
        bin_pose = gymapi.Transform(
            p=gymapi.Vec3(*self.cfg_env['env']['bin_pos']))
        bbox = gymutil.WireframeBBoxGeometry(extent, pose=bin_pose,
                                             color=(0, 1, 1))
        gymutil.draw_lines(bbox, self.gym, self.viewer, self.env_ptrs[env_id],
                           pose=gymapi.Transform())

