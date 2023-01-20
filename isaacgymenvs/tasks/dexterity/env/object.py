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

"""Dexterity: class for object env.

Inherits base class and abstract environment class. Inherited by object task
classes. Not directly executed.

Configuration defined in DexterityEnvObject.yaml. Asset info defined in
asset_info_object_sets.yaml.
"""

import glob
import hydra
import numpy as np
import os
from typing import *

from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.dexterity.base.base import DexterityBase
from isaacgymenvs.tasks.dexterity.env.schema_class_env import DexterityABCEnv
from isaacgymenvs.tasks.dexterity.env.schema_config_env import \
    DexteritySchemaConfigEnv


class DexterityObject:
    """Helper class that wraps object assets to make information about object
    geometry, etc. more easily available."""
    def __init__(self, gym, sim, asset_root, asset_file) -> None:
        self._gym = gym

        # Load default asset options
        self._asset_options = gymapi.AssetOptions()
        self._asset_options.override_com = True
        self._asset_options.override_inertia = True
        self._asset_options.vhacd_enabled = True  # Enable convex decomposition
        self._asset_options.vhacd_params = gymapi.VhacdParams()
        self._asset_options.vhacd_params.resolution = 1000000

        # Create IsaacGym asset
        self._asset = gym.load_asset(sim, asset_root, asset_file,
                                     self._asset_options)

        # Set object name based on asset file name
        self.name = asset_file.split('/')[-1].split('.')[0]

    @property
    def asset(self):
        return self._asset

    @property
    def asset_options(self) -> gymapi.AssetOptions:
        return self._asset_options

    @asset_options.setter
    def asset_options(self, asset_options: gymapi.AssetOptions) -> None:
        self._asset_options = asset_options

    @property
    def rigid_body_count(self) -> int:
        return self._gym.get_asset_rigid_body_count(self._asset)

    @property
    def rigid_shape_count(self) -> int:
        return self._gym.get_asset_rigid_shape_count(self._asset)

    @property
    def start_pose(self) -> gymapi.Transform:
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0, 0, 0.5)
        return start_pose


class DexterityEnvObject(DexterityBase, DexterityABCEnv):

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

        config_path = 'task/DexterityEnvObject.yaml'  # relative to cfg dir
        self.cfg_env = hydra.compose(config_name=config_path)
        self.cfg_env = self.cfg_env['task']  # strip superfluous nesting

        self.object_sets_asset_root = os.path.normpath(os.path.join(
            os.path.dirname(__file__),
            self.cfg_env['env']['object_sets_asset_root']))

    def create_envs(self):
        """Set env options. Import assets. Create actors."""

        lower = gymapi.Vec3(-self.cfg_base.env.env_spacing,
                            -self.cfg_base.env.env_spacing, 0.0)
        upper = gymapi.Vec3(self.cfg_base.env.env_spacing,
                            self.cfg_base.env.env_spacing,
                            self.cfg_base.env.env_spacing)
        num_per_row = int(np.sqrt(self.num_envs))

        robot_asset, table_asset = self.import_robot_assets()
        object_assets = self._import_env_assets()

        self._create_actors(lower, upper, num_per_row, robot_asset, table_asset,
                            object_assets)

    def _import_env_assets(self):
        """Set object assets options. Import objects."""
        self.object_sets_dict, self.object_count = self._get_objects()

        self.objects = []
        for object_set, object_list in self.object_sets_dict.items():
            for object_name in object_list:
                self.objects.append(
                    DexterityObject(
                        self.gym, self.sim, self.object_sets_asset_root,
                        f'urdf/{object_set}/' + object_name + '.urdf'))
        return [obj.asset for obj in self.objects]

    def _get_objects(self) -> Dict[str, List[str]]:
        def solve_object_regex(regex: str, object_set: str) -> List[str]:
            root = os.path.join(self.object_sets_asset_root, 'urdf', object_set)
            ret_list = []
            regex = os.path.normpath(os.path.join(root, regex))
            for path in glob.glob(regex):
                file_name = path.split("/")[-1]
                if "." in file_name:
                    obj, extension = file_name.split(".")
                else:
                    obj = file_name
                    extension = ""
                if extension == "urdf":
                    ret_list.append(obj)
            return ret_list

        object_count = 0
        object_dict = {}
        for dataset in self.cfg_env['env']['object_sets'].keys():
            if isinstance(self.cfg_env['env']['object_sets'][dataset], str):
                object_names = [self.cfg_env['env']['object_sets'][dataset]]
            else:
                object_names = self.cfg_env['env']['object_sets'][dataset]
            dataset_object_list = []
            for object_name in object_names:
                if "*" in object_name:
                    dataset_object_list += solve_object_regex(
                        object_name, dataset)
                else:
                    dataset_object_list.append(object_name)
            object_dict[dataset] = dataset_object_list
            object_count += len(dataset_object_list)
        return object_dict, object_count

    def _create_actors(self, lower: gymapi.Vec3, upper: gymapi.Vec3,
                       num_per_row: int, robot_asset, table_asset,
                       object_assets) -> None:
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

        self.env_ptrs = []
        self.robot_handles = []
        self.table_handles = []
        self.object_handles = []
        self.shape_ids = []
        self.robot_actor_ids_sim = []  # within-sim indices
        self.table_actor_ids_sim = []  # within-sim indices
        self.object_actor_ids_sim = []  # within-sim indices
        actor_count = 0

        for i in range(self.num_envs):
            # Create new env
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Loop through all used objects
            object_idx = i % self.object_count
            used_object = self.objects[object_idx]

            # Aggregate all actors
            if self.cfg_base.sim.aggregate_mode > 1:
                max_rigid_bodies = \
                    used_object.rigid_body_count + \
                    self.robot.rigid_body_count + \
                    int(self.cfg_base.env.has_table) + \
                    self.camera_rigid_body_count
                max_rigid_shapes = \
                    used_object.rigid_shape_count + \
                    self.robot.rigid_shape_count + \
                    int(self.cfg_base.env.has_table) + \
                    self.camera_rigid_body_count
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
                    used_object.rigid_body_count
                max_rigid_shapes = \
                    used_object.rigid_shape_count
                self.gym.begin_aggregate(env_ptr, max_rigid_bodies,
                                         max_rigid_shapes, True)

            # Create object actor
            object_handle = self.gym.create_actor(
                env_ptr, used_object.asset, used_object.start_pose,
                used_object.name, i, 0, 2)
            self.object_actor_ids_sim.append(actor_count)
            self.object_handles.append(object_handle)
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
        self.object_actor_id_env = self.gym.find_actor_index(
            env_ptr, used_object.name, gymapi.DOMAIN_ENV)

    def _acquire_env_tensors(self):
        """Acquire and wrap tensors. Create views."""
        self.object_pos = self.root_pos[:, self.object_actor_id_env, 0:3]
        self.object_quat = self.root_quat[:, self.object_actor_id_env, 0:4]
        self.object_linvel = self.root_linvel[:, self.object_actor_id_env, 0:3]
        self.object_angvel = self.root_angvel[:, self.object_actor_id_env, 0:3]

    def refresh_env_tensors(self):
        """Refresh tensors."""
        # NOTE: Tensor refresh functions should be called once per step, before
        # setters.
        # NOTE: Since the object_pos, object_quat etc. are obtained from the
        # root state tensor through regular slicing, they are views rather than
        # separate tensors and hence don't have to be updated separately.
        pass

    def _get_random_drop_pos(self, env_ids) -> torch.Tensor:
        object_pos_drop = torch.tensor(
            self.cfg_task.randomize.object_pos_drop, device=self.device
        ).unsqueeze(0).repeat(len(env_ids), 1)
        object_pos_drop_noise = \
            2 * (torch.rand((len(env_ids), 3), dtype=torch.float32,
                            device=self.device) - 0.5)  # [-1, 1]
        object_pos_drop_noise = object_pos_drop_noise @ torch.diag(torch.tensor(
            self.cfg_task.randomize.object_pos_drop_noise, device=self.device))
        object_pos_drop += object_pos_drop_noise
        return object_pos_drop

    def _get_random_drop_quat(self, env_ids) -> torch.Tensor:
        x_unit_tensor = to_torch(
            [1, 0, 0], dtype=torch.float, device=self.device).repeat(
            (len(env_ids), 1))
        y_unit_tensor = to_torch(
            [0, 1, 0], dtype=torch.float, device=self.device).repeat(
            (len(env_ids), 1))
        rand_floats = torch_rand_float(
            -1.0, 1.0, (len(env_ids), 2), device=self.device)
        object_quat_drop = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1], x_unit_tensor, y_unit_tensor)
        return object_quat_drop

    def _object_in_workspace(self, object_pos) -> torch.Tensor:
        x_lower = self.cfg_task.randomize.workspace_extent_xy[0][0]
        x_upper = self.cfg_task.randomize.workspace_extent_xy[1][0]
        y_lower = self.cfg_task.randomize.workspace_extent_xy[0][1]
        y_upper = self.cfg_task.randomize.workspace_extent_xy[1][1]
        object_in_workspace = x_lower <= object_pos[:, 0]
        object_in_workspace = torch.logical_and(
            object_in_workspace, object_pos[:, 0] <= x_upper)
        object_in_workspace = torch.logical_and(
            object_in_workspace, y_lower <= object_pos[:, 1])
        object_in_workspace = torch.logical_and(
            object_in_workspace, object_pos[:, 1] <= y_upper)
        return object_in_workspace

    def visualize_object_pose(self, env_id: int, axis_length: float = 0.3
                              ) -> None:
        self.visualize_body_pose("object", env_id, axis_length)

    def visualize_workspace_xy(self, env_id: int) -> None:
        # Set extent in z-direction to 0
        lower = self.cfg_task.randomize.workspace_extent_xy[0] + [0.]
        upper = self.cfg_task.randomize.workspace_extent_xy[1] + [0.]
        extent = torch.tensor([lower, upper])
        drop_pose = gymapi.Transform(
            p=gymapi.Vec3(self.cfg_task.randomize.object_pos_drop[0],
                          self.cfg_task.randomize.object_pos_drop[1],
                          0.))
        bbox = gymutil.WireframeBBoxGeometry(extent, pose=drop_pose,
                                             color=(0, 1, 1))
        gymutil.draw_lines(bbox, self.gym, self.viewer, self.env_ptrs[env_id],
                           pose=gymapi.Transform())


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))
