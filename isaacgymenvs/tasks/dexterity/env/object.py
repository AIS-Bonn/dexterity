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
import math
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import trimesh
from typing import *
from urdfpy import URDF

from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.dexterity.base.base import DexterityBase
from isaacgymenvs.tasks.dexterity.env.schema_class_env import DexterityABCEnv
from isaacgymenvs.tasks.dexterity.env.schema_config_env import \
    DexteritySchemaConfigEnv


class DexterityObject:
    """Helper class that wraps object assets to make information about object
    geometry, etc. more easily available."""
    def __init__(self, gym, sim, asset_root: str, asset_file: str) -> None:
        self._gym = gym
        self._sim = sim
        self._asset_root = asset_root
        self._asset_file = asset_file
        self.name = asset_file.split('/')[-1].split('.')[0]

        self.acquire_asset_options()
        self.asset = gym.load_asset(sim, asset_root, asset_file,
                                     self.asset_options)

    def acquire_asset_options(self, vhacd_resolution: int = 100000) -> None:
        self._asset_options = gymapi.AssetOptions()
        self._asset_options.override_com = True
        self._asset_options.override_inertia = True
        self._asset_options.vhacd_enabled = True  # Enable convex decomposition
        self._asset_options.vhacd_params = gymapi.VhacdParams()
        self._asset_options.vhacd_params.resolution = vhacd_resolution

    def acquire_mesh(self) -> None:
        urdf = URDF.load(os.path.join(self._asset_root, self._asset_file))
        self.mesh = urdf.base_link.collision_mesh

    @property
    def asset_options(self) -> gymapi.AssetOptions:
        return self._asset_options

    @asset_options.setter
    def asset_options(self, asset_options: gymapi.AssetOptions) -> None:
        self._asset_options = asset_options

    @property
    def rigid_body_count(self) -> int:
        return self._gym.get_asset_rigid_body_count(self.asset)

    @property
    def rigid_shape_count(self) -> int:
        return self._gym.get_asset_rigid_shape_count(self.asset)

    @property
    def start_pose(self) -> gymapi.Transform:
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0, 0, 0.5)
        return start_pose

    def sample_points_from_mesh(self, num_samples: int) -> np.array:
        if not hasattr(self, "mesh"):
            self.acquire_mesh()
        points = np.array(trimesh.sample.sample_surface(
            self.mesh, count=num_samples)[0]).astype(float)
        return points
    
    def find_bounding_box_from_mesh(self) -> Tuple[np.array, np.array]:
        if not hasattr(self, "mesh"):
            self.acquire_mesh()
        to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
        return to_origin, extents


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
    
    def _update_observation_num(self, cfg) -> int:
        # Get the number of environment-specific observations.
        skip_keys = ()
        num_observations = 0
        for observation in cfg['env']['observations']:
            if observation.startswith('object_synthetic_pointcloud'):
                skip_keys += (observation,)
                split_observation = observation.split('_')
                assert split_observation[-1] == 'pos'
                if split_observation[-2].isdigit():
                    self.synthetic_pointcloud_dimension = int(split_observation[-2])
                else:
                    self.synthetic_pointcloud_dimension = 64
                obs_dim = self.synthetic_pointcloud_dimension * 3
            elif observation == 'object_bounding_box_pos':
                skip_keys += (observation,)
                obs_dim = 8 * 3
            else:
                continue
            num_observations += obs_dim
        
        # Add the number of base observations.
        num_observations += super()._update_observation_num(cfg, skip_keys=skip_keys)
        return num_observations
    
    def get_observation_tensor(self, observation: str) -> torch.Tensor:
        if observation.startswith('object_synthetic_pointcloud'):
            return self.object_synthetic_pointcloud_pos
        else:
            return super().get_observation_tensor(observation)

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
            regex_path_len = len(regex.split("/"))
            regex = os.path.normpath(os.path.join(root, regex))
            for path in glob.glob(regex):
                file_name = "/".join(path.split("/")[-regex_path_len:])
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
        self.object_handles = []  # Isaac Gym actors
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
                max_rigid_bodies = self.base_rigid_bodies + \
                    used_object.rigid_body_count
                max_rigid_shapes = self.base_rigid_shapes + \
                    used_object.rigid_shape_count
                self.gym.begin_aggregate(env_ptr, max_rigid_bodies,
                                         max_rigid_shapes, True)

            # Create common actors (robot, table, cameras)
            actor_count = self.create_base_actors(
                env_ptr, i, actor_count, robot_asset, robot_pose,
                table_asset, table_pose)

            # Aggregate task-specific actors (objects)
            if self.cfg_base.sim.aggregate_mode == 1:
                max_rigid_bodies = used_object.rigid_body_count
                max_rigid_shapes = used_object.rigid_shape_count
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
        self.object_actor_id_env = self.gym.find_actor_index(
            env_ptr, used_object.name, gymapi.DOMAIN_ENV)

    def _acquire_env_tensors(self):
        """Acquire and wrap tensors. Create views."""
        self.object_pos = self.root_pos[:, self.object_actor_id_env, 0:3]
        self.object_quat = self.root_quat[:, self.object_actor_id_env, 0:4]
        self.object_linvel = self.root_linvel[:, self.object_actor_id_env, 0:3]
        self.object_angvel = self.root_angvel[:, self.object_actor_id_env, 0:3]

        if any(obs.startswith("object_synthetic_pointcloud") for obs in self.cfg["env"]["observations"]):
            self.object_synthetic_pointcloud_pos = self._acquire_object_synthetic_pointcloud()

        if "object_bounding_box_pos" in self.cfg["env"]["observations"]:
            self.object_bounding_box_pos = self._acquire_object_bounding_box()

    def _acquire_object_synthetic_pointcloud(self) -> torch.Tensor:
        """Acquire position of the points relative to the mesh origin (object_mesh_sample_pos) and return a 
        placeholder for the absolute position of the points in space (object_synthetic_pointcloud_pos) to be updated
        by _refresh_object_synthetic_pointcloud().
        """
        object_mesh_samples_pos = []
        for obj in self.objects:
            object_mesh_samples_pos.append(obj.sample_points_from_mesh(num_samples=self.synthetic_pointcloud_dimension))
        object_mesh_samples_pos = np.stack(object_mesh_samples_pos)
        num_repeats = math.ceil(self.num_envs / len(self.objects))
        self.object_mesh_samples_pos = torch.from_numpy(object_mesh_samples_pos).to(
            self.device, dtype=torch.float32).repeat(num_repeats, 1, 1)[:self.num_envs]
        object_synthetic_pointcloud_pos = torch.zeros_like(self.object_mesh_samples_pos)
        return object_synthetic_pointcloud_pos
    
    def _refresh_object_synthetic_pointcloud(self) -> None:
        """Update the relative position of the sampled points (object_mesh_samples_pos) by the current 
        object pose."""
        num_samples = self.object_synthetic_pointcloud_pos.shape[1]
        self.object_synthetic_pointcloud_pos[:] = self.object_pos.unsqueeze(1).repeat(
            1, num_samples, 1) + quat_apply(self.object_quat.unsqueeze(1).repeat(1, num_samples, 1),
            self.object_mesh_samples_pos)

    def _acquire_object_bounding_box(self) -> torch.Tensor:
        """Acquire extent and pose offset of the bounding box and return a placeholder for the absolute 
        position of the bounding box corners in space (object_bounding_box_pos) to be updated by
        _refresh_object_bounding_box().
        """
        self.object_bounding_box_extents = []
        self.object_bounding_box_pos_offset = []
        self.object_bounding_box_quat_offset = []
        for obj in self.objects:
            to_origin, extents = obj.find_bounding_box_from_mesh()
            from_origin = np.linalg.inv(to_origin)
            rotation_matrix = R.from_matrix(from_origin[0:3, 0:3])
            translation = np.array(
                [from_origin[0, 3], from_origin[1, 3], from_origin[2, 3]])
            self.object_bounding_box_pos_offset.append(translation)
            self.object_bounding_box_quat_offset.append(rotation_matrix.as_quat())
            self.object_bounding_box_extents.append(extents)

        num_repeats = math.ceil(self.num_envs / len(self.objects))
        self.object_bounding_box_extents = torch.from_numpy(
            np.stack(self.object_bounding_box_extents)).to(self.device).float().repeat(num_repeats, 1)[:self.num_envs]
        self.object_bounding_box_pos_offset = torch.from_numpy(
            np.stack(self.object_bounding_box_pos_offset)).to(self.device).float().repeat(num_repeats, 1)[:self.num_envs]
        self.object_bounding_box_quat_offset = torch.from_numpy(
            np.stack(self.object_bounding_box_quat_offset)).to(self.device).float().repeat(num_repeats, 1)[:self.num_envs]

        self.bounding_box_corner_coords = torch.tensor(
            [[[-0.5, -0.5, -0.5],
              [-0.5, -0.5, 0.5],
              [-0.5, 0.5, -0.5],
              [-0.5, 0.5, 0.5],
              [0.5, -0.5, -0.5],
              [0.5, -0.5, 0.5],
              [0.5, 0.5, -0.5],
              [0.5, 0.5, 0.5]]],
            device=self.device).repeat(self.num_envs, 1, 1)

        object_bounding_box_pos = torch.zeros([self.num_envs, 8, 3]).to(self.device)
        return object_bounding_box_pos

    def _refresh_object_bounding_box(self) -> None:
        bounding_box_pos = self.object_pos + quat_apply(
            self.object_quat, self.object_bounding_box_pos_offset)
        bounding_box_quat = quat_mul(self.object_quat, self.object_bounding_box_quat_offset)
        self.object_bounding_box_pos[:] = bounding_box_pos.unsqueeze(1).repeat(
            1, 8, 1) + quat_apply(bounding_box_quat.unsqueeze(1).repeat(1, 8, 1),
                self.bounding_box_corner_coords * self.object_bounding_box_extents.unsqueeze(
                    1).repeat(1, 8, 1))

    def refresh_env_tensors(self):
        """Refresh tensors."""
        # NOTE: Tensor refresh functions should be called once per step, before
        # setters.
        # NOTE: Since the object_pos, object_quat etc. are obtained from the
        # root state tensor through regular slicing, they are views rather than
        # separate tensors and hence don't have to be updated separately.

        if any(obs.startswith("object_synthetic_pointcloud") for obs in self.cfg["env"]["observations"]):
            self._refresh_object_synthetic_pointcloud()

        if "object_bounding_box_pos" in self.cfg["env"]["observations"]:
            self._refresh_object_bounding_box()

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
        x_lower = self.cfg_task.randomize.workspace_extent_xy[0][0] + self.cfg_task.randomize.object_pos_drop[0]
        x_upper = self.cfg_task.randomize.workspace_extent_xy[1][0] + self.cfg_task.randomize.object_pos_drop[0]
        y_lower = self.cfg_task.randomize.workspace_extent_xy[0][1] + self.cfg_task.randomize.object_pos_drop[1]
        y_upper = self.cfg_task.randomize.workspace_extent_xy[1][1] + self.cfg_task.randomize.object_pos_drop[1]
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
