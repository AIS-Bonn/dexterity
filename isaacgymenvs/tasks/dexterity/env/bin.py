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

import hydra
import numpy as np
import os
import torch
from typing import *
from isaacgym.torch_utils import *
import random
from isaacgym import gymapi, gymtorch, torch_utils, gymutil
from isaacgymenvs.tasks.dexterity.env.schema_config_env import \
    DexteritySchemaConfigEnv
from isaacgymenvs.tasks.dexterity.env.object import DexterityEnvObject
from scipy.spatial.transform import Rotation as R


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

        
        if self.cfg_env["env"]["bin_asset"] == 'no_bin':
            self.bin_info = {"extent": [[-0.15, -0.15, 0.0], [0.15, 0.15, 0.15]]}
        else:
            bin_info_path = f'../../assets/dexterity/{self.cfg_env["env"]["bin_asset"]}/bin_info.yaml'  # relative to Gym's Hydra search path (cfg dir)
            self.bin_info = hydra.compose(config_name=bin_info_path)
            self.bin_info = self.bin_info['']['']['']['']['']['']['assets']['dexterity'][self.cfg_env["env"]["bin_asset"]]  # strip superfluous nesting

        self.object_sets_asset_root = os.path.normpath(os.path.join(
            os.path.dirname(__file__),
            self.cfg_env['env']['object_sets_asset_root']))

    def _import_env_assets(self):
        """Set object assets options. Import objects."""
        # Import objects assets like in the single object task
        object_assets = super()._import_env_assets()

        if self.cfg_env["env"]["bin_asset"] == 'no_bin':
            bin_asset = None
        else:   
            # Import bin asset
            bin_asset_root = os.path.normpath(
                os.path.join(os.path.dirname(__file__), '..', '..', '..', '..',
                                'assets', 'dexterity', self.cfg_env['env']['bin_asset']))
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
            p=gymapi.Vec3(*self.cfg_env.env.bin_pos[self.cfg_env.env.setup]),
            r=gymapi.Quat(*self.cfg_env.env.bin_quat[self.cfg_env.env.setup]))

        self.env_ptrs = []
        self.object_handles = [[] for _ in range(self.num_envs)]
        self.bin_handles = []
        self.object_actor_ids_sim = [[] for _ in range(self.num_envs)]  # within-sim indices

        self.object_ids_in_each_bin = []
        self.object_names_in_each_bin = [[] for _ in range(self.num_envs)]

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
            self.object_ids_in_each_bin.append(objects_idx)

            # Get rigid body and shape count of selected subset
            objects_rigid_body_count = sum(
                [o.rigid_body_count for o in [self.objects[i] for i in
                 objects_idx]])
            objects_rigid_shape_count = sum(
                [o.rigid_shape_count for o in [self.objects[i] for i in
                 objects_idx]])

            # Get rigid body and shape count for bin asset
            if bin_asset is not None:
                bin_rigid_body_count = self.gym.get_asset_rigid_body_count(
                    bin_asset)
                bin_rigid_shape_count = self.gym.get_asset_rigid_shape_count(
                    bin_asset)
            else:
                bin_rigid_body_count = 0
                bin_rigid_shape_count = 0

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
            if bin_asset is not None:
                bin_handle = self.gym.create_actor(
                    env_ptr, bin_asset, bin_pose, 'bin', i, 0, 2)
                self.bin_handles.append(bin_handle)
                actor_count += 1

            # Create object actors
            for id, object_idx in enumerate(objects_idx):
                used_object = self.objects[object_idx]
                object_handle = self.gym.create_actor(
                    env_ptr, used_object.asset, used_object.start_pose,
                    used_object.name, i, 0, 3 + id)
                self.object_actor_ids_sim[i].append(actor_count)
                self.object_handles[i].append(object_handle)
                self.object_names_in_each_bin[i].append(used_object.name)
                actor_count += 1

            # Finish aggregation group
            if self.cfg_base.sim.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.env_ptrs.append(env_ptr)

        self.num_actors = int(actor_count / self.num_envs)  # per env
        self.num_bodies = self.gym.get_env_rigid_body_count(env_ptr)  # per env
        self.num_dofs = self.gym.get_env_dof_count(env_ptr)  # per env

        # For setting targets
        self.robot_actor_ids_sim = torch.tensor(self.robot_actor_ids_sim, dtype=torch.int32, device=self.device)
        self.object_actor_ids_sim = torch.tensor(self.object_actor_ids_sim, dtype=torch.int32, device=self.device)

        # To access object-specific information about the target objects.
        self.object_ids_in_each_bin = torch.tensor(self.object_ids_in_each_bin, dtype=torch.long, device=self.device)
        self.target_object_instance = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

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

    def _acquire_object_bounding_box(self) -> None:
        # Init tensors for object bounding boxes.
        self.object_bounding_box = torch.zeros((self.num_envs, self.cfg_env['env']['num_objects'], 10)).to(self.device)  # [pos, quat, extents] with shape (num_envs, num_objects_per_bin, 10)
        self.target_object_bounding_box = torch.zeros((self.num_envs, 10)).to(self.device)  # [pos, quat, extents] with shape (num_envs, 10)
        
        # Retrieve bounding box pose and extents for each object.
        object_bounding_box_extents = torch.zeros((len(self.objects), 3)).to(self.device)
        object_bounding_box_from_origin_pos = torch.zeros((len(self.objects), 3)).to(self.device)
        object_bounding_box_from_origin_quat = torch.zeros((len(self.objects), 4)).to(self.device)
        for i, obj in enumerate(self.objects):
            to_origin, extents = obj.find_bounding_box_from_mesh()
            from_origin = np.linalg.inv(to_origin)
            from_origin_pos = from_origin[0:3, 3]
            from_origin_quat = R.from_matrix(from_origin[0:3, 0:3]).as_quat()
            object_bounding_box_from_origin_pos[i] = torch.from_numpy(from_origin_pos)
            object_bounding_box_from_origin_quat[i] = torch.from_numpy(from_origin_quat)
            object_bounding_box_extents[i] = torch.from_numpy(extents)

        # Gather with linear indices avoids for-loop over the environments.
        self.object_bounding_box_from_origin_pos = object_bounding_box_from_origin_pos.view(-1).gather(
            0, (self.object_ids_in_each_bin.unsqueeze(-1).expand(-1, -1, 3) * 3 + torch.arange(3, device=self.device)).reshape(-1)).view(
            self.num_envs, self.cfg_env['env']['num_objects'], 3)
        self.object_bounding_box_from_origin_quat = object_bounding_box_from_origin_quat.view(-1).gather(
            0, (self.object_ids_in_each_bin.unsqueeze(-1).expand(-1, -1, 4) * 4 + torch.arange(4, device=self.device)).reshape(-1)).view(
            self.num_envs, self.cfg_env['env']['num_objects'], 4)
        self.object_bounding_box[..., 7:10] = object_bounding_box_extents.view(-1).gather(
            0, (self.object_ids_in_each_bin.unsqueeze(-1).expand(-1, -1, 3) * 3 + torch.arange(3, device=self.device)).reshape(-1)).view(
            self.num_envs, self.cfg_env['env']['num_objects'], 3)
        
        if any("object_bounding_box_as_points" in obs for obs in self.cfg["env"]["observations"]) or any("bounding_box_clearance" in reward_term for reward_term in self.cfg['rl']['reward']):
            self.bounding_box_to_points = torch.tensor(
                [[[[-0.5, -0.5, -0.5],
                   [-0.5, -0.5,  0.5],
                   [-0.5,  0.5, -0.5],
                   [-0.5,  0.5,  0.5],
                   [ 0.5, -0.5, -0.5],
                   [ 0.5, -0.5,  0.5],
                   [ 0.5,  0.5, -0.5],
                   [ 0.5,  0.5,  0.5]]]], device=self.device).repeat(self.num_envs, self.cfg_env['env']['num_objects'], 1, 1)
            self.object_bounding_box_as_points = torch.zeros((self.num_envs, self.cfg_env['env']['num_objects'], 8, 3)).to(self.device)
            self.target_object_bounding_box_as_points = torch.zeros((self.num_envs, 8, 3)).to(self.device)

    def _refresh_object_bounding_box(self) -> None:
        # Update bounding box position.
        self.object_bounding_box[:, :, 0:3] = self.object_pos + quat_apply(
            self.object_quat, self.object_bounding_box_from_origin_pos)
        # Update bounding box quaternion.
        self.object_bounding_box[:, :, 3:7] = quat_mul(self.object_quat, self.object_bounding_box_from_origin_quat)

        # Select target object bounding box.
        self.target_object_bounding_box[:] = self.object_bounding_box.gather(1, self.target_object_id.unsqueeze(1).unsqueeze(2).repeat(1, 1, 10)).squeeze(1)

    
        if any("object_bounding_box_as_points" in obs for obs in self.cfg["env"]["observations"]) or any("bounding_box_clearance" in reward_term for reward_term in self.cfg['rl']['reward']):
            self.object_bounding_box_as_points[:] = self.object_bounding_box[:, :, 0:3].unsqueeze(2).repeat(1, 1, 8, 1) + quat_apply(self.object_bounding_box[:, :, 3:7].unsqueeze(2).repeat(1, 1, 8, 1), self.bounding_box_to_points * self.object_bounding_box[:, :, 7:10].unsqueeze(2).repeat(1, 1, 8, 1))

            # Select target object bounding box in points format.
            self.target_object_bounding_box_as_points[:] = self.object_bounding_box_as_points.gather(1, self.target_object_id.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, 8, 3)).squeeze(1)

    def _acquire_synthetic_pointcloud(self, num_samples: int = 64, sample_mode: str = 'area') -> torch.Tensor:
        self.object_mesh_samples_pos = torch.zeros((len(self.objects), self.max_num_points_padded, 4)).to(self.device)

        if sample_mode == 'uniform':
            num_samples = [num_samples, ] * len(self.objects)
        elif sample_mode == 'area':
            areas = [obj.surface_area for obj in self.objects]
            mean_area = sum(areas) / len(areas)
            print("names: ", [obj.name for obj in self.objects])
            print("areas: ", areas)
            num_samples = [int(num_samples * area / mean_area) for area in areas]
            print("num_samples: ", num_samples)

        object_mesh_samples_pos = []
        for i, obj in enumerate(self.objects):
            object_mesh_samples_pos = torch.from_numpy(obj.sample_points_from_mesh(num_samples=num_samples[i])).to(self.device, dtype=torch.float32)
            self.object_mesh_samples_pos[i, 0:min(object_mesh_samples_pos.shape[0], self.max_num_points_padded), :3] = object_mesh_samples_pos[:self.max_num_points_padded, :]
            self.object_mesh_samples_pos[i, 0:min(object_mesh_samples_pos.shape[0], self.max_num_points_padded), 3] = 1

        self.object_mesh_samples_pos = self.object_mesh_samples_pos.unsqueeze(0).repeat(self.num_envs, 1, 1, 1)  # shape: (num_envs, total_num_objects, max_num_point_padded, 4)

        self.target_object_mesh_samples_pos = torch.zeros((self.num_envs, self.max_num_points_padded, 4),
                                                          device=self.device)
        object_synthetic_pointcloud_pos = self.object_mesh_samples_pos[:, 0].detach().clone()
        return object_synthetic_pointcloud_pos

    def refresh_env_tensors(self):
        """Refresh tensors."""

        # Refresh pose and velocities of all objects (since object_pos, object_quat, etc. are obtained from the root state tensor through advanced slicing, they are separate tensors and have to be updated separately).
        self.object_pos[:] = self.root_pos[:, self.object_actor_id_env, 0:3]
        self.object_quat[:] = self.root_quat[:, self.object_actor_id_env, 0:4]
        self.object_linvel[:] = self.root_linvel[:, self.object_actor_id_env, 0:3]
        self.object_angvel[:] = self.root_angvel[:, self.object_actor_id_env, 0:3]

        # Refresh additional required tensors such as bounding boxes and point-clouds.
        super().refresh_env_tensors()

        # Subsample target object from updated observations.
        self.target_object_pos = self.object_pos.gather(1, self.target_object_id.unsqueeze(1).unsqueeze(2).repeat(1, 1, 3)).squeeze(1)
        self.target_object_quat = self.object_quat.gather(1, self.target_object_id.unsqueeze(1).unsqueeze(2).repeat(1, 1, 4)).squeeze(1)
        self.target_object_linvel = self.object_linvel.gather(1, self.target_object_id.unsqueeze(1).unsqueeze(2).repeat(1, 1, 3)).squeeze(1)
        self.target_object_angvel = self.object_angvel.gather(1, self.target_object_id.unsqueeze(1).unsqueeze(2).repeat(1, 1, 3)).squeeze(1)

        if any("position_clearance" in reward_term for reward_term in self.cfg['rl']['reward']) and hasattr(self, 'object_pos_initial'):
            self.target_object_pos_initial = self.object_pos_initial.gather( 1, self.target_object_id.unsqueeze(1).unsqueeze(2).repeat(1, 1, 3)).squeeze(1)

        if any("bounding_box_clearance" in reward_term for reward_term in self.cfg['rl']['reward']) and hasattr(self, 'object_bounding_box_as_points_initial'):
            self.target_object_bounding_box_as_points_initial = self.object_bounding_box_as_points_initial.gather(1, self.target_object_id.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 1, 8, 3)).squeeze(1)
        
        if any("pointcloud_clearance" in reward_term for reward_term in self.cfg['rl']['reward']):
            raise NotImplementedError

        

    def _refresh_synthetic_pointcloud(self, object_name: str = 'object',
                                      mesh_samples_name: str = 'object_mesh_samples') -> None:
        super()._refresh_synthetic_pointcloud(object_name='target_object',
                                              mesh_samples_name='target_object_mesh_samples')
        
    def _refresh_rendered_pointcloud(self):
        target_segmentation_id = self.target_object_id + 3
        super()._refresh_rendered_pointcloud(target_segmentation_id=target_segmentation_id)

    def _reset_segmentation_tracking(self, env_ids, draw_debug_visualization: bool = True):
        target_segmentation_id = (self.target_object_id + 3).cpu().numpy().tolist()
        return super()._reset_segmentation_tracking(env_ids, target_segmentation_id, draw_debug_visualization)


    def _object_in_bin(self, object_pos) -> torch.Tensor:
        x_lower = self.bin_info['extent'][0][0] + self.cfg_env.env.bin_pos[self.cfg_env.env.setup][0]
        x_upper = self.bin_info['extent'][1][0] + self.cfg_env.env.bin_pos[self.cfg_env.env.setup][0]
        y_lower = self.bin_info['extent'][0][1] + self.cfg_env.env.bin_pos[self.cfg_env.env.setup][1]
        y_upper = self.bin_info['extent'][1][1] + self.cfg_env.env.bin_pos[self.cfg_env.env.setup][1]
        z_lower = self.bin_info['extent'][0][2] + self.cfg_env.env.bin_pos[self.cfg_env.env.setup][2]
        z_upper = self.bin_info['extent'][1][2] + self.cfg_env.env.bin_pos[self.cfg_env.env.setup][2]
        in_bin = x_lower <= object_pos[..., 0]
        in_bin = torch.logical_and(in_bin, object_pos[..., 0] <= x_upper)
        in_bin = torch.logical_and(in_bin, y_lower <= object_pos[..., 1])
        in_bin = torch.logical_and(in_bin, object_pos[..., 1] <= y_upper)
        in_bin = torch.logical_and(in_bin, z_lower <= object_pos[..., 2])
        in_bin = torch.logical_and(in_bin, object_pos[..., 2] <= z_upper)
        return in_bin

    def _place_objects_before_bin(self):
        # Place the objects in front of the bin
        self.root_pos[:, self.object_actor_id_env, 0] = 1.1
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

    def _env_observation_num(self, observation: str) -> int:

        # Different versions of point-clouds going from fastest to most accurate.
        # 1. Synthetic point-clouds: Run without rendering and are hence extremely fast, but do not account for occlusions and viewpoint.
        # 2. Rendered point-clouds: Run using Isaac Gym camera sensors and ground truth segmentations making them slower. But they account for occlusions and viewpoint.
        # 3. Detected point-clouds: Run using an instance segmentation pipeline on visual observations. As neither object-meshes nor ground truth segmentations are available in the real-world, this is the observation-type used for transfer.

        if observation.startswith('synthetic_pointcloud'):
            split_observation = observation.split('_')
            if split_observation[-1].isdigit():
                self.synthetic_pointcloud_dimension = int(split_observation[-1])
            num_observations = 4 * self.max_num_points_padded  # (x, y, z, mask)

        elif observation.startswith('rendered_pointcloud'):
            num_observations = 4 * self.max_num_points_padded  # (x, y, z, mask)

        elif observation.startswith('detected_pointcloud'):
            num_observations = 4 * self.max_num_points_padded  # (x, y, z, mask)
        
        elif observation == 'target_object_bounding_box':
            num_observations = 10  # (pos, quat, extents)

        elif observation == 'object_bounding_box':
            num_observations = 10 * self.cfg_env['env']['num_objects']

        elif observation == 'target_object_bounding_box_as_points':
            num_observations = 24
        
        elif observation == 'object_bounding_box_as_points':
            num_observations = 24 * self.cfg_env['env']['num_objects']

        else:
            raise NotImplementedError
        
        return num_observations
    
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

    def visualize_target_object_bounding_box(self, env_id: int) -> None:
        self.visualize_bounding_boxes(env_id, 'target_object_bounding_box')

    def visualize_object_bounding_box_as_points(self, env_id: int) -> None:
        self.visualize_pos(self.object_bounding_box_as_points.flatten(1, 2), env_id, color=(1, 1, 0))

    def visualize_target_object_bounding_box_as_points(self, env_id: int) -> None:
        self.visualize_pos(self.target_object_bounding_box_as_points, env_id, color=(1, 1, 0))

    def visualize_bin_extent(self, env_id) -> None:
        extent = torch.tensor(self.bin_info['extent'])
        bin_pose = gymapi.Transform(
            p=gymapi.Vec3(*self.cfg_env.env.bin_pos[self.cfg_env.env.setup]))
        bbox = gymutil.WireframeBBoxGeometry(extent, pose=bin_pose,
                                             color=(0, 1, 1))
        gymutil.draw_lines(bbox, self.gym, self.viewer, self.env_ptrs[env_id],
                           pose=gymapi.Transform())
