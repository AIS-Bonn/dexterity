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

import cv2
import hydra
import math
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
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
from isaacgymenvs.tasks.dexterity.base.base_cameras import DexterityCameraSensorProperties
from isaacgymenvs.tasks.dexterity.base.base_cameras import xyz_to_image, image_plane_to_bounding_box, draw_square, draw_bounding_box, xyz_world_to_camera, xyz_camera_to_world


from scipy.ndimage import binary_dilation
from PIL import Image
from aot_tracker import _palette
import gc


def colorize_mask(pred_mask):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode='RGB')
    return np.array(save_mask)


def draw_mask(img, mask, alpha=0.5, id_countour=False):
    img_mask = np.zeros_like(img)
    img_mask = img
    if id_countour:
        # very slow ~ 1s per image
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]

        for id in obj_ids:
            # Overlay color on  binary mask
            if id <= 255:
                color = _palette[id * 3:id * 3 + 3]
            else:
                color = [0, 0, 0]
            foreground = img * (1 - alpha) + np.ones_like(
                img) * alpha * np.array(color)
            binary_mask = (mask == id)

            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]

            countours = binary_dilation(binary_mask,
                                        iterations=1) ^ binary_mask
            img_mask[countours, :] = 0
    else:
        binary_mask = (mask != 0)
        countours = binary_dilation(binary_mask,
                                    iterations=1) ^ binary_mask
        foreground = img * (1 - alpha) + colorize_mask(mask) * alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[countours, :] = 0

    return img_mask.astype(img.dtype)


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
        self.synthetic_pointcloud_dimension = 64

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
                obs_dim = self.synthetic_pointcloud_dimension * 3
            elif observation == 'object_bounding_box_pos':
                skip_keys += (observation,)
                obs_dim = 8 * 3
            elif observation.startswith('detected_2d_bounding_box_image'):
                obs_dim = 4  # xywh of bounding box detected by CV pipeline
            elif observation.startswith('synthetic_2d_bounding_box_image'):
                skip_keys += (observation,)
                obs_dim = 4  # xywh of the bounding box projected to the camera image
            elif observation.startswith('synthetic_2d_bounding_box_world'):
                skip_keys += (observation,)
                obs_dim = 4 * 3  # 4 projected corner points
            elif observation.startswith('detected_segmented_point_cloud'):
                skip_keys += (observation,)
                obs_dim = 3
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

        self.used_2d_cameras = []
        self.segmented_point_cloud_cameras = []

        self.detected_segemented_point_clouds = {}
        
        # Using synthetic point-cloud in observation.
        if any(obs.startswith("object_synthetic_pointcloud") for obs in self.cfg["env"]["observations"]):
            self.object_synthetic_pointcloud_pos = self._acquire_object_synthetic_pointcloud()
        # Get synthetic point-cloud if I want to calculate bounding boxes.
        elif any(obs.startswith("synthetic_2d_bounding_box") for obs in self.cfg["env"]["observations"]):
            self.object_synthetic_pointcloud_pos = self._acquire_object_synthetic_pointcloud()

        # Acquire 3D bounding box.
        if "object_bounding_box_pos" in self.cfg["env"]["observations"]:
            self.object_bounding_box_pos = self._acquire_object_bounding_box()

        # Acquire synthetic or detected 2D bounding box.
        for obs in self.cfg["env"]["observations"]:
            if obs.startswith("synthetic_2d_bounding_box_image_"):
                self._acquire_2d_bounding_box_image(obs, prefix='synthetic_2d_bounding_box_image_')
            elif obs.startswith("synthetic_2d_bounding_box_world_"):
                self._acquire_2d_bounding_box_world(obs, prefix='synthetic_2d_bounding_box_world_')
            elif obs.startswith("detected_2d_bounding_box_image_"):
                self._acquire_2d_bounding_box_image(obs, prefix='detected_2d_bounding_box_image_')
            elif obs.startswith("detected_segmented_point_cloud_"):
                self._acquire_detected_segmented_point_cloud(obs)

    def _acquire_detected_segmented_point_cloud(self, obs: str):
        camera_name = obs[len("detected_segmented_point_cloud_"):]
        self.segmented_point_cloud_cameras.append(camera_name)

        setattr(self, obs, torch.zeros((self.num_envs, 3), device=self.device))

        setattr(self, obs + '_testing', [None for _ in range(self.num_envs)])
        setattr(self, obs + '_testing_subset', [None for _ in range(self.num_envs)])

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

    def _acquire_2d_bounding_box_image(self, obs: str, prefix: str) -> None:
        camera_name = obs[len(prefix):]
        assert camera_name in self.cfg_env.cameras.keys(), \
            f"Camera {camera_name}, required in observation {obs} is not " \
            f"specified in the environment config."
        self.used_2d_cameras.append(camera_name)

        # Store camera properties.
        camera_properties = DexterityCameraSensorProperties(
            **self.cfg_env.cameras[camera_name])
        setattr(self, camera_name + "_properties", camera_properties)

        # Initialize observation_tensor.
        setattr(self, obs, torch.zeros(self.num_envs, 4).to(self.device))

    def _acquire_2d_bounding_box_world(self, obs: str, prefix: str) -> None:
        camera_name = obs[len(prefix):]
        assert camera_name in self.cfg_env.cameras.keys(), \
            f"Camera {camera_name}, required in observation {obs} is not " \
            f"specified in the environment config."
        self.used_2d_cameras.append(camera_name)

        # Store camera properties.
        camera_properties = DexterityCameraSensorProperties(
            **self.cfg_env.cameras[camera_name])
        setattr(self, camera_name + "_properties", camera_properties)

        # Initialize observation_tensor.
        setattr(self, obs, torch.zeros(self.num_envs, 4, 3).to(self.device))

    def _refresh_synthetic_2d_bounding_box_image(self, draw_debug_visualization: bool = True) -> None:
        """To find an object's synthetic 2D bounding box without rendering the
        scene and employing an instance segmentation pipeline, we project the
        synthetic 3D pointcloud (points sampled on the object's mesh) into the
        image plane and then compute their 2D bounding box.
        """
        self._refresh_object_synthetic_pointcloud()

        for camera_name in self.used_2d_cameras:
            camera_properties = getattr(self, camera_name + "_properties")

            # Compute view matrix and projection matrix manually so Isaac Gym
            # does not need to create a camera sensor.
            if not hasattr(self, camera_name + "_view_matrix"):
                setattr(self, camera_name + "_view_matrix",
                        camera_properties.compute_view_matrix(
                            self.num_envs * self.synthetic_pointcloud_dimension,
                            self.device))
            if not hasattr(self, camera_name + "_projection_matrix"):
                setattr(self, camera_name + "_projection_matrix",
                        camera_properties.compute_projection_matrix(
                            self.num_envs * self.synthetic_pointcloud_dimension,
                            self.device))

            # Project synthetic pointcloud from world space to the image plane.
            image_plane = xyz_to_image(
                self.object_synthetic_pointcloud_pos,
                getattr(self, camera_name + '_projection_matrix'),
                getattr(self, camera_name + '_view_matrix'),
                camera_properties.width, camera_properties.height)

            synthetic_bounding_box = image_plane_to_bounding_box(image_plane)
            getattr(self, 'synthetic_2d_bounding_box_image_' + camera_name)[:] = \
                synthetic_bounding_box

            if draw_debug_visualization:
                assert camera_name in self.cfg["env"]["observations"], \
                    f"Camera '{camera_name}' is needed to draw the debug " \
                    f"visualization for observation 'synthetic_2d_bounding_" \
                    f"box_{camera_name}'."
                image_dict = self.get_images()

                fig, axs = plt.subplots(2, 2)

                for env_id in range(2):
                    image = image_dict[camera_name][env_id].cpu().numpy()

                    from detectron2.utils.visualizer import Visualizer
                    from detectron2.structures import Boxes, Instances

                    synthetic_instances = Instances(image_size=image.shape[0:2])
                    synthetic_instances.pred_boxes = Boxes(synthetic_bounding_box[env_id].unsqueeze(0))
                    axs[0, env_id].set_title(f'Env {env_id}')
                    axs[0, env_id].imshow(image)
                    axs[1, env_id].imshow(Visualizer(image, scale=1.2).draw_instance_predictions(synthetic_instances.to("cpu")).get_image())

                axs[0, 0].set(ylabel='Original RGB image')
                axs[1, 0].set(ylabel='Synthetic 2D bounding box')
                plt.show()

    def _refresh_synthetic_2d_bounding_box_world(self) -> None:
        self._refresh_object_synthetic_pointcloud()

        for camera_name in self.used_2d_cameras:
            camera_properties = getattr(self, camera_name + "_properties")

            # Compute view matrix and projection matrix manually so Isaac Gym
            # does not need to create a camera sensor.
            if not hasattr(self, camera_name + "_view_matrix"):
                setattr(self, camera_name + "_view_matrix",
                        camera_properties.compute_view_matrix(
                            self.num_envs * self.synthetic_pointcloud_dimension,
                            self.device))

            if not hasattr(self, camera_name + "_inverse_view_matrix"):
                setattr(self, camera_name + "_inverse_view_matrix",
                        getattr(self, camera_name + "_view_matrix").transpose(1, 2).inverse())

            if not hasattr(self, camera_name + "_projection_matrix"):
                setattr(self, camera_name + "_projection_matrix",
                        camera_properties.compute_projection_matrix(
                            self.num_envs * self.synthetic_pointcloud_dimension,
                            self.device))

            xyz_camera = xyz_world_to_camera(
                self.object_synthetic_pointcloud_pos,
                getattr(self, camera_name + '_view_matrix'))

            x_min_value, x_min_idx = torch.min(xyz_camera[..., 0], dim=1)
            x_max_value, x_max_idx = torch.max(xyz_camera[..., 0], dim=1)
            y_min_value, y_min_idx = torch.min(xyz_camera[..., 1], dim=1)
            y_max_value, y_max_idx = torch.max(xyz_camera[..., 1], dim=1)
            z_mean = torch.mean(xyz_camera[..., 2], dim=1)

            top_left = torch.stack([x_min_value, y_max_value, z_mean], dim=-1)
            top_right = torch.stack([x_max_value, y_max_value, z_mean], dim=-1)
            bottom_left = torch.stack([x_min_value, y_min_value, z_mean], dim=-1)
            bottom_right = torch.stack([x_max_value, y_min_value, z_mean], dim=-1)

            bounding_box_corners_camera = torch.stack([top_left, top_right, bottom_right, bottom_left], dim=1)
            bounding_box_corners_world = xyz_camera_to_world(bounding_box_corners_camera, getattr(self, camera_name + "_inverse_view_matrix")[:self.num_envs * 4])

            getattr(self, 'synthetic_2d_bounding_box_world_' + camera_name)[:] = \
                bounding_box_corners_world

    def _refresh_detected_segmented_point_cloud(
            self, draw_debug_visualization: bool = False):
        if not hasattr(self, 'segtrackers'):
            return

        self.gym.clear_lines(self.viewer)
        image_dict = self.get_images()
        for camera_name in self.segmented_point_cloud_cameras:
            camera = self._camera_dict[camera_name]
            xyz = camera.depth_to_xyz(self.env_ptrs, list(range(self.num_envs)))

            for env_id in range(self.num_envs):
                # Get image.
                color_image_numpy = (image_dict[camera_name][
                                         env_id].detach().cpu().numpy()[..., 0:3] * 255).astype(np.uint8)
                depth_image = image_dict[camera_name][env_id][..., 3]
                # Update tracker.
                pred_mask = self.segtrackers[camera_name][env_id].track(
                    color_image_numpy, update_memory=True)

                # Update segmented point-cloud.
                idxs = np.where(self.pred_mask.flatten())[0]

                #print("idxs:", idxs)

                #idxs = torch.from_numpy(self.pred_mask).view(-1).nonzero().squeeze()
                #print("idxs:", idxs)
                getattr(self, 'detected_segmented_point_cloud_' + camera_name + '_testing')[env_id] = xyz[env_id] #[idxs]
                getattr(self,
                        'detected_segmented_point_cloud_' + camera_name + '_testing_subset')[env_id] = xyz[env_id][idxs]

                if draw_debug_visualization:
                    selected_segmentation = draw_mask(color_image_numpy, pred_mask,
                                                      id_countour=False)
                    fig = plt.figure()
                    plt.axis('off')
                    ax = fig.add_subplot(111)
                    ax.set_title(
                        f'Tracked segmentation for camera {camera_name} on env {env_id}.')
                    plt.imshow(selected_segmentation)
                    plt.show()

    def _refresh_detected_2d_bounding_box_image(
            self, pipeline: str = 'sam',
            draw_debug_visualization: bool = False) -> None:
        image_dict = self.get_images()

        if pipeline == 'yolov8':
            from isaacgymenvs.tasks.dexterity.tools.yolov8_tracking.yolov8.ultralytics.nn.autobackend import AutoBackend
            from isaacgymenvs.tasks.dexterity.tools.yolov8_tracking.yolov8.ultralytics.yolo.utils.checks import check_file, \
                check_imgsz, check_imshow, print_args, check_requirements
            from isaacgymenvs.tasks.dexterity.tools.yolov8_tracking.yolov8.ultralytics.yolo.utils.ops import Profile, \
                non_max_suppression, scale_boxes, process_mask, \
                process_mask_native
            from isaacgymenvs.tasks.dexterity.tools.yolov8_tracking.yolov8.ultralytics.yolo.utils.plotting import Annotator, \
                colors, save_one_box
            from isaacgymenvs.tasks.dexterity.tools.yolov8_tracking.trackers.multi_tracker_zoo import create_tracker
            from pathlib import Path

            # Setup model and tracker.
            if not hasattr(self, "yolov8_model"):
                # Create YOLO model.
                yolo_weights = Path('yolov8s.pt')
                yolo_weights = Path('/home/user/mosbach/tools/yolo_ycb/yolov5/exp2/weights/best.pt')
                imgsz = list(image_dict['closeup'][0].shape[0:2])
                self.yolov8_model = AutoBackend(yolo_weights, device=torch.device(self.device), dnn=False, fp16=False)
                #stride, names, pt = self.model.stride, self.model.names, self.model.pt
                imgsz = check_imgsz(imgsz, stride=self.yolov8_model.stride)
                bs = 1
                self.yolov8_model.warmup(imgsz=(1 if self.yolov8_model.pt or self.yolov8_model.triton else bs, 3, *imgsz))  # warmup
                self.windows = []

                # Create tracker.
                tracking_method = 'deepocsort'
                tracking_config = './tasks/dexterity/tools/yolov8_tracking/trackers/deepocsort/configs/deepocsort.yaml'
                reid_weights = Path('osnet_x0_25_msmt17.pt')
                device = torch.device(self.device)
                self.tracker = create_tracker(tracking_method, tracking_config,
                                         reid_weights, device, False)

            # Retrieve and normalize image.
            im = image_dict['closeup'].permute(0, 3, 1, 2).to(self.device)
            im0 = image_dict['closeup'][0].detach().cpu().numpy()
            im = im.float()  # uint8 to fp32
            im /= 255.0  # 0 - 255 to 0.0 - 1.0

            print("im.shape:", im.shape)

            # Run YOLOv8 inference.
            preds = self.yolov8_model(im, augment=False, visualize=False)

            # Apply Non-maximum suppression (NMS).
            conf_thres = 0.5
            iou_thres = 0.5
            classes = None
            agnostic_nms = False
            max_det = 1000
            p = non_max_suppression(preds, conf_thres, iou_thres, classes,
                                    agnostic_nms, max_det=max_det)

            s = ''

            # Process detections.
            for i, det in enumerate(p):
                print("i:", i)
                print("det:", det)


                line_thickness = 2
                annotator = Annotator(im0, line_width=line_thickness,
                                      example=str(self.yolov8_model.names))

                if det is not None and len(det):

                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4],
                                             im0.shape).round()  # rescale boxes to im0 size

                    # Print results.
                    for c in det[:, 5].unique():
                        n = (det[:, 5] == c).sum()
                        s += f"{n} {self.yolov8_model.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Pass detections to strongsort.
                    print("im0.shape:", im0.shape)
                    print("det:", det)
                    print("det.requires_grad:", det.requires_grad)
                    self.outputs = self.tracker.update(det.detach().cpu(), im0)
                    print("outputs:", self.outputs)

                    # Draw boxes for visualization.
                    if len(self.outputs) > 0:

                        for j, (output) in enumerate(self.outputs):
                            bbox = output[0:4]
                            id = output[4]
                            cls = output[5]
                            conf = output[6]

                            save_vid = True

                            if save_vid:
                                hide_labels = False
                                hide_class = False
                                hide_conf = True
                                c = int(cls)
                                id = int(id)
                                label = self.yolov8_model.names[c]
                                label = None if hide_labels else (
                                    f'{id} {self.yolov8_model.names[c]}' if hide_conf else \
                                        (f'{id} {conf:.2f}' if hide_class else f'{id} {self.yolov8_model.names[c]} {conf:.2f}'))

                                color = colors(c, True)

                                annotator.box_label(bbox, label, color=color)

                im0 = annotator.result()

            plt.imshow(im0)
            plt.show()

            #import time
            #time.sleep(1000)

        elif pipeline == 'sam':



            from scipy.ndimage import binary_dilation
            from PIL import Image
            from aot_tracker import _palette
            import gc

            def colorize_mask(pred_mask):
                save_mask = Image.fromarray(pred_mask.astype(np.uint8))
                save_mask = save_mask.convert(mode='P')
                save_mask.putpalette(_palette)
                save_mask = save_mask.convert(mode='RGB')
                return np.array(save_mask)

            def draw_mask(img, mask, alpha=0.5, id_countour=False):
                img_mask = np.zeros_like(img)
                img_mask = img
                if id_countour:
                    # very slow ~ 1s per image
                    obj_ids = np.unique(mask)
                    obj_ids = obj_ids[obj_ids != 0]

                    for id in obj_ids:
                        # Overlay color on  binary mask
                        if id <= 255:
                            color = _palette[id * 3:id * 3 + 3]
                        else:
                            color = [0, 0, 0]
                        foreground = img * (1 - alpha) + np.ones_like(
                            img) * alpha * np.array(color)
                        binary_mask = (mask == id)

                        # Compose image
                        img_mask[binary_mask] = foreground[binary_mask]

                        countours = binary_dilation(binary_mask,
                                                    iterations=1) ^ binary_mask
                        img_mask[countours, :] = 0
                else:
                    binary_mask = (mask != 0)
                    countours = binary_dilation(binary_mask,
                                                iterations=1) ^ binary_mask
                    foreground = img * (1 - alpha) + colorize_mask(mask) * alpha
                    img_mask[binary_mask] = foreground[binary_mask]
                    img_mask[countours, :] = 0

                return img_mask.astype(img.dtype)

            im0 = image_dict['closeup'][0].detach().cpu().numpy()
            pred_mask = self.segtracker.track(im0, update_memory=True)
            selected_segmentation = draw_mask(im0, pred_mask, id_countour=False)


            plt.imshow(selected_segmentation)
            plt.show()

        '''

        # Some basic setup:
        # Setup detectron2 logger
        import detectron2
        from detectron2.utils.logger import setup_logger
        setup_logger()

        # import some common libraries
        import numpy as np
        import os, json, cv2, random

        # import some common detectron2 utilities
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg
        from detectron2.utils.visualizer import Visualizer
        from detectron2.data import MetadataCatalog, DatasetCatalog

        for camera_name in self.used_2d_cameras:
            assert camera_name in self.cfg["env"]["observations"], \
                f"Camera '{camera_name}' is needed detect bounding boxes in " \
                f"'detected_2d_bounding_box_image_{camera_name}'."

            test_image = image_dict[camera_name][0].cpu().numpy()
            print("showing image for camera:", camera_name)
            #cv2.imshow('image', test_image)
            #cv2.waitKey(0)

            cfg = get_cfg()
            # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
            cfg.merge_from_file(model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
            # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
            predictor = DefaultPredictor(cfg)
            outputs = predictor(test_image)

            # We can use `Visualizer` to draw the predictions on the image.
            v = Visualizer(test_image,
                           MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
                           scale=1.2)

            print("outputs:", outputs)

            print("outputs['instances']:", outputs['instances'])

            
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

            fig, axs = plt.subplots(2)

            axs[0].imshow(test_image)
            axs[0].set_title("Input image")
            axs[1].imshow(out.get_image())
            axs[1].set_title("Detections")

            plt.show()
            
        '''

        for camera_name in self.segmented_point_cloud_cameras:
            print("camera_name:", camera_name)

            camera_properties = getattr(self, camera_name + "_properties")

            print(camera_properties)

            import time
            time.sleep(1000)

    def _reset_segmentation_tracking(self, env_ids):
        if not hasattr(self, 'segtracker'):
            self._acquire_segtracker()

        self.gym.clear_lines(self.viewer)
        image_dict = self.get_images()
        for camera_name in self.segmented_point_cloud_cameras:
            camera = self._camera_dict[camera_name]
            xyz = camera.depth_to_xyz(self.env_ptrs, env_ids)

            for env_id in env_ids:
                self.segtrackers[camera_name][env_id].restart_tracker()

                self.input_points = []
                self.input_labels = []
                self.pred_mask = None

                color_image_numpy = (image_dict[camera_name][env_id].detach().cpu().numpy()[..., 0:3] * 255).astype(np.uint8)
                depth_image = image_dict[camera_name][env_id][..., 3]

                def update_mask(event):
                    ax.cla()
                    self.input_points.append([event.xdata, event.ydata])
                    self.input_labels.append(int(event.button == MouseButton.LEFT))
                    for point, label in zip(self.input_points, self.input_labels):
                        col = 'green' if label == 1 else 'red'
                        plt.scatter(point[0], point[1], color=col, edgecolors='white', s=50)

                    self.pred_mask, masked_frame = self.segtrackers[camera_name][env_id].refine_first_frame_click(
                        color_image_numpy.copy(), np.array(self.input_points).astype(np.int), np.array(self.input_labels), True)
                    self.segtrackers[camera_name][env_id].sam.reset_image()

                    torch.cuda.empty_cache()
                    gc.collect()
                    selected_segmentation = draw_mask(color_image_numpy.copy(), self.pred_mask, id_countour=True)

                    ax.imshow(selected_segmentation)
                    fig.canvas.draw()

                fig = plt.figure()
                plt.axis('off')
                ax = fig.add_subplot(111)
                ax.set_title(f'Select segmentation for camera {camera_name} on env {env_id}.')
                ax.imshow(color_image_numpy)
                cid = fig.canvas.mpl_connect('button_press_event', update_mask)
                plt.show()

                idxs = np.where(self.pred_mask.flatten())
                print("idxs:", idxs)

                print("xyz.shape:", xyz.shape)


                getattr(self, 'detected_segmented_point_cloud_' + camera_name + '_testing')[env_id] = xyz[env_id][idxs]



                #self.detected_segemented_point_clouds[camera_name][env_id] = xyz[env_id][idxs]

                # Set tracker reference once the segmentation is selected.
                self.segtrackers[camera_name][env_id].add_reference(
                    color_image_numpy, self.pred_mask)

    def _acquire_segtracker(self):
        from SegTracker import SegTracker
        from model_args import aot_args, sam_args, segtracker_args

        sam_args['generator_args'] = {
            'points_per_side': 30,
            'pred_iou_thresh': 0.8,
            'stability_score_thresh': 0.9,
            'crop_n_layers': 1,
            'crop_n_points_downscale_factor': 2,
            'min_mask_region_area': 200,
        }
        sam_args['sam_checkpoint'] = '/home/user/mosbach/tools/sam_tracking/sam_tracking/ckpt/sam_vit_b_01ec64.pth'
        aot_args['model_path'] = '/home/user/mosbach/tools/sam_tracking/sam_tracking/ckpt/R50_DeAOTL_PRE_YTB_DAV.pth'

        # Create a tracker per camera and environment.
        self.segtrackers = {}
        for camera_name in self.segmented_point_cloud_cameras:
            self.segtrackers[camera_name] = [SegTracker(segtracker_args, sam_args, aot_args) for _ in range(self.num_envs)]

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

        # Refresh synthetic 2D bounding boxes (image space).
        if any(obs.startswith("synthetic_2d_bounding_box_image") for obs in self.cfg["env"]["observations"]):
            self._refresh_synthetic_2d_bounding_box_image()

        # Refresh synthetic 2D bounding boxes (image space).
        if any(obs.startswith("synthetic_2d_bounding_box_world") for obs in self.cfg["env"]["observations"]):
            self._refresh_synthetic_2d_bounding_box_world()

        # Refresh 2D bounding boxes detected by SAM.
        if any(obs.startswith("detected_2d_bounding_box_image") for obs in self.cfg["env"]["observations"]):
            self._refresh_detected_2d_bounding_box_image()

        # Refresh segmented point-clouds enabled by SAM.
        if any(obs.startswith("detected_segmented_point_cloud") for obs in
               self.cfg["env"]["observations"]):
            self._refresh_detected_segmented_point_cloud()

    def _get_random_drop_pos(self, env_ids) -> torch.Tensor:
        object_pos_drop = torch.tensor(
            self.object_pos_drop, device=self.device
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
        x_lower = self.workspace_extent_xy[0][0] + self.object_pos_drop[0]
        x_upper = self.workspace_extent_xy[1][0] + self.object_pos_drop[0]
        y_lower = self.workspace_extent_xy[0][1] + self.object_pos_drop[1]
        y_upper = self.workspace_extent_xy[1][1] + self.object_pos_drop[1]
        object_in_workspace = x_lower <= object_pos[:, 0]
        object_in_workspace = torch.logical_and(
            object_in_workspace, object_pos[:, 0] <= x_upper)
        object_in_workspace = torch.logical_and(
            object_in_workspace, y_lower <= object_pos[:, 1])
        object_in_workspace = torch.logical_and(
            object_in_workspace, object_pos[:, 1] <= y_upper)
        return object_in_workspace

    def visualize_workspace_xy(self, env_id: int) -> None:
        # Set extent in z-direction to 0
        lower = self.workspace_extent_xy[0] + [0.]
        upper = self.workspace_extent_xy[1] + [0.]
        extent = torch.tensor([lower, upper])
        drop_pose = gymapi.Transform(
            p=gymapi.Vec3(self.cfg_task.randomize.object_pos_drop[0],
                          self.cfg_task.randomize.object_pos_drop[1],
                          0.))
        bbox = gymutil.WireframeBBoxGeometry(extent, pose=drop_pose,
                                             color=(0, 1, 1))
        gymutil.draw_lines(bbox, self.gym, self.viewer, self.env_ptrs[env_id],
                           pose=gymapi.Transform())

    def visualize_synthetic_2d_bounding_box_world_closeup(self, env_id: int) -> None:
        self.visualize_pos('synthetic_2d_bounding_box_world_closeup', env_id)
        self.visualize_polygon('synthetic_2d_bounding_box_world_closeup', env_id)

    def visualize_detected_segmented_point_cloud_closeup(self, env_id: int) -> None:
        if getattr(self, 'detected_segmented_point_cloud_closeup_testing')[env_id] is not None:
            self.visualize_pos('detected_segmented_point_cloud_closeup_testing', env_id)

            self.visualize_pos('detected_segmented_point_cloud_closeup_testing_subset', env_id, color=(1, 0, 1))

    def visualize_real_robot_table(self, env_id: int) -> None:
        extent = torch.tensor([[-0.07, -0.17, 0.], [0.63, 0.83, 0.]])
        bbox = gymutil.WireframeBBoxGeometry(extent, pose=gymapi.Transform(),
                                             color=(0.8, 0.35, 0.))
        gymutil.draw_lines(bbox, self.gym, self.viewer, self.env_ptrs[env_id],
                           pose=gymapi.Transform())


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))
