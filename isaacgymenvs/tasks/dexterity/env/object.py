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
from matplotlib.widgets import TextBox
import os
import rospy
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
    
    @property
    def surface_area(self) -> float:
        if not hasattr(self, "mesh"):
            self.acquire_mesh()
        return self.mesh.area

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

        self.synthetic_pointcloud_dimension = 64
        self.max_num_points_padded = 128
        self.pick_detections = True

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

        else:
            raise NotImplementedError
        
        return num_observations

    def compute_observations(self):
        super().compute_observations()

        # Synthetic point-clouds of uniform size sampled on the object's surface.
        #if any(obs.startswith("synthetic_pointcloud") for obs in self.cfg["env"]["observations"]):
        #    self.padded_pointcloud[:, 0:self.synthetic_pointcloud.shape[1], :] = self.synthetic_pointcloud
        #    self.obs_buf = torch.cat([self.obs_buf, self.padded_pointcloud.flatten(1, 2)], dim=1)
        #    print("adding synthetic pointcloud to obs_dict")

        # Detected point-clouds acquired through an instance segmentation pipeline (SAM) from visual observations.
        #if any(obs.startswith("detected_pointcloud") for obs in self.cfg["env"]["observations"]):
        #    self.obs_dict["detected_pointcloud"] = self.detected_pointcloud.to(self.rl_device)

    def _acquire_env_tensors(self):
        """Acquire and wrap tensors. Create views."""
        self.object_pos = self.root_pos[:, self.object_actor_id_env, 0:3]
        self.object_quat = self.root_quat[:, self.object_actor_id_env, 0:4]
        self.object_linvel = self.root_linvel[:, self.object_actor_id_env, 0:3]
        self.object_angvel = self.root_angvel[:, self.object_actor_id_env, 0:3]

        self._acquire_pointcloud_tensors()

    def _acquire_pointcloud_tensors(self) -> None:
        # Synthetic point-clouds.
        if any(obs.startswith("synthetic_pointcloud") for obs in self.cfg["env"]["observations"]):
            self.synthetic_pointcloud = self._acquire_synthetic_pointcloud()

        self.rendered_pointcloud_camera_names = []
        self.detected_pointcloud_camera_names = []
        for obs in self.cfg["env"]["observations"]:
            # Rendered point-clouds.
            if obs.startswith("rendered_pointcloud"):
                camera_name = obs[len("rendered_pointcloud_"):]
                assert camera_name in self.cfg["env"]["observations"], \
                    f"Cannot use observation '{obs}' if camera {camera_name} is not part of the observations."
                assert self._camera_dict[camera_name].image_type == 'pc_seg', \
                    f"The image type of the camera '{camera_name}' used in '{obs}' must be 'pc_seg', but found '{self._camera_dict[camera_name].image_type}' instead."
                self._acquire_rendered_pointcloud(camera_name)
                self.rendered_pointcloud_camera_names.append(camera_name)
            # Detected point-clouds.
            if obs.startswith("detected_pointcloud"):
                camera_name = obs[len("detected_pointcloud_"):]
                assert camera_name in self.cfg["env"]["observations"], \
                    f"Cannot use observation '{obs}' if camera {camera_name} is not part of the observations."
                if self.pick_detections:
                    assert self._camera_dict[camera_name].image_type.startswith('rgbxyz'), \
                        f"The image type of the camera '{camera_name}' used in '{obs}' must be 'rgbxyz', but found '{self._camera_dict[camera_name].image_type}' instead."
                else:
                    assert self._camera_dict[camera_name].image_type == 'rgbxyzseg', \
                        f"The image type of the camera '{camera_name}' used in '{obs}' must be 'rgbxyzseg', but found '{self._camera_dict[camera_name].image_type}' instead."
                self._acquire_detected_pointcloud(camera_name)
                self.detected_pointcloud_camera_names.append(camera_name)

    def _acquire_synthetic_pointcloud(self, num_samples: int = 64, sample_mode: str = 'area') -> torch.Tensor:
        """Acquire position of the points relative to the mesh origin (object_mesh_sample_pos) and return a
        placeholder for the absolute position of the points in space (object_synthetic_pointcloud_pos) to be updated
        by _refresh_object_synthetic_pointcloud().
        """
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

        num_repeats = math.ceil(self.num_envs / len(self.objects))
        self.object_mesh_samples_pos = self.object_mesh_samples_pos.repeat(num_repeats, 1, 1)[:self.num_envs]
        
        object_synthetic_pointcloud_pos = self.object_mesh_samples_pos.detach().clone()
        return object_synthetic_pointcloud_pos
    
    def _acquire_rendered_pointcloud(self, camera_name: str):
        setattr(self, f'rendered_pointcloud_{camera_name}', torch.zeros((self.num_envs, self.max_num_points_padded, 4)).to(self.device))

    def _acquire_detected_pointcloud(self, camera_name: str):
        """Detected point-clouds are stored in a list over environments for each camera because they have varying
        sizes/numbers of points."""
        setattr(self, f'detected_pointcloud_{camera_name}', torch.zeros((self.num_envs, self.max_num_points_padded, 4)).to(self.device))

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
        self.object_id = []  # identifier of each object in the used_objects list

        actor_count = 0
        for i in range(self.num_envs):
            # Create new env
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Loop through all used objects
            object_idx = i % self.object_count
            used_object = self.objects[object_idx]
            self.object_id.append(object_idx)

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
        
        # For identifying with object is present.
        self.object_id = torch.tensor(self.object_id, dtype=torch.float, device=self.device).unsqueeze(1)

    def _refresh_synthetic_pointcloud(self, object_name: str = 'object',
                                      mesh_samples_name: str = 'object_mesh_samples') -> None:
        """Update the relative position of the sampled points (object_mesh_samples_pos) by the current 
        object pose."""
        object_pos = getattr(self, object_name + '_pos')
        object_quat = getattr(self, object_name + '_quat')
        mesh_samples_pos = getattr(self, mesh_samples_name + '_pos')

        num_samples = self.synthetic_pointcloud.shape[1]
        self.synthetic_pointcloud[..., 0:3] = object_pos.unsqueeze(1).repeat(
            1, num_samples, 1) + quat_apply(object_quat.unsqueeze(1).repeat(1, num_samples, 1),
            mesh_samples_pos[..., 0:3])
    
    def _refresh_rendered_pointcloud(self, target_segmentation_id: Union[int, List[int]] = 2, draw_debug_visualization: bool = False) -> None:
        if isinstance(target_segmentation_id, int):
            target_segmentation_id = [target_segmentation_id] * self.num_envs
        
        # Retrieve current camera images without debug visualizations.
        if (self.headless or len(self.cfg_base.debug.visualize) == 0) and 'image' in self.obs_dict.keys():
            image_dict = self.obs_dict['image']
        else:
            self.gym.clear_lines(self.viewer)
            image_dict = self.get_images()

        for camera_name in self.rendered_pointcloud_camera_names:
            xyz = image_dict[camera_name][..., 0:3]
            xyz_1 = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)  # Add valid value id to xyz
            segmentation_ids = image_dict[camera_name][..., 3]


            rendered_pointclouds = []
            for i in range(self.num_envs):
                mask = segmentation_ids[i] == target_segmentation_id[i]
                points_on_object = xyz_1[i, mask].clone().detach()
                # More points are on the object than can fit into the padded point-cloud tensor. Subsample points.
                if points_on_object.shape[0] > self.max_num_points_padded:
                    perm = torch.randperm(points_on_object.shape[0])
                    sub_idx = perm[:self.max_num_points_padded]
                    points_on_object = points_on_object[sub_idx]
                # Fewer points than the padded point-cloud tensor. Pad with zeros.
                else:
                    points_on_object = torch.cat([points_on_object, torch.zeros(self.max_num_points_padded - points_on_object.shape[0], 4, device=self.device)], dim=0)
                rendered_pointclouds.append(points_on_object)
            getattr(self, f'rendered_pointcloud_{camera_name}')[:] = torch.stack(rendered_pointclouds, dim=0)
                    
            if draw_debug_visualization:
                max_envs_to_show = 4
                    # Initialize figure.
                if not hasattr(self, "rendered_pointcloud_fig"):
                    self.rendered_pointcloud_fig = plt.figure()
                    self.rendered_pointcloud_axs = []
                    for env_id in range(min(self.num_envs, max_envs_to_show)):
                        self.rendered_pointcloud_axs.append(self.rendered_pointcloud_fig.add_subplot(2, 2, env_id + 1, projection='3d'))
                    plt.show(block=False)

                # Update figure.
                for env_id in range(min(self.num_envs, max_envs_to_show)):
                    self.rendered_pointcloud_axs[env_id].cla()
                    self.rendered_pointcloud_axs[env_id].set_ylim(0.38, 0.78)
                    self.rendered_pointcloud_axs[env_id].set_xlim(0.0, 0.5)
                    self.rendered_pointcloud_axs[env_id].set_zlim(0., 0.5)
                    self.rendered_pointcloud_axs[env_id].set_box_aspect([ub - lb for lb, ub in (getattr(self.rendered_pointcloud_axs[env_id], f'get_{a}lim')() for a in 'xyz')])
                    self.rendered_pointcloud_axs[env_id].set_title(f'env_id: {env_id}')
                    self.rendered_pointcloud_axs[env_id].scatter(getattr(self, f"rendered_pointcloud_{camera_name}")[env_id, :, 0].cpu(), 
                                                                 getattr(self, f"rendered_pointcloud_{camera_name}")[env_id, :, 1].cpu(), 
                                                                 getattr(self, f"rendered_pointcloud_{camera_name}")[env_id, :, 2].cpu())
                self.rendered_pointcloud_fig.canvas.draw()
                plt.pause(0.01)

    def _refresh_detected_pointcloud(self, draw_debug_visualization: bool = False) -> None:
        if not hasattr(self, 'segtrackers'):
            return

        # Retrieve current camera images without debug visualizations.
        if (self.headless or len(self.cfg_base.debug.visualize) == 0) and 'image' in self.obs_dict.keys():
            image_dict = self.obs_dict['image']
        else:
            self.gym.clear_lines(self.viewer)
            image_dict = self.get_images()

        for camera_name in self.detected_pointcloud_camera_names:
            xyz = image_dict[camera_name][..., 3:6].flatten(1, 2)
            xyz_1 = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)  # Add valid value id to xyz

            detected_pointclouds = []
            for env_id in range(self.num_envs):
                color_image_numpy = (image_dict[camera_name][env_id].detach().cpu().numpy()[..., 0:3] * 255).astype(
                    np.uint8)
                self.pred_mask = self.segtrackers[camera_name][env_id].track(color_image_numpy, update_memory=True)
                pred_idx = np.where(self.pred_mask.flatten())[0]
                points_on_object = xyz_1[env_id][pred_idx]

                # Transform points to base_link frame for ROS-Cameras and remove invalid points.
                if camera_name  in self.cfg_env.ros_cameras.keys():
                    # Remove invalid points that now lie at [0, 0, 0].
                    points_on_object = points_on_object[torch.norm(points_on_object[:, 0:3], dim=1) > 0.001]
                    trans, rot = self.tf_sub.lookupTransform('base_link', self._camera_dict[camera_name].xyz_sub.image.header.frame_id, rospy.Time(0))
                    tm = torch.from_numpy(self.tf_sub.fromTranslationRotation(trans, rot)).float().to(self.device)
                    points_on_object = torch.matmul(tm, points_on_object.t()).t()
                    

                # More points are on the object than can fit into the padded point-cloud tensor. Subsample points.
                if points_on_object.shape[0] > self.max_num_points_padded:
                    perm = torch.randperm(points_on_object.shape[0])
                    sub_idx = perm[:self.max_num_points_padded]
                    points_on_object = points_on_object[sub_idx]
                # Fewer points than the padded point-cloud tensor. Pad with zeros.
                else:
                    points_on_object = torch.cat([points_on_object, torch.zeros(self.max_num_points_padded - points_on_object.shape[0], 4, device=self.device)], dim=0)
                detected_pointclouds.append(points_on_object)
            getattr(self, f'detected_pointcloud_{camera_name}')[:] = torch.stack(detected_pointclouds, dim=0)

            if draw_debug_visualization or self.cfg_base.debug.save_videos:
                tracked_segmentation = draw_mask(color_image_numpy.copy(), self.pred_mask, id_countour=False)

                if self.cfg_base.debug.save_videos:
                    self._segmented_frames[camera_name][env_id].append(tracked_segmentation)
                    self._original_frames[camera_name][env_id].append(color_image_numpy)

                if draw_debug_visualization:
                    if self.progress_buf[0] == 1:
                        self.segm_fig[camera_name][env_id] = plt.figure()
                        ax = self.segm_fig[camera_name][env_id].add_subplot(111)
                        ax.cla()
                        ax.set_title(
                            f"Tracked target object for camera '{camera_name}' on env {env_id}.")
                        self.tracked_image_debug[camera_name][env_id] = ax.imshow(tracked_segmentation)
                        plt.axis('off')
                        plt.show(block=False)
                    else:
                        self.tracked_image_debug[camera_name][env_id].set_data(tracked_segmentation)
                        self.segm_fig[camera_name][env_id].canvas.draw()
                        plt.pause(0.01)

    def _reset_segmentation_tracking(self, env_ids, target_segmentation_id: Union[int, List[int]] = 2, draw_debug_visualization: bool = False):
        if isinstance(target_segmentation_id, int):
            target_segmentation_id = [target_segmentation_id] * self.num_envs


        # Write videos of segmentation tracking.
        if hasattr(self, '_segmented_frames'):
            for camera_name in self.detected_pointcloud_camera_names:
                for env_id in env_ids:
                    if len(self._segmented_frames[camera_name][env_id]) > 0:
                        segmented_video_writer = cv2.VideoWriter(f'{self.videos_dir}/tracked_segmentation_{camera_name}_env_{env_id}_episode_{self._episodes[env_id] - 1}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (self._segmented_frames[camera_name][env_id][0].shape[1], self._segmented_frames[camera_name][env_id][0].shape[0]))
                        original_video_writer = cv2.VideoWriter(f'{self.videos_dir}/original_{camera_name}_env_{env_id}_episode_{self._episodes[env_id] - 1}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (self._segmented_frames[camera_name][env_id][0].shape[1], self._segmented_frames[camera_name][env_id][0].shape[0]))
                        for i in range(len(self._segmented_frames[camera_name][env_id])):
                            segmented_video_writer.write(self._segmented_frames[camera_name][env_id][i][..., ::-1])
                            original_video_writer.write(self._original_frames[camera_name][env_id][i][..., ::-1])
                        segmented_video_writer.release()          
                        original_video_writer.release()
        self._segmented_frames = {camera_name: [[] for _ in range(self.num_envs)] for camera_name in self.detected_pointcloud_camera_names}
        self._original_frames = {camera_name: [[] for _ in range(self.num_envs)] for camera_name in self.detected_pointcloud_camera_names}

        if not hasattr(self, 'segtracker'):
            self._acquire_segtracker()

        plt.close('all')

        image_dict = self.get_images()
        for camera_name in self.detected_pointcloud_camera_names:
            for env_id in env_ids:
                self.segtrackers[camera_name][env_id].restart_tracker()
                color_image_numpy = (image_dict[camera_name][env_id].detach().cpu().numpy()[..., 0:3] * 255).astype(np.uint8)
                self.pred_mask = None

                # Pick target object manually.
                if self.pick_detections:
                    self.input_points = []
                    self.input_labels = []

                    def update_mask_on_click(event):
                        if event.inaxes == ax_image:    
                            self.input_points.append([event.xdata, event.ydata])
                            self.input_labels.append(int(event.button == MouseButton.LEFT))
                            

                            self.pred_mask, masked_frame = self.segtrackers[camera_name][env_id].seg_acc_click(
                                color_image_numpy.copy(), np.array(self.input_points).astype(np.int), np.array(self.input_labels), True)

                            torch.cuda.empty_cache()
                            gc.collect()
                            selected_segmentation = draw_mask(color_image_numpy.copy(), self.pred_mask, id_countour=True)

                            ax_image.imshow(selected_segmentation)

                            for point, label in zip(self.input_points, self.input_labels):
                                col = 'green' if label == 1 else 'red'
                                ax_image.scatter(point[0], point[1], color=col, edgecolors='white', s=50)
                                
                            self.segm_fig[camera_name][env_id].canvas.draw()

                    def update_mask_on_text(text):
                        if text not in ['', 'Left-click to add positive marker, right-click to add negative marker.']:
                            if len(self.input_points) > 0:
                                print("Cannot use text input when markers have already been added.")
                            else:
                                self.pred_mask, masked_frame = self.segtrackers[camera_name][env_id].detect_and_seg(color_image_numpy.copy(), text, 0.25, 0.25)
                                selected_segmentation = draw_mask(color_image_numpy.copy(), self.pred_mask, id_countour=True)
                                ax_image.imshow(selected_segmentation)
                                #ax_image.axis('off')
                                self.segm_fig[camera_name][env_id].canvas.draw()

                    def on_hover_over_image(event):
                        if event.inaxes == ax_image and event.xdata is not None and event.ydata is not None:
                            if text_box.text == "":
                                text_box.set_val('Left-click to add positive marker, right-click to add negative marker.')
                        else:
                            if text_box.text == 'Left-click to add positive marker, right-click to add negative marker.':
                                text_box.set_val('')
                        self.segm_fig[camera_name][env_id].canvas.draw()

                    self.segm_fig[camera_name][env_id] = plt.figure(num=f"Select target object for camera '{camera_name}' on env {env_id}.")
                    ax_image = self.segm_fig[camera_name][env_id].add_subplot(111)
                    ax_image.axis('off')

                    self.segm_fig[camera_name][env_id].subplots_adjust(bottom=0.2)

                    ax_prompt = plt.axes([0.1, 0.05, 0.8, 0.075])

                    text_box = TextBox(ax_prompt, "Text prompt:", initial="")
                    text_box.on_submit(update_mask_on_text)

                    ax_image.imshow(color_image_numpy)
                    self.segm_fig[camera_name][env_id].canvas.mpl_connect('button_press_event', update_mask_on_click)
                    self.segm_fig[camera_name][env_id].canvas.mpl_connect('motion_notify_event', on_hover_over_image)
                    plt.show()

                # Target object is picked automatically based on points sampled on ground truth segmentation.
                else:
                    segmentation_image = image_dict[camera_name][env_id][..., 6]
                    target_segmentation = segmentation_image == target_segmentation_id[env_id]
                    assert torch.any(target_segmentation), f"Target object is initially fully occluded in env {env_id}."

                    grid_y, grid_x = torch.meshgrid(torch.arange(segmentation_image.shape[0]), torch.arange(segmentation_image.shape[1]))
                    grid_x = grid_x.to(self.device)
                    grid_y = grid_y.to(self.device)
                    mean_x = grid_x[target_segmentation].float().mean()
                    mean_y = grid_y[target_segmentation].float().mean()
                    std_x = (torch.max(grid_x[target_segmentation].float()) - torch.min(grid_x[target_segmentation].float())) / 1.5 #0.75
                    std_y = (torch.max(grid_y[target_segmentation].float()) - torch.min(grid_y[target_segmentation].float())) / 1.5 #0.75

                    x_samples = torch.linspace(mean_x - std_x, mean_x + std_x, 6).clamp(0, segmentation_image.shape[1] - 1).int()
                    y_samples = torch.linspace(mean_y - std_y, mean_y + std_y, 6).clamp(0, segmentation_image.shape[0] - 1).int()

                    input_points = torch.stack(torch.meshgrid(x_samples, y_samples), dim=-1).reshape(-1, 2).numpy()

                    #input_points = torch.stack([x_samples, y_samples], dim=1).numpy()
                    input_labels = []
                    for point in input_points:
                        input_labels.append(segmentation_image[point[1], point[0]].item() == target_segmentation_id[env_id])
                    input_labels = np.array(input_labels)

                    #assert np.any(input_labels), f"Not a single sampled point is on the target object in env {env_id}."

                    self.pred_mask, masked_frame = self.segtrackers[camera_name][env_id].refine_first_frame_click(
                        color_image_numpy.copy(), input_points, input_labels, True)
                    self.segtrackers[camera_name][env_id].sam.reset_image()
                
                    if draw_debug_visualization:
                        selected_segmentation = draw_mask(color_image_numpy.copy(), self.pred_mask, id_countour=True)
                        self.segm_fig[camera_name][env_id] = plt.figure()
                        ax_segm = self.segm_fig[camera_name][env_id].add_subplot(121)
                        ax_color = self.segm_fig[camera_name][env_id].add_subplot(122)
                        ax_segm.set_title(f"Sampled input points for '{camera_name}' on env {env_id}.")
                        ax_segm.imshow(segmentation_image.cpu())
                        for point, label in zip(input_points, input_labels):
                            col = 'green' if label == 1 else 'red'
                            ax_segm.scatter(point[0], point[1], color=col, edgecolors='white', s=50)
                        ax_color.set_title(f"Resulting segmentation for '{camera_name}' on env {env_id}.")
                        ax_color.imshow(selected_segmentation)
                        plt.show()



                # Set tracker reference once the segmentation is selected.
                assert self.pred_mask is not None, f"Segmentation for '{camera_name}' on env {env_id} is not selected."
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

        segtracker_args['sam_gap'] = 9999   # We don't need SAM to detect newly-appearing objects.

        # Create a tracker per camera and environment.
        self.segtrackers, self.segm_fig, self.tracked_image_debug = {}, {}, {}
        for camera_name in self.detected_pointcloud_camera_names:
            self.segtrackers[camera_name] = [SegTracker(segtracker_args, sam_args, aot_args) for _ in range(self.num_envs)]
            self.segm_fig[camera_name] = [None for _ in range(self.num_envs)]
            self.tracked_image_debug[camera_name] = [None for _ in range(self.num_envs)]

    def refresh_env_tensors(self):
        """Refresh tensors."""
        # NOTE: Tensor refresh functions should be called once per step, before
        # setters.
        # NOTE: Since the object_pos, object_quat etc. are obtained from the
        # root state tensor through regular slicing, they are views rather than
        # separate tensors and hence don't have to be updated separately.
        self._refresh_pointcloud_tensors()

    def _refresh_pointcloud_tensors(self):
        if "synthetic_pointcloud" in self.cfg["env"]["observations"]:
            self._refresh_synthetic_pointcloud()
        
        if any(obs.startswith("rendered_pointcloud") for obs in self.cfg["env"]["observations"]):
            self._refresh_rendered_pointcloud()

        if any(obs.startswith("detected_pointcloud") for obs in self.cfg["env"]["observations"]):
            self._refresh_detected_pointcloud()

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
    
    def _disable_object_collisions(self, object_ids: List[int] = None):
        self._set_object_collisions(object_ids, collision_filter=-1)

    def _enable_object_collisions(self, object_ids: List[int] = None):
        self._set_object_collisions(object_ids, collision_filter=0)

    def _set_object_collisions(self, object_ids: List[int], collision_filter: int) -> None:
        def set_collision_filter(env_id: int, actor_handle, collision_filter: int) -> None:
            actor_shape_props = self.gym.get_actor_rigid_shape_properties(
                self.env_ptrs[env_id], actor_handle)
            for shape_id in range(len(actor_shape_props)):
                actor_shape_props[shape_id].filter = collision_filter
                self.gym.set_actor_rigid_shape_properties(
                    self.env_ptrs[env_id], actor_handle, actor_shape_props)

        # No tensor API to set actor rigid shape props, so a loop is required
        for env_id in range(self.num_envs):
            if object_ids is not None:
                for object_id in object_ids:
                    set_collision_filter(env_id, self.object_handles[env_id][object_id], collision_filter)
            else:
                set_collision_filter(env_id, self.object_handles[env_id], collision_filter)

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

    def visualize_synthetic_pointcloud(self, env_id: int) -> None:
        unpadded_pointclouds = []
        for i in range(self.num_envs):
            synthetic_pointcloud = self.synthetic_pointcloud[i]
            mask = synthetic_pointcloud[:, 3] > 0
            unpadded_pointcloud = synthetic_pointcloud[mask, 0:3]
            unpadded_pointclouds.append(unpadded_pointcloud)
        self.visualize_pos(unpadded_pointclouds, env_id, color=(1, 0, 1))

    def visualize_rendered_pointcloud(self, env_id: int) -> None:
        for camera_name in self._camera_dict.keys():
            if hasattr(self, f'rendered_pointcloud_{camera_name}'):
                if env_id == 0:
                    self.visualization_rendered_unpadded_pointclouds = []
                    for i in range(self.num_envs):
                        rendered_pointcloud = getattr(self, f'rendered_pointcloud_{camera_name}')[i].detach().clone()
                        mask = rendered_pointcloud[:, 3] > 0
                        unpadded_pointcloud = rendered_pointcloud[mask, 0:3]
                        self.visualization_rendered_unpadded_pointclouds.append(unpadded_pointcloud)
                self.visualize_pos(self.visualization_rendered_unpadded_pointclouds, env_id, color=(1, 0, 1))

    def visualize_detected_pointcloud(self, env_id: int) -> None:
        for camera_name in self._camera_dict.keys():
            if hasattr(self, f'detected_pointcloud_{camera_name}'):
                if env_id == 0:
                    self.visualization_detected_unpadded_pointclouds = []
                    for i in range(self.num_envs):
                        detected_pointcloud = getattr(self, f'detected_pointcloud_{camera_name}')[i]
                        mask = detected_pointcloud[:, 3] > 0
                        unpadded_pointcloud = detected_pointcloud[mask, 0:3]
                        self.visualization_detected_unpadded_pointclouds.append(unpadded_pointcloud)
                self.visualize_pos(self.visualization_detected_unpadded_pointclouds, env_id, color=(1, 0, 1))

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
