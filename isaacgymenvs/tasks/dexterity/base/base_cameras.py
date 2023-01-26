import cv2
import hydra
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os
from pytorch3d.renderer import (
                look_at_view_transform,
                FoVOrthographicCameras,
                PointsRasterizationSettings,
                PointsRasterizer,
                PulsarPointsRenderer)
from pytorch3d.structures import Pointclouds
from typing import *


IMAGE_TYPES = ['d', 'rgb', 'rgbd', 'seg', 'pc', 'cpc']


@torch.jit.script
def depth_image_to_xyz(depth_image, proj_mat, view_mat, device: torch.device):
    batch_size, width, height = depth_image.shape
    sparse_depth = depth_image.to(device).to_sparse()
    indices = sparse_depth.indices()
    values = sparse_depth.values()
    xy_depth = torch.cat([indices.T[:, 1:].flip(1), values[..., None]], dim=-1)

    center_u = width / 2
    center_v = height / 2

    xy_depth[:, 0] = -(xy_depth[:, 0] - center_u) / width
    xy_depth[:, 1] = (xy_depth[:, 1] - center_v) / height
    xy_depth[:, 0] *= xy_depth[:, 2]
    xy_depth[:, 1] *= xy_depth[:, 2]

    x2 = xy_depth @ proj_mat
    x2_hom = torch.cat([x2, torch.ones_like(x2[:, 0:1])], dim=1).view(
        batch_size, -1, 4)
    xyz = torch.bmm(x2_hom, view_mat.inverse())[..., 0:3]
    return xyz


class DexterityCamera:
    """Wrapper class for IsaacGym camera sensors.

    Args:
        gym: (Gym) Gym object.
        sim: (Sim) IsaacGym simulation object.
        env_ptrs: (List[int]) List of pointers to the environments the
            simulation contains.
        pos: (Tuple[float, float, float]) Position of the camera sensor.
        quat: (Tuple[float, float, float, float]) Quaternion rotation of the
            camera sensor.
        model: (str, optional) Model of the camera (e.g. 'realsense_d405'). If
            the camera model is specified, the properties of the sensor (fovx,
            resolution, and image_type) are inferred from its camera_info.yaml.
        fovx: (int) Horizontal field of view.
        resolution: (Tuple[int, int]) Resolution (width, height) of the camera.
        image_type: (str) Type of image that the camera sensor returns. Should
            be in ['d', 'rgb', 'rgbd', 'seg'].
        attach_to_body: (str, optional) Specifies the name of a rigid body the
            camera should be attached to. Camera will be static if
            attach_to_body is None.
        use_camera_tensors: (bool, True) Whether to use GPU tensor access when
            retrieving the camera images.
        add_camera_actor: (bool, True) Whether to add an actor of the camera
            model to the simulation. A URDF description of the camera model
            must exist to do so.
    """
    def __init__(
            self,
            pos: Tuple[float, float, float],
            quat: Tuple[float, float, float, float],
            model: str = None,
            fovx: int = None,
            resolution: Tuple[int, int] = None,
            image_type: str = None,
            attach_to_body: str = None,
            use_camera_tensors: bool = True,
            add_camera_actor: bool = True,
    ) -> None:
        self._pos = pos
        self._quat = quat
        self._model = model
        self._attach_to_body = attach_to_body
        self._use_camera_tensors = use_camera_tensors
        self._add_camera_actor = add_camera_actor

        if self._model is not None:
            self._camera_properties_from_model()
        else:
            self._camera_properties_from_parameters(
                fovx, resolution, image_type)
        assert self._image_type in IMAGE_TYPES, \
            f"Image type should be one of {IMAGE_TYPES}, but unknown type " \
            f"'{image_type}' was found."

        self._camera_handles = []

    def _camera_properties_from_model(self) -> None:
        # Check whether camera model exists
        self._cameras_asset_root = os.path.join(
            os.path.dirname(__file__),
            '../../../../assets/dexterity/cameras')
        available_camera_models = [
            f.name for f in os.scandir(self._cameras_asset_root)
            if f.is_dir()]
        assert self._model in available_camera_models, \
            f"Camera model should be one of {available_camera_models}," \
            f" but unknown model '{self._model}' was found."

        # Retrieve and set camera properties
        camera_info_file = f'{self._model}/camera_info.yaml'
        camera_info = hydra.compose(
            config_name=os.path.join(
                '../../assets/dexterity/cameras', camera_info_file))
        camera_info = camera_info['']['']['']['']['']['']['assets'][
            'dexterity']['cameras'][self._model]  # strip superfluous nesting
        self._fovx = camera_info.fovx
        self._resolution = camera_info.resolution
        self._image_type = camera_info.image_type

    def _camera_properties_from_parameters(
            self, fovx: int, resolution: Tuple[int, int], image_type: str
    ) -> None:
        assert None not in [fovx, resolution, image_type], \
            "fovx, resolution, and image_type must be specified if no " \
            "camera model is given."
        self._fovx = fovx
        self._resolution = resolution
        self._image_type = image_type

    @property
    def camera_type(self) -> str:
        return self._image_type

    @property
    def width(self) -> int:
        return self._resolution[0]

    @property
    def height(self) -> int:
        return self._resolution[1]

    @property
    def camera_props(self) -> gymapi.CameraProperties:
        camera_props = gymapi.CameraProperties()
        camera_props.width = self._resolution[0]
        camera_props.height = self._resolution[1]
        camera_props.horizontal_fov = self._fovx
        camera_props.enable_tensors = self._use_camera_tensors
        return camera_props

    @property
    def camera_transform(self) -> gymapi.Transform:
        """Returns absolute transform if the camera is static and local
        transform to parent body, if the camera is attached to a body."""
        pos = gymapi.Vec3(*self._pos)
        quat = gymapi.Quat(*self._quat)
        return gymapi.Transform(p=pos, r=quat)

    @property
    def pc_renderer(self) -> PulsarPointsRenderer:
        """Returns PyTorch3D renderer that converts point-clouds to images."""
        if not hasattr(self, "_pc_renderer"):
            R, T = look_at_view_transform(
                eye=((0.75, -0.75, 0.5),), up=((0, 0, 1),),
                at=((0., 0., 0.2),),)
            R, T = R.repeat(self._num_envs, 1, 1), T.repeat(self._num_envs, 1)

            cameras = FoVOrthographicCameras(
                device=self._device, R=R, T=T, znear=0.01)
            raster_settings = PointsRasterizationSettings(
                image_size=(1080, 1920), radius=0.003, points_per_pixel=1)
            self._pc_renderer = PulsarPointsRenderer(
                rasterizer=PointsRasterizer(
                    cameras=cameras, raster_settings=raster_settings),
                n_channels=3).to(self._device)
        return self._pc_renderer

    def get_projection_matrix(self, gym, sim, env_ptr, i) -> torch.Tensor:
        return torch.from_numpy(
            gym.get_camera_proj_matrix(
                sim, env_ptr, self._camera_handles[i])).to(self._device)

    def get_view_matrix(self, gym, sim, env_ptrs: List[int], i: List[int]
                        ) -> torch.Tensor:
        """Returns the batch of view matrices of shape: [len(env_ptrs), 4, 4].
        The camera view matrix appear to be weird in IsaacGym since it is
        returned in global instead of env coordinates. Querying the view matrix
        for the first environments and the appropriate camera handle seems to
        result in the desired view matrices."""
        view_mat = []
        for env_ptr, env_idx in zip(env_ptrs, i):
            view_mat.append(torch.from_numpy(
                gym.get_camera_view_matrix(
                    sim, env_ptr, self._camera_handles[env_idx])
            ).to(self._device))
        return torch.stack(view_mat)

    def get_adjusted_projection_matrix(self, gym, sim, env_ptr, i
                                       ) -> torch.Tensor:
        proj_mat = self.get_projection_matrix(gym, sim, env_ptr, i)
        fu = 2 / proj_mat[0, 0]
        fv = 2 / proj_mat[1, 1]
        return torch.Tensor([[fu, 0., 0.],
                             [0., fv, 0.],
                             [0., 0., 1.]]).to(self._device)

    def _acquire_local_transform_tensors(self, num_envs: int) -> None:
        self._local_pos = torch.Tensor(
            [[self.camera_transform.p.x,
              self.camera_transform.p.y,
              self.camera_transform.p.z]]).repeat(num_envs, 1).to(self._device)
        self._local_quat = torch.Tensor(
            [[self.camera_transform.r.x,
              self.camera_transform.r.y,
              self.camera_transform.r.z,
              self.camera_transform.r.w]]).repeat(num_envs, 1).to(self._device)

    def get_rigid_body_count(self, gym) -> int:
        if hasattr(self, "_camera_asset"):
            return gym.get_asset_rigid_body_count(self._camera_asset)
        else:
            return 0

    def get_rigid_shape_count(self, gym) -> int:
        if hasattr(self, "_camera_asset"):
            return gym.get_asset_rigid_shape_count(self._camera_asset)
        else:
            return 0

    def create_camera(self, gym, sim, num_envs: int, env_ptrs: List[int],
                      robot_handles: List[int]) -> None:
        # Verify body handle and acquire local transform for attached cameras
        if self._attach_to_body:
            robot_body_names = gym.get_actor_rigid_body_names(
                env_ptrs[0], robot_handles[0])
            assert self._attach_to_body in robot_body_names, \
                f"Expected attach_to_body to be in robot bodies " \
                f"{robot_body_names}, but found {self._attach_to_body}."

        for env_id in range(num_envs):
            # Create Isaac Gym Camera instance
            camera_handle = gym.create_camera_sensor(
                env_ptrs[env_id], self.camera_props)

            # Attach camera to a rigid body
            if self._attach_to_body:
                body_handle = gym.find_actor_rigid_body_handle(
                    env_ptrs[env_id], robot_handles[env_id],
                    self._attach_to_body)
                gym.attach_camera_to_body(
                    camera_handle, env_ptrs[env_id], body_handle,
                    self.camera_transform, gymapi.FOLLOW_TRANSFORM)
            # Set absolute transform of a static camera
            else:
                gym.set_camera_transform(
                    camera_handle, env_ptrs[env_id], self.camera_transform)

            self._camera_handles.append(camera_handle)

            if self._use_camera_tensors:
                self._acquire_camera_tensors(gym, sim, env_ptrs[env_id],
                                             camera_handle)

    def _create_camera_asset(self, gym, sim):
        camera_asset_file = f'{self._model}/{self._model}.urdf'
        assert os.path.isfile(os.path.join(
            self._cameras_asset_root, camera_asset_file)), \
            f"Using add_camera_asset=True, but the camera " \
            f"description " \
            f"{os.path.join(self._cameras_asset_root, camera_asset_file)}" \
            f" was not found."
        camera_asset_options = gymapi.AssetOptions()
        camera_asset_options.fix_base_link = True
        self._camera_asset = gym.load_asset(
            sim, self._cameras_asset_root, camera_asset_file,
            camera_asset_options)
        self._camera_actor_handles = []
        self._actor_ids_sim = []

    def create_camera_actor(self, gym, env_ptr: int, i: int,
                            camera_name: str, actor_count: int,
                            device: torch.device, num_envs) -> None:
        """Adds an actor to the simulation that visualizes the camera."""
        self._device = device
        self._num_envs = num_envs
        if self._add_camera_actor:
            camera_actor_handle = gym.create_actor(
                env_ptr, self._camera_asset, self.camera_transform,
                camera_name, i, 0, 0)
            self._camera_actor_handles.append(camera_actor_handle)
            self._actor_ids_sim.append(actor_count)
            if self._attach_to_body:
                self._acquire_local_transform_tensors(num_envs)

    def _acquire_camera_tensors(self, gym, sim, env_ptr: int,
                                camera_handle: int) -> None:
        if any(t in self._image_type for t in ["rgb", "cpc"]):
            if not hasattr(self, "_camera_tensors_color"):
                self._camera_tensors_color = []
            camera_tensor_color = gym.get_camera_image_gpu_tensor(
                sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR)
            torch_camera_tensor_color = gymtorch.wrap_tensor(
                camera_tensor_color)
            self._camera_tensors_color.append(torch_camera_tensor_color)

        if any(t in self._image_type for t in ["d", "pc"]):
            if not hasattr(self, "_camera_tensors_depth"):
                self._camera_tensors_depth = []
            camera_tensor_depth = gym.get_camera_image_gpu_tensor(
                sim, env_ptr, camera_handle, gymapi.IMAGE_DEPTH)
            torch_camera_tensor_depth = gymtorch.wrap_tensor(
                camera_tensor_depth)
            self._camera_tensors_depth.append(torch_camera_tensor_depth)

        if "seg" in self._image_type:
            if not hasattr(self, "_camera_tensors_segmentation"):
                self._camera_tensors_segmentation = []
            camera_tensor_segmentation = gym.get_camera_image_gpu_tensor(
                sim, env_ptr, camera_handle, gymapi.IMAGE_SEGMENTATION)
            torch_camera_tensor_segmentation = gymtorch.wrap_tensor(
                camera_tensor_segmentation)
            self._camera_tensors_segmentation.append(
                torch_camera_tensor_segmentation)

    def get_image(self, gym, sim, env_ptrs: List[int],
                  env_ids: List[int], robot_handles: List[int],
                  body_pos: torch.Tensor, body_quat: torch.Tensor,
                  root_pos: torch.Tensor, root_quat: torch.Tensor,
                  camera_actor_id_env: int,
                  ) -> torch.Tensor:
        # Move attached camera actors to updated poses
        if self._attach_to_body and self._add_camera_actor:
            self.update_attached_camera_body_pose(
                gym, env_ptrs, robot_handles, body_pos, body_quat, root_pos,
                root_quat, camera_actor_id_env)

        if self._image_type == "rgb":
            return self._get_rgb_image(gym, sim, env_ptrs, env_ids)
        elif self._image_type == "d":
            return self._get_d_image(gym, sim, env_ptrs, env_ids)
        elif self._image_type == "rgbd":
            return self._get_rgbd_image(gym, sim, env_ptrs, env_ids)
        elif self._image_type == "seg":
            return self._get_seg_image(gym, sim, env_ptrs, env_ids)
        elif self._image_type == "pc":
            return self._get_point_cloud(gym, sim, env_ptrs, env_ids)
        elif self._image_type == "cpc":
            return self._get_point_cloud(gym, sim, env_ptrs, env_ids,
                                         colored=True)
        else:
            assert False

    def _get_rgb_image(self, gym, sim, env_ptrs: List[int], env_ids: List[int]
                       ) -> torch.Tensor:
        rgb_image = []
        for env_ptr, env_id in zip(env_ptrs, env_ids):
            color_image = self._get_color_image(gym, sim, env_ptr, env_id)
            rgb_image.append(color_image[..., 0:3])
        return torch.stack(rgb_image, dim=0)

    def _get_d_image(self, gym, sim, env_ptrs: List[int], env_ids: List[int]
                     ) -> torch.Tensor:
        d_image = []
        for env_ptr, env_id in zip(env_ptrs, env_ids):
            depth_image = self._get_depth_image(
                gym, sim, env_ptr, env_id)
            d_image.append(depth_image)
        return torch.stack(d_image, dim=0)

    def _get_rgbd_image(self, gym, sim, env_ptrs: List[int], env_ids: List[int]
                        ) -> torch.Tensor:
        rgbd_image = []
        for env_ptr, env_id in zip(env_ptrs, env_ids):
            color_image = self._get_color_image(gym, sim, env_ptr, env_id) / 255
            depth_image = self._get_depth_image(
                gym, sim, env_ptr, env_id).unsqueeze(-1)
            rgbd_image.append(
                torch.cat([color_image[..., 0:3], depth_image], dim=-1))
        return torch.stack(rgbd_image, dim=0)

    def _get_seg_image(self, gym, sim, env_ptrs: List[int], env_ids: List[int]
                       ) -> torch.Tensor:
        seg_image = []
        for env_ptr, env_id in zip(env_ptrs, env_ids):
            segmentation_image = self._get_segmentation_image(
                gym, sim, env_ptr, env_id)
            seg_image.append(segmentation_image)
        return torch.stack(seg_image, dim=0)

    def _get_color_image(self, gym, sim, env_ptr: int, env_id: int
                         ) -> torch.Tensor:
        if self._use_camera_tensors:
            return self._camera_tensors_color[env_id]
        else:
            return gym.get_camera_image(
                sim, env_ptr, self._camera_handles[env_id], gymapi.IMAGE_COLOR)

    def _get_depth_image(self, gym, sim, env_ptr: int, env_id: int
                         ) -> torch.Tensor:
        if self._use_camera_tensors:
            return self._camera_tensors_depth[env_id]
        else:
            return gym.get_camera_image(
                sim, env_ptr, self._camera_handles[env_id], gymapi.IMAGE_DEPTH)

    def _get_segmentation_image(self, gym, sim, env_ptr: int, env_id: int
                                ) -> torch.Tensor:
        if self._use_camera_tensors:
            return self._camera_tensors_segmentation[env_id]
        else:
            return gym.get_camera_image(
                sim, env_ptr, self._camera_handles[env_id],
                gymapi.IMAGE_SEGMENTATION)

    def _get_point_cloud(self, gym, sim, env_ptrs: List[int],
                         env_ids: List[int], colored: bool = False,
                         downsample: int = None, env_spacing: float = 1.
                         ) -> torch.Tensor:
        depth_image = self._get_d_image(
            gym, sim, env_ptrs, env_ids)
        # adjusted camera projection matrix [3, 3]
        adj_proj_mat = self.get_adjusted_projection_matrix(
            gym, sim, env_ptrs[0], env_ids[0])
        # camera view matrix [len(env_ids), 4, 4]
        view_mat = self.get_view_matrix(gym, sim, env_ptrs, env_ids)

        # convert depth to xyz coordinates via script function
        xyz = depth_image_to_xyz(depth_image, adj_proj_mat, view_mat,
                                 self._device)

        # correct for Isaac Gym view matrices, which are in global instead of
        # local/environment-wise coordinates.
        for env_id in env_ids:
            num_per_row = int(np.sqrt(self._num_envs))
            row = int(np.floor(env_id / num_per_row))
            column = env_id % num_per_row
            xyz[env_id, :, 0] -= column * 2 * env_spacing
            xyz[env_id, :, 1] -= row * 2 * env_spacing

        if downsample:
            assert downsample < self.width * self.height, \
                f"Number of points in downsampled point cloud {downsample} " \
                f"must be smaller than the total number of points " \
                f"{self.height * self.width}."
            if not hasattr(self, "_pc_downsample_idx"):
                self._pc_downsample_idx = torch.randperm(
                    self.width * self.height)[:downsample]
            xyz = xyz[:, self._pc_downsample_idx]

        if colored:
            rgb_image = self._get_rgb_image(
                gym, sim, env_ptrs, env_ids)
            rgb = rgb_image.view(len(env_ids), -1, 3).float().to(self._device)
            if downsample:
                rgb = rgb[:, self._pc_downsample_idx]

            # Debug option to verify the transformation to global coordinates is
            # working. Can be seen for example in the dexterity docs.
            add_coordinate_system = False
            if add_coordinate_system:
                for axis_id in range(3):
                    axis_xyz = torch.zeros((len(env_ids), 200, 3)).to(
                        self._device)
                    axis_xyz[..., axis_id] = torch.linspace(-1, 1, 200).to(
                        self._device)
                    axis_rgb = torch.zeros((len(env_ids), 200, 3)).to(
                        self._device)
                    axis_rgb[..., axis_id] = 255
                    xyz = torch.cat([xyz, axis_xyz], dim=1)
                    rgb = torch.cat([rgb, axis_rgb], dim=1)

            cpc = torch.cat([xyz, rgb], dim=-1)
            return cpc
        return xyz

    def update_attached_camera_body_pose(
            self, gym, env_ptrs, robot_handles, body_pos: torch.Tensor,
            body_quat: torch.Tensor, root_pos: torch.Tensor,
            root_quat: torch.Tensor, camera_actor_id_env: int):
        parent_body_id = gym.find_actor_rigid_body_index(
            env_ptrs[0], robot_handles[0], self._attach_to_body,
            gymapi.DOMAIN_ENV)
        parent_body_pos = body_pos[:, parent_body_id, 0:3]
        parent_body_quat = body_quat[:, parent_body_id, 0:4]

        root_pos[:, camera_actor_id_env] = parent_body_pos + quat_apply(
            parent_body_quat, self._local_pos)
        root_quat[:, camera_actor_id_env] = quat_mul(
            parent_body_quat, self._local_quat)

    @property
    def actor_ids_sim(self) -> torch.Tensor:
        """Returns tensor os sim indices of camera actors."""
        if isinstance(self._actor_ids_sim, List):
            self._actor_ids_sim = torch.Tensor(
                self._actor_ids_sim).to(self._device, torch.int32)
        return self._actor_ids_sim


class DexterityBaseCameras:
    def parse_camera_spec(self, cfg) -> None:
        self._use_camera_tensors = True

        # Create cameras
        self._camera_dict = {}
        if 'cameras' in self.cfg_env.keys():
            for camera_name, camera_cfg in self.cfg_env.cameras.items():
                if camera_name in cfg['env']['observations']:
                    self._camera_dict[camera_name] = DexterityCamera(
                        **camera_cfg)

    @property
    def camera_count(self) -> int:
        return len(self._camera_dict.keys())

    @property
    def camera_rigid_body_count(self) -> int:
        rigid_body_count = 0
        for camera_name, dexterity_camera in self._camera_dict.items():
            if dexterity_camera._add_camera_actor:
                if not hasattr(dexterity_camera, "_camera_asset"):
                    dexterity_camera._create_camera_asset(self.gym, self.sim)
            rigid_body_count += dexterity_camera.get_rigid_body_count(self.gym)
        return rigid_body_count

    @property
    def camera_rigid_shape_count(self) -> int:
        rigid_shape_count = 0
        for camera_name, dexterity_camera in self._camera_dict.items():
            if dexterity_camera._add_camera_actor:
                if not hasattr(dexterity_camera, "_camera_asset"):
                    dexterity_camera._create_camera_asset(self.gym, self.sim)
            rigid_shape_count += dexterity_camera.get_rigid_shape_count(
                self.gym)
        return rigid_shape_count

    def create_camera_actors(self, env_prt: int, i: int, actor_count: int,
                             device: torch.device, num_envs) -> None:
        for camera_name, dexterity_camera in self._camera_dict.items():
            dexterity_camera.create_camera_actor(
                self.gym, env_prt, i, camera_name, actor_count, device,
                num_envs)
            actor_count += 1

    def get_images(self) -> Dict[str, torch.Tensor]:
        """Retrieve images from all camera sensors and in all environments."""
        # Create cameras in all envs if that has not already been done
        for camera_name, dexterity_camera in self._camera_dict.items():
            if len(dexterity_camera._camera_handles) == 0:
                dexterity_camera.create_camera(
                    self.gym, self.sim, self.num_envs, self.env_ptrs,
                    self.robot_handles)

        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        image_dict = {}
        if self._use_camera_tensors:
            self.gym.start_access_image_tensors(self.sim)

            for camera_name, dexterity_camera in self._camera_dict.items():
                image_dict[camera_name] = dexterity_camera.get_image(
                    self.gym, self.sim, self.env_ptrs,
                    list(range(self.num_envs)), self.robot_handles,
                    self.body_pos, self.body_quat, self.root_pos,
                    self.root_quat,
                    getattr(self, camera_name + "_actor_id_env"))

                self.gym.set_actor_root_state_tensor_indexed(
                    self.sim,
                    gymtorch.unwrap_tensor(self.root_state),
                    gymtorch.unwrap_tensor(dexterity_camera.actor_ids_sim),
                    len(dexterity_camera.actor_ids_sim))

        if self._use_camera_tensors:
            self.gym.end_access_image_tensors(self.sim)
        return image_dict

    def save_videos(self, max_recording_depth: float = 3.0) -> None:
        """Saves videos of the Isaac Gym cameras to file.

        Args:
            max_recording_depth: (float, optional) Depth in meters that will be
            mapped to the rgb value (0, 0, 0) in the video.
        """
        if "image" not in self.obs_dict.keys():
            return
        image_dict = self.obs_dict["image"]

        if not hasattr(self, "_videos"):
            self._init_video_recordings()

        # Iterate through all cameras
        for camera_name in image_dict.keys():
            if camera_name not in self._videos.keys():
                self._videos[camera_name] = self._create_video_writers(
                    camera_name, env_ids=list(range(self.num_envs)))

            # Convert point clouds to images in a single step beforehand, since
            # PyTorch3D can operate on batches of point clouds.
            if 'pc' in self._camera_dict[camera_name].camera_type:
                xyz = image_dict[camera_name][..., 0:3]
                if self._camera_dict[camera_name].camera_type == 'cpc':
                    rgb = image_dict[camera_name][..., 3:6] / 255
                else:
                    rgb = torch.zeros_like(xyz)

                pytorch3d_pc = Pointclouds(points=xyz, features=rgb)
                pc_np_image = (255 * self._camera_dict[camera_name].pc_renderer(
                    pytorch3d_pc, gamma=(1e-4,) * self.num_envs,
                    bg_col=torch.tensor([1.0, 1.0, 1.0]).to(self.device)
                ).cpu().numpy()).astype(np.uint8)[..., ::-1]

            for env_id in range(self.num_envs):
                if self._camera_dict[camera_name].camera_type == 'rgb':
                    np_image = image_dict[camera_name][env_id].cpu().numpy()[
                               ..., ::-1]
                elif self._camera_dict[camera_name].camera_type == 'd':
                    np_image = image_dict[camera_name][env_id].unsqueeze(
                        -1).cpu().numpy()
                    np_image = -np.repeat(np_image, 3, axis=-1)
                    np_image = max_recording_depth - np_image.clip(
                        min=0, max=max_recording_depth)
                    np_image = (np_image * 255 / max_recording_depth).astype(
                        np.uint8)
                elif self._camera_dict[camera_name].camera_type == 'rgbd':
                    rgb_image = (255 * image_dict[camera_name][env_id].cpu(
                    ).numpy()[..., 0:3][..., ::-1]).astype(np.uint8)

                    depth_image = image_dict[camera_name][env_id][
                        ..., 3].unsqueeze(-1).cpu().numpy()
                    depth_image = -np.repeat(depth_image, 3, axis=-1)
                    depth_image = max_recording_depth - depth_image.clip(
                        min=0, max=max_recording_depth)
                    depth_image = (depth_image * 255 / max_recording_depth
                                   ).astype(np.uint8)
                    np_image = np.concatenate([rgb_image, depth_image], axis=1)

                elif self._camera_dict[camera_name].camera_type == 'seg':
                    seg_image = image_dict[camera_name][env_id].unsqueeze(
                        -1).cpu().numpy()
                    seg_image = np.repeat(seg_image, 3, axis=-1)
                    np_image = np.zeros_like(seg_image).astype(np.uint8)

                    # Get mapping from segmentation ids to rgb colors from
                    # matplotlib color map
                    cmap = plt.get_cmap('jet')
                    norm = colors.Normalize(vmin=0, vmax=np.max(seg_image))
                    scalar_map = cmx.ScalarMappable(norm=norm, cmap=cmap)

                    for seg_id in range(np.max(seg_image) + 1):
                        color = np.repeat(np.repeat(np.expand_dims(
                            255 * np.array(scalar_map.to_rgba(
                                seg_id)[0:3]), axis=(0, 1)),
                            seg_image.shape[0], axis=0), seg_image.shape[1],
                            axis=1).astype(np.uint8)
                        np_image = np.where(seg_image == seg_id,
                                            color, np_image)

                elif 'pc' in self._camera_dict[camera_name].camera_type:
                    np_image = pc_np_image[env_id]
                else:
                    assert False, \
                        f"Cannot write videos to file for image type " \
                        f"{self._camera_dict[camera_name].camera_type} yet."
                self._videos[camera_name][env_id].write(np_image)

        done_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(done_env_ids) > 0:
            for done_env_id in done_env_ids:
                self._episodes[done_env_id] += 1
                for camera_name in image_dict.keys():
                    self._videos[camera_name][done_env_id].release()
                    self._videos[camera_name][done_env_id] = \
                        self._create_video_writers(
                            camera_name, env_ids=done_env_id)

    def _init_video_recordings(self) -> None:
        experiment_dir = os.path.join(
            'runs', self.cfg['full_experiment_name'])
        self.videos_dir = os.path.join(experiment_dir, 'videos')
        if not os.path.exists(self.videos_dir):
            os.makedirs(self.videos_dir)
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self._videos = {}
        self._episodes = [0 for _ in range(self.num_envs)]

    def _create_video_writers(
            self, camera_name: str, env_ids: List[int]
    ) -> Union[cv2.VideoWriter, List[cv2.VideoWriter]]:

        # Point cloud image is created by renderer in 1920 x 1080
        if 'pc' in self._camera_dict[camera_name].camera_type:
            width = 1920
            height = 1080
        # RGBD has rgb and depth image next to each other
        elif self._camera_dict[camera_name].camera_type == 'rgbd':
            width = 2 * self._camera_dict[camera_name].width
            height = self._camera_dict[camera_name].height
        else:
            width = self._camera_dict[camera_name].width
            height = self._camera_dict[camera_name].height

        env_ids_list = [env_ids] if not isinstance(env_ids, List) else env_ids
        video_writers = [cv2.VideoWriter(
            os.path.join(
                self.videos_dir,
                f"{camera_name}_env_{env_id}_episode_"
                f"{self._episodes[env_id]}.mp4"),
            self.fourcc,
            1 / self.cfg['sim']['dt'],
            (width, height)) for env_id in env_ids_list]
        if not isinstance(env_ids, List):
            return video_writers[0]
        else:
            return video_writers
