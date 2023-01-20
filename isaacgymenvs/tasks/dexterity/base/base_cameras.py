import cv2
import hydra
from isaacgym import gymapi, gymtorch
import numpy as np
import os
import torch
from typing import *

IMAGE_TYPES = ['d', 'rgb', 'rgbd', 'seg']


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
        use_camera_tensors: (bool, True) Whether to use GPU tensor access when
            retrieving the camera images.
        add_camera_asset: (bool, True) Whether to add an asset of the camera
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
            use_camera_tensors: bool = True,
            add_camera_actor: bool = True
    ) -> None:
        self._pos = pos
        self._quat = quat
        self._model = model
        self._use_camera_tensors = use_camera_tensors
        self._add_camera_actor = add_camera_actor

        if model is not None:
            # Check whether camera model exists
            self._cameras_asset_root = os.path.join(
                os.path.dirname(__file__),
                '../../../../assets/dexterity/cameras')
            available_camera_models = [
                f.name for f in os.scandir(self._cameras_asset_root)
                if f.is_dir()]
            assert model in available_camera_models, \
                f"Camera model should be one of {available_camera_models}," \
                f" but unknown model '{model}' was found."

            # Retrieve and set camera info
            camera_info_file = f'{model}/camera_info.yaml'
            camera_info = hydra.compose(
                config_name=os.path.join(
                    '../../assets/dexterity/cameras', camera_info_file))
            camera_info = camera_info['']['']['']['']['']['']['assets'][
                'dexterity']['cameras'][model]  # strip superfluous nesting
            self._fovx = camera_info.fovx
            self._resolution = camera_info.resolution
            self._image_type = camera_info.image_type

        else:
            assert None not in [fovx, resolution, image_type], \
                "fovx, resolution, and image_type must be specified if no " \
                "camera model is given."
            self._fovx = fovx
            self._resolution = resolution
            assert image_type in IMAGE_TYPES, \
                f"Image type should be one of {IMAGE_TYPES}, but unknown type " \
                f"'{image_type}' was found."
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
        pos = gymapi.Vec3(*self._pos)
        quat = gymapi.Quat(*self._quat)
        return gymapi.Transform(p=pos, r=quat)

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

    def create_camera(self, gym, sim, num_envs: int, env_ptrs: List[int]
                      ) -> None:
        self._camera_handles = []
        if self._use_camera_tensors:
            if "rgb" in self._image_type:
                self._camera_tensors_color = []
            if "d" in self._image_type:
                self._camera_tensors_depth = []
            if "seg" in self._image_type:
                self._camera_tensors_segmentation = []

        for env_id in range(num_envs):
            camera_handle = gym.create_camera_sensor(
                env_ptrs[env_id], self.camera_props)
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
        print("creating camera asset ...")
        camera_asset_options = gymapi.AssetOptions()
        camera_asset_options.fix_base_link = True
        self._camera_asset = gym.load_asset(
            sim, self._cameras_asset_root, camera_asset_file,
            camera_asset_options)
        self._camera_actor_handles = []

    def create_camera_actor(self, gym, sim, env_ptr: int, i: int) -> None:
        """Adds an actor to the simulation that visualizes the camera."""
        if self._add_camera_actor:
            camera_actor_handle = gym.create_actor(
                env_ptr, self._camera_asset, self.camera_transform,
                self._model, i, 0, 0)
            self._camera_actor_handles.append(camera_actor_handle)

    def _acquire_camera_tensors(self, gym, sim, env_ptr: int,
                                camera_handle: int) -> None:
        if "rgb" in self._image_type:
            camera_tensor_color = gym.get_camera_image_gpu_tensor(
                sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR)
            torch_camera_tensor_color = gymtorch.wrap_tensor(
                camera_tensor_color)
            self._camera_tensors_color.append(torch_camera_tensor_color)

        if "d" in self._image_type:
            camera_tensor_depth = gym.get_camera_image_gpu_tensor(
                sim, env_ptr, camera_handle, gymapi.IMAGE_DEPTH)
            torch_camera_tensor_depth = gymtorch.wrap_tensor(
                camera_tensor_depth)
            self._camera_tensors_depth.append(torch_camera_tensor_depth)

        if "seg" in self._image_type:
            raise NotImplementedError

    def get_image(self, gym, sim, num_envs: int, env_ptrs: List[int],
                  env_ids: List[int]) -> torch.Tensor:
        # Create cameras in all envs if that has not already been done
        if not hasattr(self, "_camera_handles"):
            self.create_camera(gym, sim, num_envs, env_ptrs)

        if self._image_type == "rgb":
            return self._get_rgb_image(gym, sim, env_ptrs, env_ids)
        elif self._image_type == "d":
            return self._get_d_image(gym, sim, env_ptrs, env_ids)
        elif self._image_type == "rgbd":
            return self._get_rgbd_image(gym, sim, env_ptrs, env_ids)
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
                gym, sim, env_ptr, env_id).unsqueeze(-1)
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

    def create_camera_actors(self, env_prt: int, i: int) -> None:
        for camera_name, dexterity_camera in self._camera_dict.items():
            dexterity_camera.create_camera_actor(self.gym, self.sim, env_prt, i)

    def get_images(self) -> Dict[str, torch.Tensor]:
        self.gym.render_all_camera_sensors(self.sim)
        image_dict = {}
        if self._use_camera_tensors:
            self.gym.start_access_image_tensors(self.sim)

            for camera_name, dexterity_camera in self._camera_dict.items():
                image_dict[camera_name] = dexterity_camera.get_image(
                    self.gym, self.sim, self.num_envs, self.env_ptrs,
                    list(range(self.num_envs)))

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

        for camera_name in image_dict.keys():
            if camera_name not in self._videos.keys():
                self._videos[camera_name] = self._create_video_writers(
                    camera_name, env_ids=list(range(self.num_envs)))

            for env_id in range(self.num_envs):
                if self._camera_dict[camera_name].camera_type == 'rgb':
                    np_image = image_dict[camera_name][env_id].cpu().numpy()[
                               ..., ::-1]
                elif self._camera_dict[camera_name].camera_type == 'd':
                    np_image = image_dict[camera_name][env_id].cpu().numpy()
                    np_image = -np.repeat(np_image, 3, axis=-1)
                    np_image = max_recording_depth - np_image.clip(
                        min=0, max=max_recording_depth)
                    np_image = (np_image * 255 / max_recording_depth).astype(
                        np.uint8)
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

        width = self._camera_dict[camera_name].width
        height = self._camera_dict[camera_name].height

        env_ids = [env_ids] if not isinstance(env_ids, List) else env_ids
        video_writers = [cv2.VideoWriter(
            os.path.join(
                self.videos_dir,
                f"{camera_name}_env_{env_id}_episode_"
                f"{self._episodes[env_id]}.mp4"),
            self.fourcc,
            1 / self.cfg['sim']['dt'],
            (width, height)) for env_id in env_ids]
        if len(env_ids) == 1:
            return video_writers[0]
        else:
            return video_writers

