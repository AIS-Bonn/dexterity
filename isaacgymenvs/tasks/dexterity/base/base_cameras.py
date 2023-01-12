import cv2
from isaacgym import gymapi, gymtorch
import os
import torch
from typing import *


class DexterityCamera:
    """Wrapper class for IsaacGym camera sensors."""
    def __init__(
            self,
            gym,
            sim,
            env_ptrs,
            fovx: int,
            pos: Tuple[float, float, float],
            quat: Tuple[float, float, float, float],
            resolution: Tuple[int, int],
            type: str,
            use_camera_tensors: bool = True
    ) -> None:
        self.gym = gym
        self.sim = sim
        self.env_ptrs = env_ptrs
        self.num_envs = len(env_ptrs)
        self._fovx = fovx
        self._pos = pos
        self._quat = quat
        self._resolution = resolution
        self._type = type
        assert self._type in ["rgb", "rgbd"]
        self._use_camera_tensors = use_camera_tensors

        self._create_camera()

    @property
    def camera_type(self) -> str:
        return self._type

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

    def _create_camera(self) -> None:
        self._camera_handles = []
        if self._use_camera_tensors:
            self._camera_tensors_color = []
            if "d" in self._type:
                self._camera_tensors_depth = []

        for env_id in range(self.num_envs):
            camera_handle = self.gym.create_camera_sensor(
                self.env_ptrs[env_id], self.camera_props)
            self.gym.set_camera_transform(
                camera_handle, self.env_ptrs[env_id], self.camera_transform)
            self._camera_handles.append(camera_handle)

            if self._use_camera_tensors:
                self._acquire_camera_tensors(env_id, camera_handle)

    def _acquire_camera_tensors(self, env_id: int, camera_handle: int) -> None:
        camera_tensor_color = self.gym.get_camera_image_gpu_tensor(
            self.sim, self.env_ptrs[env_id], camera_handle, gymapi.IMAGE_COLOR)
        torch_camera_tensor_color = gymtorch.wrap_tensor(camera_tensor_color)
        self._camera_tensors_color.append(torch_camera_tensor_color)

        if "d" in self._type:
            camera_tensor_depth = self.gym.get_camera_image_gpu_tensor(
                self.sim, self.env_ptrs[env_id], camera_handle,
                gymapi.IMAGE_DEPTH)
            torch_camera_tensor_depth = gymtorch.wrap_tensor(
                camera_tensor_depth)
            self._camera_tensors_depth.append(torch_camera_tensor_depth)

    def get_image(self, env_ids: List[int]) -> torch.Tensor:
        if self._type == "rgb":
            return self._get_rgb_image(env_ids)
        elif self._type == "rgbd":
            return self._get_rgbd_image(env_ids)
        else:
            assert False

    def _get_rgb_image(self, env_ids: List[int]) -> torch.Tensor:
        rgb_image = []
        for env_id in env_ids:
            color_image = self._get_color_image(env_id)
            rgb_image.append(color_image[..., 0:3])
        return torch.stack(rgb_image, dim=0)

    def _get_rgbd_image(self, env_ids: List[int]) -> torch.Tensor:
        rgbd_image = []
        for env_id in env_ids:
            color_image = self._get_color_image(env_id) / 255
            depth_image = self._get_depth_image(env_id).unsqueeze(-1)
            rgbd_image.append(
                torch.cat([color_image[..., 0:3], depth_image], dim=-1))
        return torch.stack(rgbd_image, dim=0)

    def _get_color_image(self, env_id: int) -> torch.Tensor:
        if self._use_camera_tensors:
            return self._camera_tensors_color[env_id]
        else:
            return self.gym.get_camera_image(
                self.sim, self.env_ptrs[env_id], self._camera_handles[env_id],
                gymapi.IMAGE_COLOR)

    def _get_depth_image(self, env_id: int) -> torch.Tensor:
        if self._use_camera_tensors:
            return self._camera_tensors_depth[env_id]
        else:
            return self.gym.get_camera_image(
                self.sim, self.env_ptrs[env_id], self._camera_handles[env_id],
                gymapi.IMAGE_DEPTH)


class DexterityBaseCameras:
    def parse_camera_spec(self) -> None:
        # Create cameras
        self._camera_dict = {}
        if 'cameras' in self.cfg_task.keys():
            for camera_name, camera_cfg in self.cfg_task.cameras.items():
                if camera_name in self.cfg_task.env.observations:
                    self._camera_dict[camera_name] = DexterityCamera(
                        self.gym, self.sim, self.env_ptrs, **camera_cfg)

        self._use_camera_tensors = True

        # Initialize recordings if necessary
        if self.cfg_base.debug.save_videos:
            experiment_dir = os.path.join('runs', self.cfg['full_experiment_name'])
            self.videos_dir = os.path.join(experiment_dir, 'videos')
            if not os.path.exists(self.videos_dir):
                os.mkdir(self.videos_dir)
            self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self._videos = {}
            self._episodes = [0 for _ in range(self.num_envs)]

    def get_images(self) -> Dict[str, torch.Tensor]:
        self.gym.render_all_camera_sensors(self.sim)
        image_dict = {}
        if self._use_camera_tensors:
            self.gym.start_access_image_tensors(self.sim)

            for camera_name, dexterity_camera in self._camera_dict.items():
                image_dict[camera_name] = dexterity_camera.get_image(
                    list(range(self.num_envs)))

        if self._use_camera_tensors:
            self.gym.end_access_image_tensors(self.sim)
        return image_dict

    def save_videos(self) -> None:
        if "image" not in self.obs_dict.keys():
            return
        image_dict = self.obs_dict["image"]

        for camera_name in image_dict.keys():
            if camera_name not in self._videos.keys():
                width = self._camera_dict[camera_name].width
                height = self._camera_dict[camera_name].height
                self._videos[camera_name] = [cv2.VideoWriter(
                    os.path.join(
                        self.videos_dir,
                        f"{camera_name}_env_{env_id}_episode_{self._episodes[env_id]}.mp4"),
                    self.fourcc,
                    1 / self.cfg['sim']['dt'],
                    (width, height)) for env_id in range(self.num_envs)]

            for env_id in range(self.num_envs):
                if self._camera_dict[camera_name].camera_type == 'rgb':
                    np_image = image_dict[camera_name][env_id].cpu().numpy()[..., ::-1]
                else:
                    assert False, "Can only write videos for RGB cameras."
                self._videos[camera_name][env_id].write(np_image)

        done_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(done_env_ids) > 0:
            for done_env_id in done_env_ids:
                self._episodes[done_env_id] += 1
                for camera_name in image_dict.keys():
                    width = self._camera_dict[camera_name].width
                    height = self._camera_dict[camera_name].height
                    self._videos[camera_name][done_env_id].release()
                    self._videos[camera_name][done_env_id] = cv2.VideoWriter(
                        os.path.join(
                            self.videos_dir,
                            f"{camera_name}_env_{done_env_id}_episode_{self._episodes[done_env_id]}.mp4"),
                        self.fourcc,
                        1 / self.cfg['sim']['dt'],
                        (width, height))
