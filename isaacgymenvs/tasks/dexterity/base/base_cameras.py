import cv2
import hydra
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import os
from typing import *


@torch.jit.script
def depth_image_to_xyz(depth_image, proj_mat, view_mat, device: torch.device):
    batch_size, width, height = depth_image.shape
    sparse_depth = depth_image.to(device).to_sparse()
    indices = sparse_depth.indices()
    values = sparse_depth.values()
    xy_depth = torch.cat([indices.T[:, 1:].flip(1), values[..., None]], dim=-1)

    center_u = height / 2
    center_v = width / 2

    xy_depth[:, 0] = -(xy_depth[:, 0] - center_u) / height
    xy_depth[:, 1] = (xy_depth[:, 1] - center_v) / width
    xy_depth[:, 0] *= xy_depth[:, 2]
    xy_depth[:, 1] *= xy_depth[:, 2]

    x2 = xy_depth @ proj_mat
    x2_hom = torch.cat([x2, torch.ones_like(x2[:, 0:1])], dim=1).view(
        batch_size, -1, 4)
    xyz = torch.bmm(x2_hom, view_mat.inverse())[..., 0:3]
    return xyz


def xyz_world_to_camera(xyz_world, view_mat):
    num_envs, num_samples, _ = xyz_world.shape

    # Project from world to camera frame.
    xyz_hom = torch.cat([xyz_world, torch.ones_like(xyz_world[:, :, 0:1])],
                        dim=-1).view(num_envs * num_samples, 4, 1)
    xyz_camera = torch.bmm(view_mat, xyz_hom)[:, 0:3, 0].view(num_envs, num_samples, 3)
    return xyz_camera


def xyz_camera_to_world(xyz_camera, inv_view_mat):
    num_envs, num_samples, _ = xyz_camera.shape

    # Project from camera to world frame.
    xyz_hom = torch.cat([xyz_camera, torch.ones_like(xyz_camera[:, :, 0:1])],
                        dim=-1).view(num_envs * num_samples, 1, 4)
    xyz_world = torch.bmm(xyz_hom, inv_view_mat)[:, 0, 0:3].view(num_envs, num_samples, 3)
    return xyz_world


def xyz_to_image(xyz, proj_mat, view_mat, width, height):
    batch_size, num_samples, _ = xyz.shape

    # Project from world to camera frame.
    xyz_hom = torch.cat([xyz, torch.ones_like(xyz[:, :, 0:1])], dim=-1).view(batch_size * num_samples, 4, 1)
    xyz_camera = torch.bmm(view_mat, xyz_hom)[:, 0:3, :]

    # Multiply with projection matrix.
    xyz_camera = torch.bmm(proj_mat, xyz_camera)[..., 0].view(batch_size, num_samples, 3)

    # Divide x and y dimensions by -z (z or perspective divide).
    xyz_camera[..., 0] /= -xyz_camera[..., 2]
    xyz_camera[..., 1] /= -xyz_camera[..., 2]

    # Map to image plane.
    center_u = width / 2
    center_v = height / 2
    v = center_v - (xyz_camera[..., 1] * height)
    u = (xyz_camera[..., 0] * width) + center_u
    image_plane = torch.stack([v, u], dim=-1).to(torch.long)
    return image_plane


def image_plane_to_bounding_box(image_plane, format: str = 'tlbr'):
    x_min = torch.min(image_plane[..., 1], dim=1)[0]
    y_min = torch.min(image_plane[..., 0], dim=1)[0]
    x_max = torch.max(image_plane[..., 1], dim=1)[0]
    y_max = torch.max(image_plane[..., 0], dim=1)[0]
    if format == 'xywh':
        bounding_box = torch.stack([x_min, y_min, x_max - x_min, y_max - y_min], dim=-1)
    elif format == 'tlbr':
        bounding_box = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
    else:
        assert False
    return bounding_box


def draw_square(image, point, size: int, color: Tuple[int, int, int]):
    for i in range(3):
        image[point[0] - size:point[0] + size, point[1] - size:point[1] + size, i] = color[i]
    return image


def draw_bounding_box(image, bounding_box, width: int, color: Tuple[int, int, int]):
    x, y, w, h = bounding_box[0], bounding_box[1], bounding_box[2], bounding_box[3]
    for i in range(3):
        image[y - width:y + h + width, x - width:x + width, i] = color[i]  # left
        image[y - width:y + h + width, x + w - width:x + w + width, i] = color[i]  # right
        image[y - width:y + width, x:x + w, i] = color[i]  # top
        image[y + h - width:y + h + width, x:x + w, i] = color[i]  # bottom
    return image


class DexterityVideoRecordingProperties:
    """Implements properties of video recordings of the camera observations."""
    POINT_CLOUD_RENDER_SIZE = (1920, 1080)
    POINT_CLOUD_RENDERER_EYE = (0.75, -0.75, 0.5)
    POINT_CLOUD_RENDERER_LOOKAT = (0., 0., 0.2)
    _point_cloud_renderer = None
    SEG_ID_CMAP = plt.get_cmap('jet')

    @property
    def point_cloud_render_width(self) -> int:
        return self.POINT_CLOUD_RENDER_SIZE[0]

    @property
    def point_cloud_render_height(self) -> int:
        return self.POINT_CLOUD_RENDER_SIZE[1]

    @property
    def point_cloud_renderer(self):
        """Returns PyTorch3D renderer that converts point-clouds to images."""
        if self._point_cloud_renderer is None:
            from pytorch3d.renderer import (
                look_at_view_transform, FoVOrthographicCameras,
                PointsRasterizationSettings, PointsRasterizer,
                PulsarPointsRenderer)
            R, T = look_at_view_transform(
                eye=(self.POINT_CLOUD_RENDERER_EYE,), up=((0, 0, 1),),
                at=(self.POINT_CLOUD_RENDERER_LOOKAT,))

            cameras = FoVOrthographicCameras(
                device=self.device, R=R, T=T, znear=0.01)
            raster_settings = PointsRasterizationSettings(
                image_size=(self.point_cloud_render_height,
                            self.point_cloud_render_width),
                radius=0.003, points_per_pixel=1)
            self._point_cloud_renderer = PulsarPointsRenderer(
                rasterizer=PointsRasterizer(
                    cameras=cameras, raster_settings=raster_settings),
                n_channels=3).to(self.device)
        return self._point_cloud_renderer

    def scalar_color_mapping(self, vmin: float, vmax: float):
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        return cmx.ScalarMappable(norm=norm, cmap=self.SEG_ID_CMAP)

    def segmentation_to_rgb(self, seg: torch.Tensor) -> torch.Tensor:
        seg = seg.unsqueeze(-1).repeat((1,) * len(seg.shape) + (3,))
        scalar_map = self.scalar_color_mapping(
            torch.min(seg).float(), torch.max(seg).float())
        rgb = torch.zeros_like(seg)

        for seg_id in range(int(torch.max(seg)) + 1):
            rgb_id = torch.zeros_like(seg)
            color_map = (torch.Tensor(scalar_map.to_rgba(seg_id)[0:3]) * 255).to(
                torch.uint8)
            rgb_id[..., 0:3] = color_map
            rgb = torch.where(seg == seg_id, rgb_id, rgb)
        return rgb.to(torch.uint8)


class DexterityCameraSensorProperties:
    """Implements properties of the camera sensors, such as image-types, poses,
    and camera intrinsics."""

    ALLOWED_IMAGE_TYPES = ['d', 'rgb', 'rgbd', 'seg', 'rgb_seg', 'pc', 'pc_rgb',
                           'pc_seg']
    CAMERA_ASSET_ROOT = os.path.join(
        os.path.dirname(__file__), '../../../../assets/dexterity/cameras')

    def __init__(
            self,
            pos: Tuple[float, float, float],
            quat: Tuple[float, float, float, float],
            model: Optional[str] = None,
            fovx: Optional[int] = None,
            resolution: Optional[Tuple[int, int]] = None,
            image_type: Optional[str] = None,
            attach_to_body: Optional[str] = None,
            use_camera_tensors: Optional[bool] = True,
    ) -> None:
        self.pos = pos
        self.quat = quat
        self.attach_to_body = attach_to_body
        self.use_camera_tensors = use_camera_tensors

        self._model = None
        if model is not None:
            self.configure_properties_from_model(model)
        else:
            self.configure_properties_from_parameters(
                fovx, resolution, image_type)

        self._camera_handles = []
        self._camera_tensors_color = []
        self._camera_tensors_depth = []
        self._camera_tensors_segmentation = []

    def configure_properties_from_model(self, model: str) -> None:
        self.model = model
        camera_info_file = f'{self.model}/camera_info.yaml'
        camera_info = hydra.compose(
            config_name=os.path.join(
                '../../assets/dexterity/cameras', camera_info_file))
        camera_info = camera_info['']['']['']['']['']['']['assets'][
            'dexterity']['cameras'][self.model]
        self.configure_properties_from_parameters(
            camera_info.fovx, camera_info.resolution, camera_info.image_type)

    def configure_properties_from_parameters(
            self, fovx: int, resolution: Tuple[int, int], image_type: str
    ) -> None:
        assert None not in [fovx, resolution, image_type], \
            "fovx, resolution, and image_type must be specified if no " \
            "camera model is given."
        self.fovx = fovx
        self.resolution = resolution
        self.image_type = image_type

    @property
    def pos(self) -> gymapi.Vec3:
        return self._pos

    @pos.setter
    def pos(self, value: Tuple[float, float, float]) -> None:
        self._pos = gymapi.Vec3(*value)

    def get_pos_tensor(self, num_envs: int, device: torch.device) -> torch.Tensor:
        return torch.Tensor([[self.pos.x, self.pos.y, self.pos.z]]).repeat(num_envs, 1).to(device)

    @property
    def quat(self) -> gymapi.Quat:
        return self._quat

    @quat.setter
    def quat(self, value: Tuple[float, float, float, float]) -> None:
        self._quat = gymapi.Quat(*value)

    def get_quat_tensor(self, num_envs: int, device: torch.device) -> torch.Tensor:
        return torch.Tensor([[self.quat.x, self.quat.y, self.quat.z, self.quat.w]]).repeat(num_envs, 1).to(device)

    def compute_view_matrix(self, repeats: int, device: torch.device) -> torch.Tensor:
        """Computes the view matrix for a static camera without the need for an
        Isaac Gym camera sensor."""

        # Build rotation matrix from quaternion.
        from scipy.spatial.transform import Rotation
        rotation = Rotation.from_quat([self.quat.x, self.quat.y, self.quat.z, self.quat.w])
        rotation_matrix = torch.from_numpy(rotation.as_matrix())

        # Fill transformation matrix.
        transformation_matrix = torch.eye(4)
        transformation_matrix[0:3, 0:3] = rotation_matrix
        transformation_matrix[0, 3] = self.pos.x
        transformation_matrix[1, 3] = self.pos.y
        transformation_matrix[2, 3] = self.pos.z

        # Get view matrix as inverse of transformation matrix.
        inv_transformation_matrix = transformation_matrix.inverse()
        view_matrix = torch.eye(4).to(device)
        view_matrix[0, :] = -inv_transformation_matrix[1, :]
        view_matrix[1, :] = inv_transformation_matrix[2, :]
        view_matrix[2, :] = -inv_transformation_matrix[0, :]
        return view_matrix.unsqueeze(0).repeat(repeats, 1, 1)

    def compute_projection_matrix(self, repeats: int, device: torch.device) -> torch.Tensor:
        """Computes the projection matrix of a camera without the need for an
        Isaac Gym camera sensor."""

        projection_matrix = torch.eye(3).to(device)
        fovx_rad = self.fovx * (np.pi / 180)
        aspect = self.width / self.height
        projection_matrix[0, 0] = 0.5 / (math.tan(0.5 * fovx_rad))
        projection_matrix[1, 1] = 0.5 / (math.tan(0.5 * fovx_rad) / aspect)
        return projection_matrix.repeat(repeats, 1, 1)

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        available_camera_models = [
            f.name for f in os.scandir(self.CAMERA_ASSET_ROOT) if f.is_dir()]
        assert value in available_camera_models, \
            f"Camera model should be one of {available_camera_models}," \
            f" but unknown model '{value}' was found."
        self._model = value

    @property
    def fovx(self) -> int:
        return self._fovx

    @fovx.setter
    def fovx(self, value: int) -> None:
        assert 0 < value < 180, \
            f"Horizontal field-of-view (fovx) should be in [0, 180], but " \
            f"found '{value}'."
        self._fovx = value

    @property
    def resolution(self) -> Tuple[int, int]:
        return self._resolution

    @resolution.setter
    def resolution(self, value: Tuple[int, int]) -> None:
        assert len(value) == 2 and all(isinstance(v, int) for v in value), \
            f"Resolution should be a tuple of 2 integer values, but found " \
            f"'{value}'."
        self._resolution = value

    @property
    def width(self) -> int:
        return self._resolution[0]

    @property
    def height(self) -> int:
        return self._resolution[1]

    @property
    def image_type(self) -> str:
        return self._image_type

    @image_type.setter
    def image_type(self, value: str) -> None:
        assert value in self.ALLOWED_IMAGE_TYPES, \
            f"Image type should be one of {self.ALLOWED_IMAGE_TYPES}, but " \
            f"unknown type '{value}' was found."
        self._image_type = value

    @property
    def attach_to_body(self) -> str:
        return self._attach_to_body

    @attach_to_body.setter
    def attach_to_body(self, value: str) -> None:
        self._attach_to_body = value

    @property
    def use_camera_tensors(self) -> bool:
        return self._use_camera_tensors

    @use_camera_tensors.setter
    def use_camera_tensors(self, value: bool) -> None:
        self._use_camera_tensors = value

    @property
    def camera_props(self) -> gymapi.CameraProperties:
        camera_props = gymapi.CameraProperties()
        camera_props.width = self.width
        camera_props.height = self.height
        camera_props.horizontal_fov = self.fovx
        camera_props.enable_tensors = self.use_camera_tensors
        return camera_props

    @property
    def camera_transform(self) -> gymapi.Transform:
        return gymapi.Transform(p=self.pos, r=self.quat)


class DexterityCameraSensor(DexterityCameraSensorProperties):
    """Wraps IsaacGym camera sensors and implements functions, such as creating
    and positioning the cameras, as well as querying all the image-types."""

    def __init__(
            self,
            pos: Tuple[float, float, float],
            quat: Tuple[float, float, float, float],
            model: Optional[str] = None,
            fovx: Optional[int] = None,
            resolution: Optional[Tuple[int, int]] = None,
            image_type: Optional[str] = None,
            attach_to_body: Optional[str] = None,
            use_camera_tensors: Optional[bool] = True,
    ) -> None:
        super().__init__(pos, quat, model, fovx, resolution, image_type,
                         attach_to_body, use_camera_tensors)

    def create_camera(self, env_ptr, robot_handle=None) -> None:
        self.check_body_handle(env_ptr, robot_handle)
        camera_handle = self.gym.create_camera_sensor(
            env_ptr, self.camera_props)
        self.set_camera_transform(env_ptr, camera_handle, robot_handle)
        self._camera_handles.append(camera_handle)
        if self._use_camera_tensors:
            self.acquire_camera_tensors(env_ptr, camera_handle)

    def check_body_handle(self, env_ptr, robot_handle) -> None:
        if self.attach_to_body:
            assert robot_handle is not None
            robot_body_names = self.gym.get_actor_rigid_body_names(
                env_ptr, robot_handle)
            assert self.attach_to_body in robot_body_names, \
                f"Expected attach_to_body to be in robot bodies " \
                f"{robot_body_names}, but found {self.attach_to_body}."

    def set_camera_transform(self, env_ptr, camera_handle, robot_handle
                             ) -> None:
        # Attach camera to a rigid body.
        if self._attach_to_body:
            body_handle = self.gym.find_actor_rigid_body_handle(
                env_ptr, robot_handle,
                self._attach_to_body)
            self.gym.attach_camera_to_body(
                camera_handle, env_ptr, body_handle,
                self.camera_transform, gymapi.FOLLOW_TRANSFORM)
        # Set absolute transform of a static camera.
        else:
            self.gym.set_camera_transform(
                camera_handle, env_ptr, self.camera_transform)

    def acquire_camera_tensors(self, env_ptr: int, camera_handle: int) -> None:
        if "rgb" in self._image_type:
            camera_tensor_color = self.gym.get_camera_image_gpu_tensor(
                self.sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR)
            torch_camera_tensor_color = gymtorch.wrap_tensor(
                camera_tensor_color)
            self._camera_tensors_color.append(torch_camera_tensor_color)

        if any(t in self._image_type for t in ["d", "pc"]):
            camera_tensor_depth = self.gym.get_camera_image_gpu_tensor(
                self.sim, env_ptr, camera_handle, gymapi.IMAGE_DEPTH)
            torch_camera_tensor_depth = gymtorch.wrap_tensor(
                camera_tensor_depth)
            self._camera_tensors_depth.append(torch_camera_tensor_depth)

        if "seg" in self._image_type:
            camera_tensor_segmentation = self.gym.get_camera_image_gpu_tensor(
                self.sim, env_ptr, camera_handle, gymapi.IMAGE_SEGMENTATION)
            torch_camera_tensor_segmentation = gymtorch.wrap_tensor(
                camera_tensor_segmentation)
            self._camera_tensors_segmentation.append(
                torch_camera_tensor_segmentation)

    def get_image(self, env_ptrs: List, env_ids: List[int]) -> torch.Tensor:
        return getattr(self, f"_get_{self.image_type}")(env_ptrs, env_ids)

    def _get_rgb(self, env_ptrs: List, env_ids: List[int]) -> torch.Tensor:
        rgb_image = []
        for env_ptr, env_id in zip(env_ptrs, env_ids):
            color_image = self._get_color_image(env_ptr, env_id)
            rgb_image.append(color_image[..., 0:3])
        return torch.stack(rgb_image, dim=0)

    def _get_d(self, env_ptrs: List, env_ids: List[int]) -> torch.Tensor:
        d_image = []
        for env_ptr, env_id in zip(env_ptrs, env_ids):
            depth_image = self._get_depth_image(env_ptr, env_id)
            d_image.append(depth_image)
        return torch.stack(d_image, dim=0)

    def _get_rgbd(self, env_ptrs: List, env_ids: List[int]) -> torch.Tensor:
        rgbd_image = []
        for env_ptr, env_id in zip(env_ptrs, env_ids):
            color_image = self._get_color_image(env_ptr, env_id) / 255
            depth_image = self._get_depth_image(env_ptr, env_id).unsqueeze(-1)
            rgbd_image.append(
                torch.cat([color_image[..., 0:3], depth_image], dim=-1))
        return torch.stack(rgbd_image, dim=0)

    def _get_seg(self, env_ptrs: List, env_ids: List[int]) -> torch.Tensor:
        seg_image = []
        for env_ptr, env_id in zip(env_ptrs, env_ids):
            segmentation_image = self._get_segmentation_image(env_ptr, env_id)
            seg_image.append(segmentation_image)
        return torch.stack(seg_image, dim=0)

    def _get_rgb_seg(self, env_ptrs: List, env_ids: List[int],
                     keep_idx: Optional[int] = None,
                     remove_idx: Optional[int] = 0,
                     background_color: Tuple[int, int, int] = (255, 255, 255)
                     ) -> torch.Tensor:
        assert None in [keep_idx, remove_idx], \
            "keep_idx and remove_idx cannot both be set."

        rgb_seg_image = []
        for env_ptr, env_id in zip(env_ptrs, env_ids):
            color_image = self._get_color_image(env_ptr, env_id)[..., 0:3]
            segmentation_image = self._get_segmentation_image(
                env_ptr, env_id).unsqueeze(-1).repeat(1, 1, 3)
            background = torch.Tensor([[background_color]]).repeat(
                self.height, self.width, 1).to(color_image.device, torch.uint8)

            if keep_idx is not None:
                segmented_color_image = torch.where(
                    segmentation_image == keep_idx, color_image, background)
            else:
                segmented_color_image = torch.where(
                    segmentation_image != remove_idx, color_image, background)

            rgb_seg_image.append(segmented_color_image)
        return torch.stack(rgb_seg_image, dim=0)

    def _get_pc(self, env_ptrs: List, env_ids: List[int]) -> torch.Tensor:
        return self._get_point_cloud(env_ptrs, env_ids)

    def _get_pc_rgb(self, env_ptrs: List, env_ids: List[int]) -> torch.Tensor:
        return self._get_point_cloud(env_ptrs, env_ids, features='rgb')

    def _get_pc_seg(self, env_ptrs: List, env_ids: List[int]) -> torch.Tensor:
        return self._get_point_cloud(env_ptrs, env_ids, features='seg')

    def _get_color_image(self, env_ptr: int, env_id: int) -> torch.Tensor:
        if self.use_camera_tensors:
            return self._camera_tensors_color[env_id]
        else:
            return torch.from_numpy(self.gym.get_camera_image(
                self.sim, env_ptr, self._camera_handles[env_id],
                gymapi.IMAGE_COLOR)).to(self.device).view(
                self.height, self.width, 4)

    def _get_depth_image(self, env_ptr: int, env_id: int) -> torch.Tensor:
        if self.use_camera_tensors:
            return self._camera_tensors_depth[env_id]
        else:
            return torch.from_numpy(self.gym.get_camera_image(
                self.sim, env_ptr, self._camera_handles[env_id],
                gymapi.IMAGE_DEPTH)).to(self.device)

    def _get_segmentation_image(self, env_ptr: int, env_id: int
                                ) -> torch.Tensor:
        if self.use_camera_tensors:
            return self._camera_tensors_segmentation[env_id]
        else:
            return torch.from_numpy(self.gym.get_camera_image(
                self.sim, env_ptr, self._camera_handles[env_id],
                gymapi.IMAGE_SEGMENTATION).astype(np.uint8)).to(self.device)

    def _get_point_cloud(self, env_ptrs: List, env_ids: List[int],
                         features: str = None,
                         add_coordinate_system: bool = False
                         ) -> torch.Tensor:
        assert features in [None, 'rgb', 'seg']

        xyz = self.depth_to_xyz(env_ptrs, env_ids)

        if features:
            feature_image = getattr(self, f'_get_{features}')(env_ptrs, env_ids)
            if features == 'rgb':
                feature_shape = (len(env_ids), -1, 3)
            elif features == 'seg':
                feature_shape = (len(env_ids), -1, 1)
            features = feature_image.view(feature_shape).float().to(self.device)
            if add_coordinate_system and features == 'rgb':
                xyz, features = self.add_coordinate_system(
                    xyz, features, env_ids)
            return torch.cat([xyz, features], dim=-1)
        return xyz

    def depth_to_xyz(self, env_ptrs, env_ids):
        depth_image = self._get_d(env_ptrs, env_ids)

        # Adjusted camera projection matrix [3, 3].
        adj_proj_mat = self.get_adjusted_projection_matrix(
            env_ptrs[0], env_ids[0])
        # Camera view matrix [len(env_ids), 4, 4]
        view_mat = self.get_view_matrix(env_ptrs, env_ids)

        # Convert depth to xyz coordinates via script function
        xyz = depth_image_to_xyz(depth_image, adj_proj_mat, view_mat,
                                 self.device)
        xyz = self.global_to_environment_xyz(xyz, env_ids)
        return xyz

    def global_to_environment_xyz(self, xyz, env_ids, env_spacing: float = 1.):
        """View matrices are returned in global instead of environment
        coordinates in IsaacGym. This function projects the point-clouds into
        their environment-specific frame, which is usually desired."""
        for env_id in env_ids:
            num_per_row = int(np.sqrt(self.num_envs))
            row = int(np.floor(env_id / num_per_row))
            column = env_id % num_per_row
            xyz[env_id, :, 0] -= column * 2 * env_spacing
            xyz[env_id, :, 1] -= row * 2 * env_spacing
        return xyz

    def add_coordinate_system(self, xyz, rgb, env_ids: List[int]):
        for axis_id in range(3):
            axis_xyz = torch.zeros((len(env_ids), 200, 3)).to(
                self.device)
            axis_xyz[..., axis_id] = torch.linspace(-1, 1, 200).to(
                self.device)
            axis_rgb = torch.zeros((len(env_ids), 200, 3)).to(
                self.device)
            axis_rgb[..., axis_id] = 255
            xyz = torch.cat([xyz, axis_xyz], dim=1)
            rgb = torch.cat([rgb, axis_rgb], dim=1)
        return xyz, rgb

    def get_projection_matrix(self, env_ptr, i) -> torch.Tensor:
        return torch.from_numpy(
            self.gym.get_camera_proj_matrix(
                self.sim, env_ptr, self._camera_handles[i])).to(self.device)

    def get_view_matrix(self, env_ptrs: List[int], i: List[int]
                        ) -> torch.Tensor:
        """Returns the batch of view matrices of shape: [len(env_ptrs), 4, 4].
        The camera view matrix is returned in global instead of env coordinates
        in IsaacGym."""
        view_mat = []
        for env_ptr, env_idx in zip(env_ptrs, i):
            view_mat.append(torch.from_numpy(
                self.gym.get_camera_view_matrix(
                    self.sim, env_ptr, self._camera_handles[env_idx])
            ).to(self.device))
        return torch.stack(view_mat)

    def get_adjusted_projection_matrix(self, env_ptr, i) -> torch.Tensor:
        proj_mat = self.get_projection_matrix(env_ptr, i)
        fu = 2 / proj_mat[0, 0]
        fv = 2 / proj_mat[1, 1]
        return torch.Tensor([[fu, 0., 0.],
                             [0., fv, 0.],
                             [0., 0., 1.]]).to(self.device)


class DexterityCameraActorProperties:
    """Implements properties of actor-bodies, which visualize the camera pose
    in the simulation."""

    def __init__(
            self,
            add_camera_actor: Optional[bool] = False
    ) -> None:
        self.add_camera_actor = add_camera_actor
        self._actor_handles = []
        self._actor_ids_sim = []

    @property
    def add_camera_actor(self) -> bool:
        return self._add_camera_actor

    @add_camera_actor.setter
    def add_camera_actor(self, value: bool) -> None:
        if value:
            assert self.model is not None, \
                f"Camera model must be provided if add_camera_actor is True."
        self._add_camera_actor = value

    def get_rigid_body_count(self, gym) -> int:
        if self.add_camera_actor:
            return gym.get_asset_rigid_body_count(self._asset)
        else:
            return 0

    def get_rigid_shape_count(self, gym) -> int:
        if self.add_camera_actor:
            return gym.get_asset_rigid_shape_count(self._asset)
        else:
            return 0

    @property
    def actor_ids_sim(self) -> torch.Tensor:
        """Returns tensor os sim indices of camera actors."""
        if isinstance(self._actor_ids_sim, List):
            self._actor_ids_sim = torch.Tensor(
                self._actor_ids_sim).to(self.device, torch.int32)
        return self._actor_ids_sim


class DexterityCameraActor(DexterityCameraActorProperties):
    """Implements initialization and of camera actors and updating of attached
    actors poses."""

    def __init__(
            self,
            add_camera_actor: Optional[bool] = False
    ) -> None:
        super().__init__(add_camera_actor)

    def create_asset(self, gym, sim) -> None:
        camera_asset_file = f'{self.model}/{self.model}.urdf'
        assert os.path.isfile(os.path.join(
            self.CAMERA_ASSET_ROOT, camera_asset_file)), \
            f"Using add_camera_asset=True, but the camera description " \
            f"{os.path.join(self.CAMERA_ASSET_ROOT, camera_asset_file)}" \
            f" was not found."
        camera_asset_options = gymapi.AssetOptions()
        camera_asset_options.fix_base_link = True
        self._asset = gym.load_asset(
            sim, self.CAMERA_ASSET_ROOT, camera_asset_file,
            camera_asset_options)

    def create_actor(self, gym, device, num_envs: int, env_ptr: int, i: int,
                     camera_name: str, actor_count: int) -> None:
        """Adds an actor to the simulation that visualizes the camera."""
        if self.add_camera_actor:
            actor_handle = gym.create_actor(
                env_ptr, self._asset, self.camera_transform, camera_name, i, 0,
                0)
            self._actor_handles.append(actor_handle)
            self._actor_ids_sim.append(actor_count)
            if self.attach_to_body:
                self._acquire_local_transform_tensors(num_envs, device)

    def _acquire_local_transform_tensors(self, num_envs: int, device) -> None:
        self._local_pos = torch.Tensor(
            [[self.camera_transform.p.x,
              self.camera_transform.p.y,
              self.camera_transform.p.z]]).repeat(num_envs, 1).to(device)
        self._local_quat = torch.Tensor(
            [[self.camera_transform.r.x,
              self.camera_transform.r.y,
              self.camera_transform.r.z,
              self.camera_transform.r.w]]).repeat(num_envs, 1).to(device)

    def update_attached_camera_body_pose(
            self, body_pos: torch.Tensor, body_quat: torch.Tensor,
            root_pos: torch.Tensor, root_quat: torch.Tensor,
            root_state: torch.Tensor):
        parent_body_id = self.gym.find_actor_rigid_body_index(
            self.env_ptrs[0], self.robot_handles[0], self.attach_to_body,
            gymapi.DOMAIN_ENV)
        parent_body_pos = body_pos[:, parent_body_id, 0:3]
        parent_body_quat = body_quat[:, parent_body_id, 0:4]

        root_pos[:, self.actor_id_env] = parent_body_pos + quat_apply(
            parent_body_quat, self._local_pos)
        root_quat[:, self.actor_id_env] = quat_mul(
            parent_body_quat, self._local_quat)

        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(root_state),
            gymtorch.unwrap_tensor(self.actor_ids_sim),
            len(self.actor_ids_sim))


class DexterityCamera(DexterityCameraSensor, DexterityCameraActor,
                      DexterityVideoRecordingProperties):
    """Inherits from and merges functionalities of DexterityCameraSensor and
    DexterityCameraActor to form the DexterityCamera interface.

    Args:
        pos: (Tuple[float, float, float]) Position of the camera sensor.
        quat: (Tuple[float, float, float, float]) Quaternion rotation of the
            camera sensor.
        model: (str, optional) Model of the camera (e.g. 'realsense_d405'). If
            the camera model is specified, the properties of the sensor (fovx,
            resolution, and image_type) are inferred from its camera_info.yaml.
        fovx: (int) Horizontal field of view.
        resolution: (Tuple[int, int]) Resolution (width, height) of the camera.
        image_type: (str) Type of image that the camera sensor returns. Should
            be in ['d', 'rgb', 'rgbd', 'seg', 'rgb_seg', 'pc', 'cpc'].
        attach_to_body: (str, optional) Specifies the name of a rigid body the
            camera should be attached to. Camera will be static if
            attach_to_body is None.
        use_camera_tensors: (bool, True) Whether to use GPU tensor access when
            retrieving the camera images.
        add_camera_actor: (bool, False) Whether to add an actor of the camera
            model to the simulation. A URDF description of the camera model
            must exist to do so. This is meant as a debugging feature to
            visualize the camera poses when setting up an environment or
            attaching a camera to the robot. It is not meant to be used during
            training. While the observations appear to be unobstructed for
            static cameras, attached camera sensors will intersect with the
            body during fast motions.
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
            add_camera_actor: bool = False,
    ) -> None:
        DexterityCameraSensor.__init__(
            self, pos, quat, model, fovx, resolution, image_type,
            attach_to_body, use_camera_tensors)
        DexterityCameraActor.__init__(self, add_camera_actor)

    def simulation_setup(self, gym, sim, env_ptrs, num_envs, robot_handles,
                         actor_id_env, device) -> None:
        """Connect the camera to the IsaacGym simulation."""
        self.gym = gym
        self.sim = sim
        self.env_ptrs = env_ptrs
        self.num_envs = num_envs
        self.robot_handles = robot_handles
        self.actor_id_env = actor_id_env
        self.device = device


class DexterityBaseCameras:
    def parse_camera_spec(self, cfg) -> None:
        self.camera_tensors_enabled = False
        # Create cameras.
        self._camera_dict = {}
        if 'cameras' in self.cfg_env.keys():
            for camera_name, camera_cfg in self.cfg_env.cameras.items():
                if camera_name in cfg['env']['observations']:
                    self._camera_dict[camera_name] = DexterityCamera(
                        **camera_cfg)

                    # Enable camera tensors if any of the cameras uses them.
                    if self._camera_dict[camera_name].use_camera_tensors:
                        self.camera_tensors_enabled = True

    def get_images(self) -> Dict[str, torch.Tensor]:
        """Retrieve images from all camera sensors and in all environments."""
        if not self.camera_handles_created:
            self.camera_setup()

        # Move attached camera actors.
        for camera_name, dexterity_camera in self._camera_dict.items():
            if dexterity_camera.add_camera_actor and \
                    dexterity_camera.attach_to_body:
                dexterity_camera.update_attached_camera_body_pose(
                    self.body_pos, self.body_quat, self.root_pos,
                    self.root_quat, self.root_state)

        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)

        # Retrieve images for all cameras.
        if self.camera_tensors_enabled:
            self.gym.start_access_image_tensors(self.sim)
        image_dict = {}
        for camera_name, dexterity_camera in self._camera_dict.items():
            image_dict[camera_name] = dexterity_camera.get_image(
                self.env_ptrs, list(range(self.num_envs)))
        if self.camera_tensors_enabled:
            self.gym.end_access_image_tensors(self.sim)
        return image_dict

    @property
    def camera_handles_created(self) -> bool:
        return len(list(self._camera_dict.values())[0]._camera_handles) > 0

    def camera_setup(self) -> None:
        for camera_name, dexterity_camera in self._camera_dict.items():
            actor_id_env = getattr(self, camera_name + "_actor_id_env") if \
                dexterity_camera.add_camera_actor else None
            dexterity_camera.simulation_setup(
                self.gym, self.sim, self.env_ptrs, self.num_envs,
                self.robot_handles, actor_id_env, self.device)

        for env_id in range(self.num_envs):
            for camera_name, dexterity_camera in self._camera_dict.items():
                dexterity_camera.create_camera(self.env_ptrs[env_id],
                                               self.robot_handles[env_id])

    @property
    def camera_count(self) -> int:
        return len(self._camera_dict.keys())

    @property
    def camera_rigid_body_count(self) -> int:
        rigid_body_count = 0
        for camera_name, dexterity_camera in self._camera_dict.items():
            if dexterity_camera.add_camera_actor:
                if not hasattr(dexterity_camera, "_camera_asset"):
                    dexterity_camera.create_asset(self.gym, self.sim)
            rigid_body_count += dexterity_camera.get_rigid_body_count(self.gym)
        return rigid_body_count

    @property
    def camera_rigid_shape_count(self) -> int:
        rigid_shape_count = 0
        for camera_name, dexterity_camera in self._camera_dict.items():
            if dexterity_camera.add_camera_actor:
                if not hasattr(dexterity_camera, "_camera_asset"):
                    dexterity_camera.create_asset(self.gym, self.sim)
            rigid_shape_count += dexterity_camera.get_rigid_shape_count(
                self.gym)
        return rigid_shape_count

    def create_camera_actors(self, env_prt: int, i: int, actor_count: int
                             ) -> None:
        for camera_name, dexterity_camera in self._camera_dict.items():
            if dexterity_camera.add_camera_actor:
                dexterity_camera.create_actor(
                    self.gym, self.device, self.num_envs, env_prt, i, camera_name,
                    actor_count)
                actor_count += 1

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

            for env_id in range(self.num_envs):
                if self._camera_dict[camera_name].image_type in \
                        ['rgb', 'rgb_seg']:
                    np_image = image_dict[camera_name][env_id].cpu().numpy()[
                               ..., ::-1]
                elif self._camera_dict[camera_name].image_type == 'd':
                    np_image = image_dict[camera_name][env_id].unsqueeze(
                        -1).cpu().numpy()
                    np_image = -np.repeat(np_image, 3, axis=-1)
                    np_image = max_recording_depth - np_image.clip(
                        min=0, max=max_recording_depth)
                    np_image = (np_image * 255 / max_recording_depth).astype(
                        np.uint8)
                elif self._camera_dict[camera_name].image_type == 'rgbd':
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

                elif self._camera_dict[camera_name].image_type == 'seg':
                    seg_image = image_dict[camera_name][env_id]
                    #import matplotlib.pyplot as plt
                    #plt.imshow(seg_image.cpu().numpy())
                    #plt.show()
                    np_image = self._camera_dict[
                        camera_name].segmentation_to_rgb(
                        seg_image).cpu().numpy()

                elif 'pc' in self._camera_dict[camera_name].image_type:
                    from pytorch3d.structures import Pointclouds
                    xyz = image_dict[camera_name][env_id, :, 0:3]
                    valid_idx = (torch.norm(xyz, dim=1) < max_recording_depth
                                 ).nonzero().squeeze(-1)
                    xyz = xyz[valid_idx]

                    if self._camera_dict[camera_name].image_type == 'pc_rgb':
                        rgb = (image_dict[camera_name][env_id, :, 3:6] / 255)[
                            valid_idx]
                    elif self._camera_dict[camera_name].image_type == 'pc_seg':
                        seg_id = image_dict[camera_name][env_id, :, 3][
                            valid_idx]
                        rgb = self._camera_dict[
                            camera_name].segmentation_to_rgb(seg_id) / 255
                    else:
                        rgb = torch.zeros_like(xyz)

                    pytorch3d_pc = Pointclouds(
                        points=xyz.unsqueeze(0), features=rgb.unsqueeze(0))
                    np_image = (255 * self._camera_dict[
                        camera_name].point_cloud_renderer(
                        pytorch3d_pc, gamma=(1e-4,),
                        bg_col=torch.tensor([1.0, 1.0, 1.0]).to(self.device)
                    ).cpu().numpy()).astype(np.uint8)[0, ..., ::-1]

                else:
                    assert False, \
                        f"Cannot write videos to file for image type " \
                        f"{self._camera_dict[camera_name].image_type} yet."
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

        # Point cloud image is created by renderer in dimensions defined in
        # the DexterityVideoRecordingProperties.
        if 'pc' in self._camera_dict[camera_name].image_type:
            width = self._camera_dict[camera_name].point_cloud_render_width
            height = self._camera_dict[camera_name].point_cloud_render_height
        # RGBD has rgb and depth image next to each other
        elif self._camera_dict[camera_name].image_type == 'rgbd':
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
