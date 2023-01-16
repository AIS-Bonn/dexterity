import numpy as np
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask
import dexterityvr

import torch
from typing import *
import numpy as np
import time
import math

import multiprocessing as mp
import isaacgymenvs.tasks.dexterity.dexterity_control as ctrl
import threading
import time


def render_process_target(render_size,
                          np_left_eye_arr, np_right_eye_arr,
                          np_left_eye_transform, np_right_eye_transform,
                          np_left_eye_fov, np_right_eye_fov,
                          np_headset_pose, np_tracker_pose,
                          np_sg_sensor_data, np_sg_flexions,
                          render_headset,
                          vr_init_done):
    vr_viewer = dexterityvr.VRViewer(
        render_size, np_left_eye_arr, np_right_eye_arr)

    np_left_eye_transform[:] = vr_viewer.get_left_eye_transform()
    np_right_eye_transform[:] = vr_viewer.get_right_eye_transform()
    np_left_eye_fov[:] = vr_viewer.get_left_eye_fov() * (180 / np.pi)
    np_right_eye_fov[:] = vr_viewer.get_right_eye_fov() * (180 / np.pi)
    vr_init_done.set()

    while True:
        # Update tracked poses and sensor angles
        np_headset_pose[:] = vr_viewer.get_headset_pose()
        np_tracker_pose[:] = vr_viewer.get_tracker_pose()
        #np_sg_sensor_data[:] = vr_viewer.get_glove_sensor_angles(True)
        #np_sg_flexions[:] = vr_viewer.get_glove_flexions(True)

        if render_headset.is_set():
            vr_viewer.submit_vr_camera_images()
            render_headset.clear()


class GymVR:
    def __init__(self, gym, sim, env, headless: bool,
                 render_size: Tuple = (1851, 2055),  # recommended: 2468, 2740
                 use_camera_tensors: bool = True,
                 use_multiprocessing: bool = False) -> None:
        self._gym = gym
        self._sim = sim
        self._env = env
        self._headless = headless
        self._use_camera_tensors = use_camera_tensors
        self._use_multiprocessing = use_multiprocessing
        self._reference_frame_set = False
        self._render_size = np.array(render_size, dtype=np.uint32)

        self._init_numpy_buffers()

        if self._use_multiprocessing:
            self._render_headset = mp.Event()  # Set to True when textures should be rendered
            self._vr_init_done = mp.Event()  # Set to True when the VR system has been initialized and all relevant information are available
            self._vr_process = mp.Process(
                target=render_process_target,
                args=(self._render_size,
                      self._np_left_eye_arr, self._np_right_eye_arr,
                      self._np_left_eye_transform, self._np_right_eye_transform,
                      self._np_left_eye_fov, self._np_right_eye_fov,
                      self._np_headset_pose, self._np_tracker_pose,
                      self._np_sg_sensor_data, self._np_sg_flexions,
                      self._render_headset, self._vr_init_done))
            self._vr_process.start()

            while not self._vr_init_done.is_set():
                print("waiting for end of VR initialization")
                time.sleep(0.5)

        else:
            self._vr_viewer = dexterityvr.VRViewer(
                self._render_size, self._np_left_eye_arr, self._np_right_eye_arr)

        if not self._use_multiprocessing:
            self._init_eye_properties()
        self._create_vr_cameras()

        self._start_time = time.time()
        self._simulated_steps = 0
        self._fps = 0

        # y coordinate is mapped to IsaacGym's z coordinate
        self._tracker_base_pos = np.array([0, 0.5, 0])

        input("Press Enter to set reference frame ...")
        self._set_reference_frame()

    def __getattr__(self, item):
        return getattr(self._gym, item)

    def _init_numpy_buffers(self):
        arr_shape = (self._render_size[1], self._render_size[0], 4)

        # Define numpy buffers for images
        self.arr_shape = arr_shape
        self._mp_left_eye_arr = mp.RawArray("B", int(np.prod(arr_shape)))
        self._np_left_eye_arr = np.frombuffer(
            self._mp_left_eye_arr,
            dtype=np.uint8).reshape(arr_shape)
        self._mp_right_eye_arr = mp.RawArray("B", int(np.prod(arr_shape)))
        self._np_right_eye_arr = np.frombuffer(
            self._mp_right_eye_arr,
            dtype=np.uint8).reshape(arr_shape)

        # Define numpy buffers for eye transforms and fovs
        self._mp_left_eye_transform = mp.RawArray("f", 3)
        self._np_left_eye_transform = np.frombuffer(
            self._mp_left_eye_transform, dtype=np.float32)
        self._mp_right_eye_transform = mp.RawArray("f", 3)
        self._np_right_eye_transform = np.frombuffer(
            self._mp_right_eye_transform, dtype=np.float32)
        self._mp_left_eye_fov = mp.RawArray("f", 1)
        self._np_left_eye_fov = np.frombuffer(
            self._mp_left_eye_fov, dtype=np.float32)
        self._mp_right_eye_fov = mp.RawArray("f", 1)
        self._np_right_eye_fov = np.frombuffer(
            self._mp_right_eye_fov, dtype=np.float32)

        # Define numpy buffers for headset and tracker
        self._mp_headset_pose = mp.RawArray("f", 7)
        self._np_headset_pose = np.frombuffer(
            self._mp_headset_pose, dtype=np.float32)
        self._mp_tracker_pose = mp.RawArray("f", 7)
        self._np_tracker_pose = np.frombuffer(
            self._mp_tracker_pose, dtype=np.float32)

        # Define numpy buffers for SenseGlove data
        self._mp_sg_sensor_data = mp.RawArray("f", 5 * 4)
        self._np_sg_sensor_data = np.frombuffer(
            self._mp_sg_sensor_data, dtype=np.float32).reshape([5, 4])
        self._mp_sg_flexions = mp.RawArray("f", 5)
        self._np_sg_flexions = np.frombuffer(
            self._mp_sg_flexions, dtype=np.float32)

    def _init_eye_properties(self) -> None:
        self._np_left_eye_transform[:] = self._vr_viewer.get_left_eye_transform()
        self._np_right_eye_transform[:] = self._vr_viewer.get_right_eye_transform()
        self._np_left_eye_fov[:] = self._vr_viewer.get_left_eye_fov() * (180 / np.pi)
        self._np_right_eye_fov[:] = self._vr_viewer.get_right_eye_fov() * (180 / np.pi)

    def simulate(self, sim) -> None:
        self._gym.simulate(sim)
        self.simulate_vr()
        if self._simulated_steps % 25 == 0:
            self._fps = 0.8 * self._fps + 0.2 * (self._simulated_steps / (time.time() - self._start_time))
            print("Avg. FPS:", self._fps)
            self._simulated_steps = 0
            self._start_time = time.time()
        self._simulated_steps += 1

    def simulate_vr(self):
        self._update_vr_camera_poses()
        self._submit_vr_camera_images()

    def _create_vr_cameras(self):
        # Set camera properties
        camera_props = gymapi.CameraProperties()
        camera_props.width = self._render_size[0]
        camera_props.height = self._render_size[1]
        camera_props.horizontal_fov = self._np_left_eye_fov[0]
        camera_props.enable_tensors = self._use_camera_tensors

        # Create camera handles
        self.camera_handle_left = self._gym.create_camera_sensor(
            self._env, camera_props)
        self.camera_handle_right = self._gym.create_camera_sensor(
            self._env, camera_props)

        if self._use_camera_tensors:
            self.camera_tensor_left = self._gym.get_camera_image_gpu_tensor(
                self._sim, self._env, self.camera_handle_left, gymapi.IMAGE_COLOR)
            self.torch_camera_tensor_left = gymtorch.wrap_tensor(
                self.camera_tensor_left)
            self.camera_tensor_right = self._gym.get_camera_image_gpu_tensor(
                self._sim, self._env, self.camera_handle_right,
                gymapi.IMAGE_COLOR)
            self.torch_camera_tensor_right = gymtorch.wrap_tensor(
                self.camera_tensor_right)

    def _update_vr_camera_poses(self):
        if not self._use_multiprocessing:
            self._np_headset_pose[:] = self._vr_viewer.get_headset_pose()
            self._np_tracker_pose[:] = self._vr_viewer.get_tracker_pose()

        # Transform head pose to IsaacGym coordinates
        head_pos = gymapi.Vec3(-self.headset_pose[2], -self.headset_pose[0], self.headset_pose[1])
        q_w, q_x, q_y, q_z = self.headset_pose[3:]
        head_quat = gymapi.Quat(-q_z, -q_x, q_y, q_w)

        left_eye_offset = gymapi.Vec3(
            -self._np_left_eye_transform[2],
            -self._np_left_eye_transform[0],
            self._np_left_eye_transform[1])
        right_eye_offset = gymapi.Vec3(
            -self._np_right_eye_transform[2],
            -self._np_right_eye_transform[0],
            self._np_right_eye_transform[1])
        left_eye_offset = head_quat.rotate(left_eye_offset)
        right_eye_offset = head_quat.rotate(right_eye_offset)

        left_eye_pos = head_pos + left_eye_offset
        right_eye_pos = head_pos + right_eye_offset

        left_eye_transform = gymapi.Transform(
            p=left_eye_pos, r=head_quat)
        right_eye_transform = gymapi.Transform(
            p=right_eye_pos, r=head_quat)

        # Update camera poses in IsaacGym
        self._gym.set_camera_transform(self.camera_handle_left, self._env,
                                       left_eye_transform)
        self._gym.set_camera_transform(self.camera_handle_right, self._env,
                                       right_eye_transform)

    def _submit_vr_camera_images(self):
        if self._headless:
            self._gym.step_graphics(self._sim)
        self._gym.render_all_camera_sensors(self._sim)

        if self._use_camera_tensors:
            self._gym.start_access_image_tensors(self._sim)
            self._np_left_eye_arr[:] = self.torch_camera_tensor_left.flip(0).cpu().numpy()
            self._np_right_eye_arr[:] = self.torch_camera_tensor_right.flip(0).cpu().numpy()
            self._gym.end_access_image_tensors(self._sim)
        else:
            self._np_left_eye_arr[:] = np.flip(self._gym.get_camera_image(
                self._sim, self._env, self.camera_handle_left,
                gymapi.IMAGE_COLOR).reshape(
                self._render_size[1], self._render_size[0], 4), 0).copy()
            self._np_right_eye_arr[:] = np.flip(self._gym.get_camera_image(
                self._sim, self._env, self.camera_handle_right,
                gymapi.IMAGE_COLOR).reshape(
                self._render_size[1], self._render_size[0], 4), 0).copy()

        if self._use_multiprocessing:
            while self._render_headset.is_set():
                #print("waiting for render event to be unset ...")
                time.sleep(0.0001)
            self._render_headset.set()
        else:
            self._vr_viewer.submit_vr_camera_images()

    def _set_reference_frame(self):
        if not self._use_multiprocessing:
            self._np_tracker_pose[:] = self._vr_viewer.get_tracker_pose()
        self._tracker_reference_pose = self._np_tracker_pose.copy()

    @property
    def tracker_pose(self):
        tracker_pose = self._np_tracker_pose.copy()
        tracker_pose[0:3] -= self._tracker_reference_pose[0:3]
        tracker_pose[0:3] += self._tracker_base_pos
        return tracker_pose

    @property
    def headset_pose(self):
        headset_pose = self._np_headset_pose.copy()
        headset_pose[0:3] -= self._tracker_reference_pose[0:3]
        headset_pose[0:3] += self._tracker_base_pos
        return headset_pose

    @property
    def glove_sensor_angles(self):
        if self._use_multiprocessing:
            return self._np_sg_sensor_data
        else:
            return self._vr_viewer.get_glove_sensor_angles(True)

    @property
    def glove_flexions(self):
        if self._use_multiprocessing:
            return self._np_sg_flexions
        else:
            return self._vr_viewer.get_glove_flexions(True)


class VRVecTask:
    def __init__(self, task: VecTask) -> None:
        self._task = task

        self._task.gym = GymVR(task.gym, task.sim, task.env_ptrs[0], task.headless)

        # Initialize inverse scaling factors for pose actions
        self.inv_pos_action_scale = (
                1 / torch.tensor(self._task.cfg_base.ctrl.pos_action_scale,
                                 device=self._task.device))
        self.inv_rot_action_scale = (
                1 / torch.tensor(self._task.cfg_base.ctrl.rot_action_scale,
                                 device=self._task.device))

        # Init residual dof ranges
        self.residual_dof_lower_limits = \
            self._task.robot_dof_lower_limits[:,
            self._task.residual_actuated_dof_indices]
        self.residual_dof_upper_limits = \
            self._task.robot_dof_upper_limits[:,
            self._task.residual_actuated_dof_indices]

    def __getattr__(self, item):
        return getattr(self._task, item)

    def step(self, actions: torch.Tensor) -> Tuple[
        Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:


        # Synchronize physics simulation with rendering rate in headless mode
        if self._task.headless:
            self._task.gym.sync_frame_time(self._task.sim)


        # Override pose and residual DoF actions
        actions[:, 0:6] = self._get_ik_body_pose_actions()
        residual_dof_actions = self._get_residual_dof_actions()

        #print("residual_dof_actions:", residual_dof_actions)
        #residual_dof_actions[:, 1] = 0. # Overwrite to test thumb while normalized flexions are not working yet
        actions[:, 6:] = residual_dof_actions
        return self._task.step(actions)

    def _get_ik_body_pose_actions(self):
        ik_body_pose_actions = torch.zeros((1, 6), device=self.device)
        tracker_pose = self._task.gym.tracker_pose

        # Adjust to IsaacGym coordinate system
        tracker_target_pos = torch.tensor(
            [[-tracker_pose[2], -tracker_pose[0], tracker_pose[1]]],
            device=self.device)

        # IsaacGym convention is [x, y, z, w] and flipping of tracker
        w = abs(tracker_pose[5])
        x = tracker_pose[4] * math.copysign(1, tracker_pose[5])
        y = -tracker_pose[6] * math.copysign(1, tracker_pose[5])
        z = -tracker_pose[3] * math.copysign(1, tracker_pose[5])
        tracker_target_quat = torch.tensor([[x, y, z, w]], device=self.device)

        if self._task.cfg_base.ctrl.add_pose_actions_to == 'pose':
            current_pos = self.ik_body_pos
            current_quat = self.ik_body_quat
        elif self._task.cfg_base.ctrl.add_pose_actions_to == 'target':
            current_pos = self.ctrl_target_ik_body_pos
            current_quat = self.ctrl_target_ik_body_quat
        else:
            assert False

        pos_error, axis_angle_error = ctrl.get_pose_error(
            current_pos, current_quat, tracker_target_pos,
            tracker_target_quat, jacobian_type='geometric',
            rot_error_type='axis_angle')

        ik_body_pose_actions[:, 0:3] = self.inv_pos_action_scale * pos_error
        ik_body_pose_actions[:, 3:6] = \
            self.inv_rot_action_scale * axis_angle_error
        ik_body_pose_actions = torch.clamp(ik_body_pose_actions, -1, 1)
        return ik_body_pose_actions

    def _get_residual_dof_actions(self):
        residual_dof_actions = []

        sensor_angles = self._task.gym.glove_sensor_angles
        flexions = self._task.gym.glove_flexions

        actuation_values = self._task.robot.model.teleop_mapping(sensor_angles, flexions)

        # Generate array of residual dof targets in the order they appear in the model actuators
        for actuator_name in self._task.robot.model.actuator_targets:
            if actuator_name in actuation_values.keys():
                residual_dof_actions.append(actuation_values[actuator_name])

        residual_dof_actions = torch.tensor(
            residual_dof_actions, device=self.device,
            dtype=torch.float32).unsqueeze(0)

        # unscale residual_dof_actions as they are scaled when generating the control targets
        residual_dof_actions = torch.clamp(unscale(
            residual_dof_actions, self.residual_dof_lower_limits,
            self.residual_dof_upper_limits), min=-1, max=1)

        #print("residual_dof_actions:", residual_dof_actions)

        if self.cfg_base.ctrl.relative_residual_actions:
            residual_dof_actions = torch.clamp(
                (residual_dof_actions - self.ctrl_target_residual_actuated_dof_pos) /
                (self.cfg_base.sim.dt * self.cfg_base.ctrl.relative_residual_target_change_scale),
            min=-1, max=1)
        return residual_dof_actions




