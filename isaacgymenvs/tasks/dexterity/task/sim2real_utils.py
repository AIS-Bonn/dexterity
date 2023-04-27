import torch
from typing import *
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from collections import defaultdict
import numpy as np
import isaacgymenvs.tasks.dexterity.dexterity_control as ctrl
import cv2
import matplotlib.pyplot as plt


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


class CalibrationUtils:
    def run_calibration_procedure(
            self,
            actions,
            arm_calibration_time_steps: int = 500,
            hand_calibration_time_steps: int = 2000
    ) -> torch.Tensor:
        assert self.cfg_task["rl"]["max_episode_length"] > \
               arm_calibration_time_steps + hand_calibration_time_steps, \
               "Increase the maximum episode length so the full calibration " \
               "procedure can runs without resets."

        actions = torch.zeros_like(actions)

        if self.progress_buf[0] <= arm_calibration_time_steps:
            actions = self.run_arm_calibration(actions, start_time_step=0, end_time_step=arm_calibration_time_steps)
        if self.progress_buf[0] >= arm_calibration_time_steps:
            actions = self.run_hand_calibration(actions, start_time_step=arm_calibration_time_steps, end_time_step=arm_calibration_time_steps + hand_calibration_time_steps)

        return actions

    def _calibration_move_to_pose(self, start_pos: List, start_quat: List,
                                  end_pos: List, end_quat: List, start_ts: int,
                                  end_ts: int) -> None:
        if start_ts <= self.progress_buf[0] < end_ts:
            progress = (self.progress_buf[0] - start_ts) / (end_ts - start_ts)
            target_pos = progress * torch.tensor(end_pos) + (
                    1 - progress) * torch.tensor(start_pos)
            start_rot = R.from_quat(start_quat)
            end_rot = R.from_quat(end_quat)
            target_quat = torch.from_numpy(Slerp(
                [0., 1.], R.concatenate([start_rot, end_rot]))(progress).as_quat())
            self._calibration_target_pos = target_pos.to(self.device).unsqueeze(0).repeat(
                self.num_envs, 1)
            self._calibration_target_quat = target_quat.to(self.device).unsqueeze(0).repeat(
                self.num_envs, 1)

    def _calibration_follow_xy_circle(self, radius_in_m: float,
                                      cycle_time_in_sec: float, start_ts: int,
                                      end_ts: int) -> None:
        if start_ts <= self.progress_buf[0] < end_ts:
            # Save starting pose to return to it in the end.
            if self.progress_buf[0] == start_ts:
                self._calibration_circle_start_pos = self.ik_body_pos.clone()
                self._calibration_circle_start_quat = self.ik_body_quat.clone()

            on_circle = self._calibration_circle_start_pos[0].clone()
            on_circle[1] += radius_in_m

            # Set target pos on the circle
            cycle_time_steps = cycle_time_in_sec / self.cfg_base['sim']['dt']
            x_diff = radius_in_m * torch.sin(
                2 * np.pi * (self.progress_buf[0] - start_ts - 100) / cycle_time_steps)
            y_diff = radius_in_m * torch.cos(
                2 * np.pi * (self.progress_buf[0] - start_ts - 100) / cycle_time_steps) - radius_in_m
            self._calibration_target_pos = on_circle.clone()
            self._calibration_target_pos[0] += x_diff
            self._calibration_target_pos[1] += y_diff
            self._calibration_target_pos = self._calibration_target_pos.unsqueeze(
                0).repeat(self.num_envs, 1)

            # Move from the center to the edge of the circle in the beginning and
            # back at the end.
            self._calibration_move_to_pose(
                start_pos=self._calibration_circle_start_pos[0],
                start_quat=self._calibration_circle_start_quat[0],
                end_pos=on_circle, end_quat=self._calibration_circle_start_quat[0],
                start_ts=start_ts, end_ts=start_ts + 100)
            self._calibration_move_to_pose(
                start_pos=self.ik_body_pos[0],
                start_quat=self.ik_body_quat[0],
                end_pos=on_circle, end_quat=self._calibration_circle_start_quat[0],
                start_ts=end_ts - 100, end_ts=end_ts)

    def run_arm_calibration(self, actions, start_time_step: int, end_time_step: int):

        # Move to initial pose.
        self._calibration_move_to_pose(
            start_pos=[0.28, 0.58, 0.3], start_quat=[0, 0, 0.707, 0.707],
            end_pos=[0.28, 0.58, 0.3], end_quat=[0, 0, 0.707, 0.707],
            start_ts=0, end_ts=100)

        # Do a slow circle.
        self._calibration_follow_xy_circle(
            radius_in_m=0.075, cycle_time_in_sec=20., start_ts=100, end_ts=1100)
        # Do a fast circle
        self._calibration_follow_xy_circle(
            radius_in_m=0.075, cycle_time_in_sec=10., start_ts=1100, end_ts=2100)

        if self.cfg_base.ctrl.add_pose_actions_to == 'pose':
            current_pos = self.ik_body_pos
            current_quat = self.ik_body_quat
        elif self.cfg_base.ctrl.add_pose_actions_to == 'target':
            current_pos = self.ctrl_target_ik_body_pos
            current_quat = self.ctrl_target_ik_body_quat
        else:
            assert False

        if self.progress_buf[0] == 0:
            actions *= 0.
            self._arm_calibration_initial_pos = self.ik_body_pos.clone()
            self.arm_calibration = defaultdict(list)
            self.calibration_time_steps = []
            self.simulated_flange_pos = []
            self.real_flange_pos = []
            self.simulated_arm_joint_pos = []
            self.real_arm_joint_pos = []
            self.command_arm_joint_pos = []

        else:
            pos_error, axis_angle_error = ctrl.get_pose_error(
                current_pos, current_quat, self._calibration_target_pos,
                self._calibration_target_quat, jacobian_type='geometric',
                rot_error_type='axis_angle')

            actions[:, 0:3] = 5. * pos_error
            actions[:, 3:6] = 5. * axis_angle_error
            actions = torch.clamp(actions, -1, 1)

        self.arm_calibration['time_step'].append(
            self.progress_buf[0].cpu().numpy().copy())
        self.arm_calibration['simulated_flange_pos'].append(
            self.flange_pos[0].cpu().numpy().copy())
        self.arm_calibration['simulated_joint_pos'].append(
            self.dof_pos[0, 0:self.ik_body_dof_count].cpu().numpy().copy())
        self.arm_calibration['command_joint_pos'].append(
            self.ctrl_target_dof_pos[0, 0:self.ik_body_dof_count].cpu().numpy().copy())

        if self.cfg_base.ros_activate:
            self.arm_calibration['real_flange_pos'].append(
                self.ros_arm_interface.get_transform('base_link', 'flange')[0])
            self.arm_calibration['real_joint_pos'].append(
                self.ros_arm_interface.get_joint_position())

        if self.progress_buf[0] == end_time_step and False:
            import matplotlib.pyplot as plt

            for k, v in self.arm_calibration.items():
                if k == 'time_step':
                    self.arm_calibration[k] = np.array(v)
                else:
                    self.arm_calibration[k] = np.stack(v)

            fig, axs = plt.subplots(3, 6)
            fig.suptitle('UR5 Arm Calibration')

            for j, joint_name in enumerate(self.robot.arm.joint_names):
                axs[1, j].set_title(joint_name)
                axs[1, j].plot(self.arm_calibration['time_step'][0:800],
                                  self.arm_calibration['simulated_joint_pos'][200:1000, j], 'r-',
                                  label='Simulation')
                axs[1, j].plot(self.arm_calibration['time_step'][0:800],
                                  self.arm_calibration['command_joint_pos'][200:1000, j],
                                  'b--', label='Command')
                if self.cfg_base.ros_activate:
                    axs[1, j].plot(self.arm_calibration['time_step'][0:800],
                                   self.arm_calibration['real_joint_pos'][200:1000, j],
                                   'g-', label='Real')
            axs[1, 0].legend(loc="upper right")
            axs[1, 0].set_ylabel('Joint position [rad]')

            for j, joint_name in enumerate(self.robot.arm.joint_names):
                axs[2, j].set_title(joint_name)
                axs[2, j].plot(self.arm_calibration['time_step'][0:800],
                                  self.arm_calibration['simulated_joint_pos'][1200:2000, j], 'r-',
                                  label='Simulation')
                axs[2, j].plot(self.arm_calibration['time_step'][0:800],
                                  self.arm_calibration['command_joint_pos'][1200:2000, j],
                                  'b--', label='Command')
                if self.cfg_base.ros_activate:
                    axs[2, j].plot(self.arm_calibration['time_step'][0:800],
                                   self.arm_calibration['real_joint_pos'][1200:2000, j],
                                   'g-', label='Real')
            axs[2, 0].legend(loc="upper right")
            axs[2, 0].set_ylabel('Joint position [rad]')

            for i, axis in enumerate(['x', 'y', 'z']):
                axs[0, i].set_title(f'flange_pos [{axis}]')
                axs[0, i].plot(self.arm_calibration['time_step'][0:800],
                            self.arm_calibration['simulated_flange_pos'][200:1000, i], 'r-',
                                   label='Simulation')
                if self.cfg_base.ros_activate:
                    axs[0, i].plot(self.arm_calibration['time_step'][0:800],
                                   self.arm_calibration['real_flange_pos'][200:1000, i], 'g-', label='Real')
            axs[0, 0].legend(loc="upper right")
            axs[0, 0].set_ylabel('Position [m]')

            for i, axis in enumerate(['x', 'y', 'z']):
                axs[0, i + 3].set_title(f'flange_pos [{axis}]')
                axs[0, i + 3].plot(self.arm_calibration['time_step'][0:800],
                            self.arm_calibration['simulated_flange_pos'][1200:2000, i], 'r-',
                                   label='Simulation')
                if self.cfg_base.ros_activate:
                    axs[0, i + 3].plot(self.arm_calibration['time_step'][0:800],
                                   self.arm_calibration['real_flange_pos'][1200:2000, i], 'g-', label='Real')

            plt.show()
        return actions

    def run_hand_calibration(self, actions, start_time_step: int, end_time_step: int,
                             hand_calibration_pos: List = [0.61, 0.31, 0.425],
                             hand_calibration_quat: List = [0.542, 0.542, 0.455, 0.455]):

        print("self.progress_buf[0]:", self.progress_buf[0])

        if self.progress_buf[0] == start_time_step:
            self.hand_calibration = defaultdict(list)

        # Move to hand calibration pose.
        self._calibration_move_to_pose(
            start_pos=self.ik_body_pos[0], start_quat=self.ik_body_quat[0],
            end_pos=hand_calibration_pos, end_quat=hand_calibration_quat,
            start_ts=start_time_step, end_ts=start_time_step + 200)

        if self.cfg_base.ctrl.add_pose_actions_to == 'pose':
            current_pos = self.ik_body_pos
            current_quat = self.ik_body_quat
        elif self.cfg_base.ctrl.add_pose_actions_to == 'target':
            current_pos = self.ctrl_target_ik_body_pos
            current_quat = self.ctrl_target_ik_body_quat
        else:
            assert False

        pos_error, axis_angle_error = ctrl.get_pose_error(
            current_pos, current_quat, self._calibration_target_pos,
            self._calibration_target_quat, jacobian_type='geometric',
            rot_error_type='axis_angle')

        actions[:, 0:3] = 5. * pos_error
        actions[:, 3:6] = 5. * axis_angle_error
        actions = torch.clamp(actions, -1, 1)

        # Open the hand.
        if self.progress_buf[0] < start_time_step + 300:
            actions[:, 6:8] = -1.
            actions[:, 8:] = 1.
        # Flex index finger.
        elif self.progress_buf[0] < start_time_step + 400:
            actions[:, 8] = -0.1
        # Release index finger.
        elif self.progress_buf[0] < start_time_step + 500:
            actions[:, 8] = 0.1
        # Flex index finger.
        elif self.progress_buf[0] < start_time_step + 600:
            actions[:, 8] = -0.5
        # Release index finger.
        elif self.progress_buf[0] < start_time_step + 700:
            actions[:, 8] = 0.5
        # Flex index finger.
        elif self.progress_buf[0] < start_time_step + 800:
            actions[:, 8] = -1.0
        # Release index finger.
        elif self.progress_buf[0] < start_time_step + 1000:
            actions[:, 8] = 1.0

        # Flex thumb.
        elif self.progress_buf[0] < start_time_step + 1100:
            actions[:, 7] = 0.1
        # Release thumb.
        elif self.progress_buf[0] < start_time_step + 1200:
            actions[:, 7] = -0.1
        # Flex thumb.
        elif self.progress_buf[0] < start_time_step + 1300:
            actions[:, 7] = 0.5
        # Release thumb.
        elif self.progress_buf[0] < start_time_step + 1400:
            actions[:, 7] = -0.5
        # Flex thumb.
        elif self.progress_buf[0] < start_time_step + 1500:
            actions[:, 7] = 1.0
        # Release thumb.
        elif self.progress_buf[0] < start_time_step + 1600:
            actions[:, 7] = -1.0

        if self.progress_buf[0] <= end_time_step:
            self.hand_calibration['time_step'].append(
                self.progress_buf[0].cpu().numpy().copy())
            self.hand_calibration['simulated_joint_pos'].append(
                self.dof_pos[0, self.ik_body_dof_count:self.robot_dof_count].cpu().numpy().copy())
            self.hand_calibration['command_joint_pos'].append(
                self.ctrl_target_dof_pos[0, self.ik_body_dof_count:self.robot_dof_count].cpu().numpy().copy())

            if self.cfg_base.ros_activate:
                self.hand_calibration['real_hand_images'].append(self.ros_image_subscriber.get_np_image())
            else:
                self.hand_calibration['real_hand_images'].append(np.load(f'real_hand_image_{self.progress_buf[0] - start_time_step}.npy'))

        if self.progress_buf[0] == end_time_step:
            # Convert real hand images to the real hand joint pos by
            # detecting the ArUco markers in the images.
            aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
            aruco_params = cv2.aruco.DetectorParameters_create()

            for i, real_hand_image in enumerate(self.hand_calibration['real_hand_images']):
                (corners, ids, rejected) = cv2.aruco.detectMarkers(
                    real_hand_image, aruco_dict, parameters=aruco_params)
                ids = [] if ids is None else [id[0] for id in ids]

                # Overwrite real hand images with current run.
                if self.cfg_base.ros_activate:
                    np.save(f'real_hand_image_{i}.npy', real_hand_image)

                all_markers_detected = set(ids) == {0, 1, 2, 3, 4}

                print("ids:", ids)
                print("all_markers_detected:", all_markers_detected)

                #if corners is None or ids is None:
                #    print("No ArUco markers detected.")

                #    plt.imshow(real_hand_image)
                #    plt.show()

                vector = np.zeros((5, 2))
                if all_markers_detected:
                    for c, id in zip(corners, ids):
                        c = c.reshape((4, 2))
                        top_left, top_right, bottom_right, bottom_left = c
                        vector[id] = top_right - top_left

                        # Draw detected sides.
                        top_left = (int(top_left[0]), int(top_left[1]))
                        top_right = (int(top_right[0]), int(top_right[1]))
                        bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
                        bottom_right = (
                        int(bottom_right[0]), int(bottom_right[1]))

                        cv2.arrowedLine(real_hand_image, top_left, top_right, (0, 255, 0),
                                        thickness=5)

                    if i % 10 == 0:
                        print("i:", i)
                        #plt.imshow(real_hand_image)
                        #plt.show()

                if np.any(vector == 0):
                    print(f"Found 0 entries in the detected vector {vector}.")

                th_proximal_to_th_inter = angle_between(vector[0], vector[3])
                th_inter_to_th_distal = angle_between(vector[3], vector[4])
                palm_to_index_proximal = -angle_between(vector[0], vector[1])
                index_proximal_to_index_distal = -angle_between(
                    vector[1], vector[2])

                self.hand_calibration['real_joint_pos'].append(
                    np.array(
                        [th_proximal_to_th_inter, th_inter_to_th_distal,
                         palm_to_index_proximal,
                         index_proximal_to_index_distal]))

            for k, v in self.hand_calibration.items():
                print("k:", k)
                if k == 'time_step':
                    self.hand_calibration[k] = np.array(v)
                else:
                    self.hand_calibration[k] = np.stack(v)

            fig, axs = plt.subplots(1, 4)
            fig.suptitle('SIH Hand Calibration')

            hand_calibration_joints = ['th_proximal_to_th_inter', 'th_inter_to_th_distal', 'palm_to_if_proximal', 'if_proximal_to_if_distal']

            for j, joint_name in enumerate(self.robot.manipulator.joint_names):
                if joint_name in hand_calibration_joints:
                    axs[hand_calibration_joints.index(joint_name)].set_title(joint_name)

                    if joint_name.startswith('th'):
                        axs[hand_calibration_joints.index(joint_name)].plot(
                            self.hand_calibration['time_step'][0:600] - start_time_step,
                            self.hand_calibration['simulated_joint_pos'][
                            1000:1600, j], 'r-',
                            label='Simulation')
                        axs[hand_calibration_joints.index(joint_name)].plot(
                            self.hand_calibration['time_step'][0:600] - start_time_step,
                            self.hand_calibration['real_joint_pos'][1000:1600, hand_calibration_joints.index(joint_name)], 'g-', label='Real')
                        if joint_name == 'th_proximal_to_th_inter':
                            axs[hand_calibration_joints.index(joint_name)].plot(self.hand_calibration['time_step'][0:600] - start_time_step,
                                       self.hand_calibration['command_joint_pos'][
                                       1000:1600, j],
                                       'b--', label='Command')

                    else:
                        axs[hand_calibration_joints.index(joint_name)].plot(
                            self.hand_calibration['time_step'][0:600] - start_time_step,
                            self.hand_calibration['simulated_joint_pos'][
                            300:900, j], 'r-',
                            label='Simulation')
                        axs[hand_calibration_joints.index(joint_name)].plot(self.hand_calibration['time_step'][0:600] - start_time_step,
                            self.hand_calibration['real_joint_pos'][300:900, hand_calibration_joints.index(joint_name)], 'g-', label='Real')
                        if joint_name == 'palm_to_if_proximal':
                            axs[hand_calibration_joints.index(joint_name)].plot(self.hand_calibration['time_step'][0:600] - start_time_step,
                                       self.hand_calibration['command_joint_pos'][
                                       300:900, j],
                                       'b--', label='Command')

            axs[0].legend(loc="upper right")
            axs[0].set_ylabel('Joint position [rad]')

            #np.save('hand_calibration.npy', self.hand_calibration)

            plt.show()

        #if self.progress_buf[0] > end_time_step:
        #    # Move back to initial pose.
        #    self._calibration_move_to_pose(
        #        start_pos=self.ik_body_pos[0],
        #        start_quat=self.ik_body_quat[0],
        #        end_pos=[0.28, 0.58, 0.3],
        #        end_quat=[0, 0, 0.707, 0.707],
        #        start_ts=end_time_step, end_ts=end_time_step + 250)

        return actions