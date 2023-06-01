"""
Methods that take care of control and sim2real via publishing of the control
signals.
"""
import os
import actionlib
from control_msgs.msg import \
    FollowJointTrajectoryAction,\
    FollowJointTrajectoryGoal
from isaacgym import gymtorch
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.dexterity import dexterity_control as ctrl
import math
import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from trajectory_msgs.msg import JointTrajectoryPoint
import tf
import time
import torch
from typing import *
import contextlib


class FPSTimer:
    def __init__(self):
        self._fps = None
        self._start_time = None

    @contextlib.contextmanager
    def fps(self, name):
        if self._fps is not None:
            self._fps = 0.8 * self._fps + 0.2 * (1 / (time.time() - self._start_time))
        elif self._start_time is not None:
            self._fps = (1 / (time.time() - self._start_time))
        self._start_time = time.time()

        print(f"{name} FPS:", self._fps)

        yield


timer = FPSTimer()


class ROSJointStateSubscriber:
    def __init__(
            self,
            joint_state_topic: str,
            joint_mapping: Dict[str, List[float]],
            verbose: bool = False
    ) -> None:
        super().__init__()
        self.joint_state_sub = rospy.Subscriber(joint_state_topic, JointState,
                                                self._update_joint_state)
        self.joint_mapping = joint_mapping
        self.verbose = verbose
        self.joint_state = JointState()

    def _update_joint_state(self, data: JointState) -> None:
        # Update joint-state if message belongs to my joints.
        if set(data.name) == set(self.joint_mapping.keys()):
            self.joint_state = data

            if self.verbose:
                print(f"Received joint_state: \n {data}")

    def get_joint_position(self) -> np.array:
        position = []
        for k in self.joint_mapping.keys():
            joint_idx = self.joint_state.name.index(k)
            position.append(self.joint_state.position[joint_idx])
        return np.array(position)


class ROSJointStateInterface(ROSJointStateSubscriber):
    def __init__(
            self,
            joint_state_topic: str,
            joint_target_topic: str,
            joint_mapping: Dict[str, List[float]],
            verbose: bool = False
    ) -> None:
        super().__init__(joint_state_topic, joint_mapping, verbose)
        self.joint_target_pub = rospy.Publisher(joint_target_topic, JointState,
                                                queue_size=1)

    def set_joint_target(self, position_target: np.array) -> None:
        assert len(list(self.joint_mapping.keys())) == position_target.shape[0], \
            f"The number of received joint target values " \
            f"{position_target.shape[0]} does not match the number of joint " \
            f"names {len(list(self.joint_mapping.keys()))}."

        command_target = self.map_position_to_command_target(position_target)

        joint_state_msg = JointState()
        for joint_name, joint_target in zip(self.joint_mapping.keys(), command_target):
            joint_state_msg.name.append(joint_name)
            joint_state_msg.position.append(joint_target)

        if self.verbose:
            print(f"Publishing joint targets: \n {joint_state_msg}")

        self.joint_target_pub.publish(joint_state_msg)

    def map_position_to_command_target(self, position_target: np.array) -> np.array:
        command_target = np.zeros_like(position_target)
        for joint_idx, polycoef in enumerate(self.joint_mapping.values()):
            for exp, coef in enumerate(polycoef):
                command_target[joint_idx] += coef * math.pow(
                    position_target[joint_idx], exp)
        return command_target


class ROSJointTrajectoryInterface(ROSJointStateSubscriber):
    def __init__(
            self,
            joint_state_topic: str,
            joint_target_topic: str,
            joint_mapping: Dict[str, List[float]],
            execution_time: float = 0.25,
            max_joint_deviation: float = 0.15,
            verbose: bool = False,
    ) -> None:
        super().__init__(joint_state_topic, joint_mapping, verbose)
        self.joint_trajectory_client = actionlib.SimpleActionClient(
            joint_target_topic, FollowJointTrajectoryAction)
        self.nsecs = int(1e9 * execution_time)
        self.max_joint_deviation = max_joint_deviation

    def set_joint_target(self, position_target: np.array) -> None:
        #command_target = self.map_position_to_command_target(position_target)
        command_target = position_target

        joint_deviation = command_target - self.get_joint_position()

        assert joint_deviation.max() < self.max_joint_deviation, \
            f"Found a deviation between the current joint position and " \
            f"target to be published of {joint_deviation.max()}, which " \
            f"violates the deviation limit of {self.max_joint_deviation}."

        # Cancel previous goal
        self.joint_trajectory_client.cancel_goal()

        # Define new trajectory point
        position_target_point = JointTrajectoryPoint()
        position_target_point.positions = command_target
        position_target_point.time_from_start.nsecs = self.nsecs

        # Create and publish trajectory goal
        joint_trajectory_msg = FollowJointTrajectoryGoal()
        joint_trajectory_msg.goal_time_tolerance = rospy.Time(0.1)
        joint_trajectory_msg.trajectory.header = Header()
        joint_trajectory_msg.trajectory.header.stamp = rospy.Time.now()
        joint_trajectory_msg.trajectory.joint_names = list(self.joint_mapping.keys())
        joint_trajectory_msg.trajectory.points.append(position_target_point)

        self.joint_trajectory_client.send_goal(joint_trajectory_msg)
        #self.joint_trajectory_client.wait_for_result(rospy.Duration.from_sec(1.0))

        self.check_trajectory_state()

    def map_position_to_command_target(self, position_target: np.array) -> np.array:
        command_target = np.zeros_like(position_target)
        for joint_idx, polycoef in enumerate(self.joint_mapping.values()):
            for exp, coef in enumerate(polycoef):
                command_target[joint_idx] += coef * math.pow(
                    position_target[joint_idx], exp)
        return command_target

    def check_trajectory_state(self):
        trajectory_state = self.joint_trajectory_client.get_state()

        if trajectory_state == 4:
            assert False, "Trajectory goal was aborted."
        elif trajectory_state == 5:
            assert False, "Trajectory goal was rejected."


class DexterityBaseControl:
    def _get_ros_interface(self):
        os.environ["ROS_MASTER_URI"] = self.cfg_base["ros_master_uri"]
        rospy.init_node('dexterity_ros_interface')
        self.ros_arm_interface = ROSJointTrajectoryInterface(
            joint_state_topic=self.robot.arm.ros_state_topic,
            joint_target_topic="/scaled_pos_joint_traj_controller/follow_joint_trajectory",
            joint_mapping=self.robot.arm.ros_joint_mapping
        )
        self.ros_manipulator_interface = ROSJointStateInterface(
            joint_state_topic=self.robot.manipulator.ros_state_topic,
            joint_target_topic=self.robot.manipulator.ros_target_topic,
            joint_mapping=self.robot.manipulator.ros_joint_mapping
        )  

        self.tf_sub = tf.TransformListener()
        import time
        time.sleep(1)

        if self.cfg["calibrate"]:
            #self.ros_image_subscriber = ROSImageSubscriber(
            #    'l515/camera/color/image_rect_color')
            self.ros_image_subscriber = ROSCompressedImageSubscriber('l515/camera/color/image_rect_color/compressed')

    def generate_ctrl_signals(self):
        """Get Jacobian. Set robot DOF position targets or DOF torques."""

        # Input targets of actuated DoFs into DoF targets
        self.ctrl_target_dof_pos[:, self.residual_actuated_dof_indices] = \
            self.ctrl_target_residual_actuated_dof_pos

        # Scale DoF targets from [-1, 1] to the DoF limits of the robot
        self.ctrl_target_dof_pos = scale(
            self.ctrl_target_dof_pos, self.robot_dof_lower_limits,
            self.robot_dof_upper_limits)

        # Enforce equalities specified in XML model
        ctrl_target_dof_pos = self.robot.model.joint_equality(
            self.ctrl_target_dof_pos)
        self.ctrl_target_residual_dof_pos = ctrl_target_dof_pos[
                                       :, self.ik_body_dof_count:]

        # Get desired Jacobian
        if self.cfg_ctrl['jacobian_type'] == 'geometric':
            self.ik_body_jacobian_tf = self.ik_body_jacobian
        elif self.cfg_ctrl['jacobian_type'] == 'analytic':
            raise NotImplementedError
            self.ik_body_jacobian_tf = ctrl.get_analytic_jacobian(
                fingertip_quat=self.ik_body_quat,
                fingertip_jacobian=self.ik_body_jacobian,
                num_envs=self.num_envs,
                device=self.device)

        # Set PD joint pos target or joint torque
        if self.cfg_ctrl['motor_ctrl_mode'] == 'gym':
            self._set_dof_pos_target()
        elif self.cfg_ctrl['motor_ctrl_mode'] == 'manual':
            self._set_dof_torque()

    @timer.fps('publish_targets')
    def _publish_targets(self, arm_targets, manipulator_targets) -> None:

        if self.viewer is None:
            self.gym.sync_frame_time(self.sim)

        if rospy.is_shutdown():
            import sys
            sys.exit('ROSPy has been stopped.')
        #print("manipulator_targets:", manipulator_targets)

        self.ros_arm_interface.set_joint_target(arm_targets)
        self.ros_manipulator_interface.set_joint_target(manipulator_targets)
        #print("arm_joint_position:", self.ros_arm_interface.get_joint_position())
        #print("simulated_arm_joint_position:", self.dof_pos[0, 0:self.robot.arm.num_joints])
        #print("self.flange_pos:", self.flange_pos)
        #print("robot flange pos:", self.ros_arm_interface.get_transform('base_link', 'flange')[0])

    def _set_dof_pos_target(self):
        """Set robot DoF position target to move ik_body towards target pose."""

        self.ctrl_target_dof_pos = ctrl.compute_dof_pos_target(
            cfg_ctrl=self.cfg_ctrl,
            ik_body_dof_pos=self.ik_body_dof_pos,
            ik_body_pos=self.ik_body_pos,
            ik_body_quat=self.ik_body_quat,
            ik_body_jacobian=self.ik_body_jacobian_tf,
            ctrl_target_ik_body_pos=self.ctrl_target_ik_body_pos,
            ctrl_target_ik_body_quat=self.ctrl_target_ik_body_quat,
            ctrl_target_residual_dof_pos=self.ctrl_target_residual_dof_pos,
            device=self.device)  # (shape: [num_envs, robot_dof_count])

        # ik_body DoF target can be set in base config, i.e., to calibrate the
        # dynamics of the robot arm.
        if self.cfg_base.debug.override_ik_body_dof_target:
            assert len(self.cfg_base.debug.ik_body_dof_targets) > 0, \
                "override_ik_body_dof_target is True, but no " \
                "ik_body_dof_targets were given."
            target_dof_pos, start_time = [], [0, ]
            for dof_pos, duration in self.cfg_base.debug.ik_body_dof_targets:
                target_dof_pos.append(dof_pos)
                start_time.append(start_time[-1] + duration)
            start_time.pop(-1)

            current_dof_pos_idx = len(
                self.cfg_base.debug.ik_body_dof_targets) - 1
            elapsed_time_steps = self.progress_buf[0]

            while start_time[current_dof_pos_idx] > elapsed_time_steps:
                current_dof_pos_idx -= 1

            self.ctrl_target_dof_pos[:, :self.ik_body_dof_count] = torch.Tensor(
                target_dof_pos[current_dof_pos_idx]).unsqueeze(0).repeat(
                self.num_envs, 1).to(self.device)

        if self.cfg_base.ros_activate:
            arm_targets = self.ctrl_target_dof_pos[
                0, :self.robot.arm.num_joints].cpu().numpy()

            manipulator_targets = self.ctrl_target_dof_pos[
                0, self.residual_actuated_dof_indices].cpu().numpy()
            self._publish_targets(arm_targets, manipulator_targets)

        # This includes non-robot DoFs, such as from a door or drill
        ctrl_target_dof_pos = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        ctrl_target_dof_pos[:, :self.robot_dof_count] = self.ctrl_target_dof_pos

        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(ctrl_target_dof_pos),
            gymtorch.unwrap_tensor(self.robot_actor_ids_sim),
            len(self.robot_actor_ids_sim))

    def _set_dof_torque(self):
        """Set robot DOF torque to move arm end effector towards target pose."""
        # Includes only actuated DOFs
        self.dof_torque[:, 0:self.robot_dof_count] = ctrl.compute_dof_torque(
            cfg_ctrl=self.cfg_ctrl,
            ik_body_dof_pos=self.dof_pos[:, 0:self.ik_body_dof_count],
            ik_body_dof_vel=self.dof_vel[:, 0:self.ik_body_dof_count],
            ik_body_pos=self.ik_body_pos,
            ik_body_quat=self.ik_body_quat,
            ik_body_linvel=self.ik_body_linvel,
            ik_body_angvel=self.ik_body_angvel,
            ik_body_jacobian=self.ik_body_jacobian_tf,
            ik_body_mass_matrix=self.ik_body_mass_matrix,
            residual_dof_pos=self.dof_pos[:, self.ik_body_dof_count:self.robot_dof_count],
            residual_dof_vel=self.dof_vel[:, self.ik_body_dof_count:self.robot_dof_count],
            ctrl_target_ik_body_pos=self.ctrl_target_ik_body_pos,
            ctrl_target_ik_body_quat=self.ctrl_target_ik_body_quat,
            ctrl_target_residual_dof_pos=self.ctrl_target_residual_dof_pos,
            device=self.device)

        self.gym.set_dof_actuation_force_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_torque),
            gymtorch.unwrap_tensor(self.robot_actor_ids_sim),
            len(self.robot_actor_ids_sim))
