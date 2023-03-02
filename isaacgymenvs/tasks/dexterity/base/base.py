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

"""Dexterity: base class.

Inherits Gym's VecTask class and abstract base class. Inherited by environment classes. Not directly executed.

Configuration defined in DexterityBase.yaml.
"""


import hydra
import math
import os
import sys

from isaacgym import gymapi, gymtorch, torch_utils
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask
import isaacgymenvs.tasks.dexterity.dexterity_control as ctrl
from isaacgymenvs.tasks.dexterity.base.schema_class_base import DexterityABCBase
from isaacgymenvs.tasks.dexterity.base.schema_config_base import DexteritySchemaConfigBase
from isaacgymenvs.tasks.dexterity.base.base_cameras import DexterityBaseCameras
from isaacgymenvs.tasks.dexterity.base.base_logger import DexterityBaseLogger
from isaacgymenvs.tasks.dexterity.base.base_properties import DexterityBaseProperties
from isaacgymenvs.tasks.dexterity.base.base_visualizations import DexterityBaseVisualizations
from isaacgymenvs.tasks.dexterity.xml.robot import DexterityRobot
from isaacgymenvs.utils import torch_jit_utils


class DexterityBase(VecTask, DexterityABCBase, DexterityBaseCameras,
                    DexterityBaseLogger, DexterityBaseProperties,
                    DexterityBaseVisualizations):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless,
                 virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize VecTask superclass."""

        self.cfg = cfg
        self.cfg['headless'] = headless

        self._get_base_yaml_params()
        cfg = self._get_robot_model(cfg)
        cfg = self._update_observation_num(cfg)

        if self.cfg_base.mode.export_scene:
            sim_device = 'cpu'

        self.arm_eef_body_id_env = None  # set in subclass
        self.tracker_body_id_env = None  # set in subclass

        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)  # create_sim() is called here

    def _get_base_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='dexterity_schema_config_base',
                 node=DexteritySchemaConfigBase)

        config_path = 'task/DexterityBase.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_base = hydra.compose(config_name=config_path)
        self.cfg_base = self.cfg_base['task']  # strip superfluous nesting

        asset_info_path = '../../assets/factory/yaml/factory_asset_info_franka_table.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_franka_table = hydra.compose(config_name=asset_info_path)
        self.asset_info_franka_table = self.asset_info_franka_table['']['']['']['']['']['']['assets']['factory']['yaml']  # strip superfluous nesting

    def _get_robot_model(self, cfg):
        """Build robot model from files specified in config and update the
        number of actions to match the model."""
        model_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '../..', '..', '..',
                         'assets', 'dexterity'))
        self.robot = DexterityRobot(model_root, self.cfg_base.env.robot)

        self.ik_body_name = self.cfg_base.ctrl.ik_body
        self.last_ik_dof_name, self.last_ik_body_name = \
            self.robot.model.get_parent_names(self.ik_body_name)
        robot_dof_names = self.robot.model.joint_names
        last_ik_dof_idx = robot_dof_names.index(self.last_ik_dof_name)
        ik_body_dof_names = robot_dof_names[0:last_ik_dof_idx + 1]
        residual_dof_names = robot_dof_names[last_ik_dof_idx + 1:]
        self.ik_body_dof_count = len(ik_body_dof_names)
        actuated_dof_names = self.robot.model.actuated_joint_names
        self.residual_actuated_dof_names = actuated_dof_names[
                                           self.ik_body_dof_count:]
        self.residual_actuator_count = len(self.residual_actuated_dof_names)

        # num_actions = 6 pose actions (3 delta_pos and 3 delta_rot) + number of
        # actuators after the ik_body (i.e. actuators controlling fingers of a
        # manipulator)
        cfg["env"]["numActions"] = 6 + self.residual_actuator_count
        return cfg

    def _update_observation_num(self, cfg):
        self.keypoint_dict = self.robot.model.get_keypoints()

        num_observations = 0
        for observation in cfg['env']['observations']:
            # Infer general type of observation (e.g. position or quaternion)
            if observation.endswith('_pos') or \
                    observation.endswith('_pos_demo'):
                obs_dim = 3
            elif observation.endswith('_quat') or \
                    observation.endswith('_quat_demo'):
                obs_dim = 4
            elif observation.endswith('_linvel'):
                obs_dim = 3
            elif observation.endswith('_angvel'):
                obs_dim = 3
            # Previous action can be included in the observation. obs_dim is
            # then the dimension of the action-space of the robot.
            elif observation == 'previous_action':
                obs_dim = cfg["env"]["numActions"]
            else:
                # Assume other observations must be camera sensors and can
                # therefore be skipped
                continue

            # Adjust dimensionality for keypoint group observations that can
            # include multiple bodies
            for keypoint_group_name in self.keypoint_dict.keys():
                if observation.startswith(keypoint_group_name):
                    # Multiply by number of keypoints in that group
                    obs_dim *= len(
                        self.keypoint_dict[keypoint_group_name].keys())

            # Add dimensionality of this observation to the total number
            num_observations += obs_dim
        cfg["env"]["numObservations"] = num_observations
        return cfg

    def create_sim(self):
        """Set sim and PhysX params. Create sim object, ground plane, and envs."""
        # Set time-step size and substeps.
        self.sim_params.dt = self.cfg_base.sim.dt
        self.sim_params.substeps = self.cfg_base.sim.num_substeps

        # Use GPU-pipeline if the simulation device is a GPU.
        self.sim_params.use_gpu_pipeline = self.device.startswith("cuda")

        # Set orientation and gravity vector.
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -self.cfg_base.sim.gravity_mag

        # Set PhysX parameters.
        self.sim_params.physx.use_gpu = self.device.startswith("cuda")
        self.sim_params.physx.solver_type = 1  # default = 1 (Temporal Gauss-Seidel)
        self.sim_params.physx.num_subscenes = 4  # for CPU PhysX only
        self.sim_params.physx.num_threads = 4  # for CPU PhysX only
        self.sim_params.physx.num_position_iterations = self.cfg_base.sim.num_pos_iters
        self.sim_params.physx.num_velocity_iterations = self.cfg_base.sim.num_vel_iters
        self.sim_params.physx.rest_offset = 0.0  # default = 0.001
        self.sim_params.physx.contact_offset = 0.002  # default = 0.02
        self.sim_params.physx.bounce_threshold_velocity = 0.2  # default = 0.01
        self.sim_params.physx.max_depenetration_velocity = 100.0  # default = 100.0
        self.sim_params.physx.friction_offset_threshold = 0.04  # default = 0.04
        self.sim_params.physx.friction_correlation_distance = 0.025  # default = 0.025
        self.sim_params.physx.max_gpu_contact_pairs = 8 * 1024 ** 2  # default = 1024^2
        self.sim_params.physx.default_buffer_size_multiplier = 2  # default = 1

        self.sim = super().create_sim(compute_device=self.device_id,
                                      graphics_device=self.graphics_device_id,
                                      physics_engine=self.physics_engine,
                                      sim_params=self.sim_params)
        self._create_ground_plane()
        self.create_envs()  # defined in subclass

    def _create_ground_plane(self):
        """Set ground plane params. Add plane."""

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0.0  # default = 0.0
        plane_params.static_friction = 1.0  # default = 1.0
        plane_params.dynamic_friction = 1.0  # default = 1.0
        plane_params.restitution = 0.0  # default = 0.0

        self.gym.add_ground(self.sim, plane_params)

    def import_robot_assets(self):
        """Set robot and table asset options. Import assets."""
        # Define robot and table asset options
        robot_options = gymapi.AssetOptions()
        robot_options.fix_base_link = True
        robot_options.collapse_fixed_joints = False
        robot_options.thickness = 0.02  # default = 0.02
        robot_options.density = 1000.0  # default = 1000.0
        robot_options.angular_damping = 0.0  # default = 0.0
        robot_options.armature = 0.01  # default = 0.0
        robot_options.use_physx_armature = True
        if self.cfg_base.sim.add_damping:
            robot_options.linear_damping = 1.0  # default = 0.0; increased to improve stability
            robot_options.max_linear_velocity = 1.0  # default = 1000.0; reduced to prevent CUDA errors
            robot_options.angular_damping = 5.0  # default = 0.5; increased to improve stability
            robot_options.max_angular_velocity = 2 * math.pi  # default = 64.0; reduced to prevent CUDA errors
        else:
            robot_options.linear_damping = 0.0  # default = 0.0
            robot_options.max_linear_velocity = 1000.0  # default = 1000.0
            robot_options.angular_damping = 0.5  # default = 0.5
            robot_options.max_angular_velocity = 64.0  # default = 64.0
        robot_options.disable_gravity = True
        robot_options.enable_gyroscopic_forces = True
        robot_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        robot_options.use_mesh_materials = False
        if self.cfg_base.mode.export_scene:
            robot_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        if self.cfg_base.sim.override_robot_com_and_inertia:
            robot_options.override_com = True
            robot_options.override_inertia = True

        table_options = gymapi.AssetOptions()
        table_options.flip_visual_attachments = False  # default = False
        table_options.fix_base_link = True
        table_options.thickness = 0.0  # default = 0.02
        table_options.density = 1000.0  # default = 1000.0
        table_options.armature = 0.0  # default = 0.0
        table_options.use_physx_armature = True
        table_options.linear_damping = 0.0  # default = 0.0
        table_options.max_linear_velocity = 1000.0  # default = 1000.0
        table_options.angular_damping = 0.0  # default = 0.5
        table_options.max_angular_velocity = 64.0  # default = 64.0
        table_options.disable_gravity = False
        table_options.enable_gyroscopic_forces = True
        table_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        table_options.use_mesh_materials = False
        if self.cfg_base.mode.export_scene:
            table_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        # Create assets for robot and table
        robot_asset = self.robot.as_asset(self.gym, self.sim, robot_options)
        table_asset = self.gym.create_box(
            self.sim, self.asset_info_franka_table.table_depth,
            self.asset_info_franka_table.table_width,
            self.cfg_base.env.table_height, table_options)

        return robot_asset, table_asset

    def create_base_actors(
            self, env_ptr: int, i: int, actor_count: int, robot_asset,
            robot_pose, table_asset, table_pose) -> int:
        """Create common actors (robot, table, cameras)."""
        if not hasattr(self, "robot_actor_ids_sim"):
            self.robot_actor_ids_sim = []  # within-sim indices
            self.robot_handles = []
        if not hasattr(self, "table_actor_ids_sim"):
            self.table_actor_ids_sim = []  # within-sim indices
            self.table_handles = []

        # Create robot actor
        self._create_robot_actor(env_ptr, i, actor_count, robot_asset,
                                 robot_pose)
        actor_count += 1
        if i == 0:
            self.robot_actor_id_env = self.gym.find_actor_index(
                env_ptr, 'robot', gymapi.DOMAIN_ENV)

        # Create table actor
        if self.cfg_base.env.has_table:
            self._create_table_actor(env_ptr, i, actor_count, table_asset,
                                     table_pose)
            actor_count += 1

        # Create camera actors
        if "cameras" in self.cfg_env.keys():
            self.create_camera_actors(env_ptr, i, actor_count)
            for camera_name, dexterity_camera in self._camera_dict.items():
                if dexterity_camera.add_camera_actor:
                    actor_count += 1
                    # Store camera actor env_ids to set the root pose of
                    # attached cameras later
                    if i == 0:
                        setattr(self, f"{camera_name}_actor_id_env",
                                self.gym.find_actor_index(
                                    env_ptr, camera_name, gymapi.DOMAIN_ENV))
        return actor_count

    def _create_robot_actor(self, env_ptr: int, i: int, actor_count: int,
                            robot_asset, robot_pose) -> None:
        # collision_filter=-1 to use asset collision filters in XML model
        robot_handle = self.gym.create_actor(
            env_ptr, robot_asset, robot_pose, 'robot', i, -1, 1)
        self.robot_actor_ids_sim.append(actor_count)
        self.robot_handles.append(robot_handle)
        # Enable force sensors for robot
        self.gym.enable_actor_dof_force_sensors(env_ptr, robot_handle)

    def _create_table_actor(self, env_ptr: int, i: int, actor_count: int,
                            table_asset, table_pose):
        # Segmentation id set to 0. Table is viewed as background.
        table_handle = self.gym.create_actor(
            env_ptr, table_asset, table_pose, 'table', i, 0, 0)
        self.table_actor_ids_sim.append(actor_count)

        # Set table shape properties
        table_shape_props = self.gym.get_actor_rigid_shape_properties(
            env_ptr, table_handle)
        table_shape_props[0].friction = self.cfg_base.env.table_friction
        table_shape_props[0].rolling_friction = 0.0  # default = 0.0
        table_shape_props[0].torsion_friction = 0.0  # default = 0.0
        table_shape_props[0].restitution = 0.0  # default = 0.0
        table_shape_props[0].compliance = 0.0  # default = 0.0
        table_shape_props[0].thickness = 0.0  # default = 0.0
        self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle,
                                                  table_shape_props)
        self.table_handles.append(table_handle)

    def acquire_base_tensors(self):
        """Acquire and wrap tensors. Create views."""

        # Acquire general simulation tensors
        _root_state = self.gym.acquire_actor_root_state_tensor(self.sim)  # shape = (num_envs * num_actors, 13)
        _body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)  # shape = (num_envs * num_bodies, 13)
        _dof_state = self.gym.acquire_dof_state_tensor(self.sim)  # shape = (num_envs * num_dofs, 2)
        _dof_force = self.gym.acquire_dof_force_tensor(self.sim)  # shape = (num_envs * num_dofs, 1)
        _contact_force = self.gym.acquire_net_contact_force_tensor(self.sim)  # shape = (num_envs * num_bodies, 3)
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, 'robot')  # shape = (num envs, num_bodies - 1, 6, num_dofs)  -1 because the base is fixed
        _mass_matrix = self.gym.acquire_mass_matrix_tensor(self.sim, 'robot')  # shape = (num_envs, num_dofs, num_dofs)

        self.refresh_base_tensors()

        self.root_state = gymtorch.wrap_tensor(_root_state)
        self.body_state = gymtorch.wrap_tensor(_body_state)
        self.dof_state = gymtorch.wrap_tensor(_dof_state)
        self.dof_force = gymtorch.wrap_tensor(_dof_force)
        self.contact_force = gymtorch.wrap_tensor(_contact_force)
        self.jacobian = gymtorch.wrap_tensor(_jacobian)
        self.mass_matrix = gymtorch.wrap_tensor(_mass_matrix)

        self.root_pos = self.root_state.view(self.num_envs, self.num_actors, 13)[..., 0:3]
        self.root_quat = self.root_state.view(self.num_envs, self.num_actors, 13)[..., 3:7]
        self.root_linvel = self.root_state.view(self.num_envs, self.num_actors, 13)[..., 7:10]
        self.root_angvel = self.root_state.view(self.num_envs, self.num_actors, 13)[..., 10:13]
        self.body_pos = self.body_state.view(self.num_envs, self.num_bodies, 13)[..., 0:3]
        self.body_quat = self.body_state.view(self.num_envs, self.num_bodies, 13)[..., 3:7]
        self.body_linvel = self.body_state.view(self.num_envs, self.num_bodies, 13)[..., 7:10]
        self.body_angvel = self.body_state.view(self.num_envs, self.num_bodies, 13)[..., 10:13]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.dof_force_view = self.dof_force.view(self.num_envs, self.num_dofs, 1)[..., 0]
        self.contact_force = self.contact_force.view(self.num_envs, self.num_bodies, 3)[..., 0:3]

        self.robot_dof_count = self.robot.model.get_asset_dof_count()
        self.robot_actuator_count = self.robot.model.get_asset_actuator_count()

        # Initialize torque or position targets for all DoFs
        self.dof_torque = torch.zeros(
            (self.num_envs, self.num_dofs), device=self.device)
        self.ctrl_target_dof_pos = torch.zeros(
            (self.num_envs, self.robot_dof_count), device=self.device)

        self.prev_actions = torch.zeros((self.num_envs, self.num_actions), device=self.device)

        # Initialize tensors used for control
        self.ik_body_id_env = self.gym.find_actor_rigid_body_index(
            self.env_ptrs[0], self.robot_handles[0], self.cfg_base.ctrl.ik_body,
            gymapi.DOMAIN_ENV)

        actuated_dof_names = [
            self.robot.model.get_asset_actuator_joint_name(i) for i
            in range(self.robot.model.get_asset_actuator_count())]
        actuated_dof_indices = [
            self.robot.model.find_asset_dof_index(name) for name in
            actuated_dof_names]

        self.residual_actuated_dof_indices = actuated_dof_indices[
                                             self.ik_body_dof_count:]

        # Create views of pose of the body positioned with inverse kinematics
        self.ik_body_pos = self.body_pos[:, self.ik_body_id_env, 0:3]
        self.ik_body_quat = self.body_quat[:, self.ik_body_id_env, 0:4]
        self.ik_body_linvel = self.body_linvel[:, self.ik_body_id_env, 0:3]
        self.ik_body_angvel = self.body_angvel[:, self.ik_body_id_env, 0:3]
        self.ik_body_dof_pos = self.dof_pos[:, 0:self.ik_body_dof_count]

        self.ik_body_jacobian = self.jacobian[:, self.ik_body_dof_count - 1, :,
                                0:self.ik_body_dof_count]  # minus 1 because base is fixed
        self.ik_body_mass_matrix = self.mass_matrix[:, 0:self.ik_body_dof_count,
                                   0:self.ik_body_dof_count]

        # Initialize pose targets for inverse kinematics
        # Initialize pose targets for inverse kinematics
        self.base_ctrl_target_ik_body_pos = torch.tensor(
            [[0., 0., 0.5]], device=self.device).repeat(self.num_envs, 1)
        self.base_ctrl_target_ik_body_quat = torch.tensor(
            [[0., 0., 0., 1.]], device=self.device).repeat(self.num_envs, 1)
        self.ctrl_target_ik_body_pos = self.base_ctrl_target_ik_body_pos.clone()
        self.ctrl_target_ik_body_quat = self.base_ctrl_target_ik_body_quat.clone()

        # Initialize DoF targets for residual DoFs
        self.ctrl_target_residual_actuated_dof_pos = torch.zeros(
            (self.num_envs, self.residual_actuator_count), device=self.device)

        # TODO: Make this ugly code cleaner
        # Initialize keypoint specs with body ids
        keypoint_specs = {}
        for group_name, group in self.keypoint_dict.items():
            keypoint_specs[group_name] = ([], [], [])
            for keypoint_name, keypoint in group.items():
                body_id_env = self.gym.find_actor_rigid_body_index(
                    self.env_ptrs[0], self.robot_handles[0],
                    keypoint['body_name'], gymapi.DOMAIN_ENV)
                keypoint_specs[group_name][0].append(body_id_env)
                keypoint_specs[group_name][1].append(keypoint['pos'])
                keypoint_specs[group_name][2].append(keypoint['quat'])

        self.keypoint_specs = []
        for group_name in keypoint_specs.keys():
            self.keypoint_specs.append((
                group_name,
                keypoint_specs[group_name][0],
                torch.Tensor(np.stack(keypoint_specs[group_name][1])).unsqueeze(0).repeat(
                    self.num_envs, 1, 1).to(self.device),
                torch.Tensor(np.stack(keypoint_specs[group_name][2])).unsqueeze(0).repeat(
                    self.num_envs, 1, 1).to(self.device)))

    def refresh_base_tensors(self):
        """Refresh tensors."""
        # NOTE: Tensor refresh functions should be called once per step, before setters.

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        self.refresh_keypoint_tensors()

    def refresh_keypoint_tensors(self) -> None:
        if not hasattr(self, 'keypoint_specs'):
            return
        # Iterate through all keypoint groups found in robot model
        for group_name, body_ids_env, pos, quat in self.keypoint_specs:
            # If this group is used in the observations or rewards
            if any(observation.startswith(group_name)
                   for observation in self.cfg['env']['observations']) or \
                    any(reward_term.startswith(group_name)
                        for reward_term in self.cfg['rl']['reward']):

                keypoint_group_quat, keypoint_group_pos = \
                    torch_jit_utils.tf_combine(
                        self.body_quat[:, body_ids_env],
                        self.body_pos[:, body_ids_env],
                        quat,
                        pos)

                setattr(self, group_name + '_pos', keypoint_group_pos)
                setattr(self, group_name + '_quat', keypoint_group_quat)

    def pre_physics_step(self, actions):
        """Reset environments. Apply actions from policy. Simulation step called
        after this method."""

        overwrite_actions = False
        if overwrite_actions:
            actions = torch.zeros_like(actions)
            import isaacgymenvs.tasks.dexterity.dexterity_control as ctrl

            new_target_pos = torch.tensor(
                [[0, 0, 0.6]], device=self.device).repeat(self.num_envs, 1)
            # IsaacGym convention is [x, y, z, w]
            new_target_quat = torch.tensor(
                [[0, 0, 0, 1]], device=self.device).repeat(self.num_envs, 1)

            # 135 deg around x
            # new_target_quat = torch.tensor(
            #    [[-0.924, 0, 0, 0.383]], device=self.device).repeat(
            #    self.num_envs, 1)

            # new_target_quat = torch.tensor(
            #    [[-0.5, 0.5, -0.5, 0.5]], device=self.device).repeat(
            #    self.num_envs, 1)

            if self.cfg_base.ctrl.add_pose_actions_to == 'pose':
                current_pos = self.ik_body_pos
                current_quat = self.ik_body_quat
            elif self.cfg_base.ctrl.add_pose_actions_to == 'target':
                current_pos = self.ctrl_target_ik_body_pos
                current_quat = self.ctrl_target_ik_body_quat
            else:
                assert False

            # print("ctrl_target_ik_body_quat:", self.ctrl_target_ik_body_quat)

            pos_error, axis_angle_error = ctrl.get_pose_error(
                current_pos, current_quat, new_target_pos,
                new_target_quat, jacobian_type='geometric',
                rot_error_type='axis_angle')

            # print("pos_error:", pos_error)
            # print("axis_angle_error:", axis_angle_error)

            actions[:, 0:3] = 0.1 * pos_error
            actions[:, 3:6] = 1 * axis_angle_error

            actions = torch.clamp(actions, -1, 1)

            actions[:, 6] = 0.05

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(
            self.device)  # shape = (num_envs, num_actions); values = [-1, 1]

        self._apply_actions_as_ctrl_targets(
            ik_body_pose_actions=self.actions[:, 0:6],
            residual_dof_actions=self.actions[:, 6:],
            do_scale=True)

    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward."""

        self.progress_buf[:] += 1

        # In this policy, episode length is constant
        is_last_step = (self.progress_buf[0] == self.max_episode_length - 1)

        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()
        self.gym.fetch_results(self.sim, True)
        self.compute_observations()
        self.compute_reward()


        if len(self.cfg_base.debug.visualize) > 0 and not self.cfg['headless']:
            self.gym.clear_lines(self.viewer)
            self.draw_visualizations(self.cfg_base.debug.visualize)

        if self.cfg_base.debug.save_videos:
            self.save_videos()

    def compute_observations(self):
        """Compute observations."""
        self._compute_proprioceptive_observations()
        self._compute_visual_observations()

    def compute_reward(self):
        """Detect successes and failures. Update reward and reset buffers."""
        self._update_reset_buf()
        self._update_rew_buf()

    def _compute_proprioceptive_observations(self):
        """Compute observations based on base, env, or task tensors, such as
        joint positions or object poses."""
        obs_tensors = []
        for observation in self.cfg_task.env.observations:
            # Camera observations are computed separately
            if "cameras" in self.cfg_env.keys():
                if observation in self.cfg_env.cameras.keys():
                    continue

            # Flatten body dimension for keypoint observations
            if observation.startswith(tuple(self.keypoint_dict.keys())):
                obs_tensors.append(getattr(self, observation).flatten(1, 2))
            elif observation == "previous_action":
                obs_tensors.append(self.actions)
            else:
                obs_tensors.append(getattr(self, observation))

        self.obs_buf = torch.cat(obs_tensors,
                                 dim=-1)  # shape = (num_envs, num_observations)

    def _compute_visual_observations(self):
        if self._camera_dict:
            self.obs_dict["image"] = self.get_images()

    def parse_controller_spec(self):
        """Parse controller specification into lower-level controller configuration."""

        cfg_ctrl_keys = {'num_envs',
                         'jacobian_type',
                         'residual_prop_gains',
                         'residual_deriv_gains',
                         'ik_method',
                         'ik_prop_gains',
                         'ik_deriv_gains',
                         'do_inertial_comp',}
        self.cfg_ctrl = {cfg_ctrl_key: None for cfg_ctrl_key in cfg_ctrl_keys}

        self.cfg_ctrl['num_envs'] = self.num_envs
        self.cfg_ctrl['jacobian_type'] = self.cfg_base.ctrl.all.jacobian_type

        self.cfg_ctrl['ik_body_actuator_count'] = self.ik_body_dof_count
        self.cfg_ctrl['ik_body_dof_count'] = self.ik_body_dof_count
        self.cfg_ctrl['robot_actuator_count'] = self.robot_actuator_count
        self.cfg_ctrl['robot_dof_count'] = self.robot_dof_count

        self.cfg_ctrl['residual_prop_gains'] = torch.ones(
            (self.num_envs, self.cfg_ctrl['robot_dof_count'] - self.cfg_ctrl['ik_body_dof_count']),
            device=self.device) * self.cfg_base.ctrl.all.residual_prop_gain
        self.cfg_ctrl['residual_deriv_gains'] = torch.ones(
            (self.num_envs, self.cfg_ctrl['robot_dof_count'] - self.cfg_ctrl['ik_body_dof_count']),
            device=self.device) * self.cfg_base.ctrl.all.residual_deriv_gain


        self.cfg_ctrl['residual_stiffness'] = self.cfg_base.ctrl.joint_space_id.residual_stiffness
        self.cfg_ctrl['residual_damping'] = self.cfg_base.ctrl.joint_space_id.residual_damping

        ctrl_type = self.cfg_base.ctrl.ctrl_type
        if ctrl_type == 'gym_default':
            self.cfg_ctrl['motor_ctrl_mode'] = 'gym'
            self.cfg_ctrl['gain_space'] = 'joint'
            self.cfg_ctrl['ik_method'] = self.cfg_base.ctrl.gym_default.ik_method

            self.cfg_ctrl['ik_prop_gains'] = (torch.ones(
                (self.num_envs, self.cfg_ctrl['ik_body_dof_count']),
                device=self.device) *
                self.cfg_base.ctrl.gym_default.ik_prop_gain)
            self.cfg_ctrl['ik_deriv_gains'] = (torch.ones(
                (self.num_envs, self.cfg_ctrl['ik_body_dof_count']),
                device=self.device) *
                self.cfg_base.ctrl.gym_default.ik_deriv_gain)
            self.cfg_ctrl['residual_prop_gains'] = (torch.ones(
                (self.num_envs, self.cfg_ctrl['robot_dof_count'] - self.cfg_ctrl['ik_body_dof_count']),
                device=self.device) *
                self.cfg_base.ctrl.gym_default.residual_prop_gain)
            self.cfg_ctrl['residual_deriv_gains'] = (torch.ones(
                (self.num_envs, self.cfg_ctrl['robot_dof_count'] - self.cfg_ctrl['ik_body_dof_count']),
                device=self.device) *
                self.cfg_base.ctrl.gym_default.residual_deriv_gain)

        elif ctrl_type == 'joint_space_id':
            self.cfg_ctrl['motor_ctrl_mode'] = 'manual'
            self.cfg_ctrl['gain_space'] = 'joint'
            self.cfg_ctrl['ik_method'] = self.cfg_base.ctrl.joint_space_id.ik_method

            self.cfg_ctrl['ik_prop_gains'] = (torch.ones(
                (self.num_envs, self.cfg_ctrl['ik_body_dof_count']),
                device=self.device) *
                self.cfg_base.ctrl.joint_space_id.ik_prop_gain)
            self.cfg_ctrl['ik_deriv_gains'] = (torch.ones(
                (self.num_envs, self.cfg_ctrl['ik_body_dof_count']),
                device=self.device) *
                self.cfg_base.ctrl.joint_space_id.ik_deriv_gain)

            if self.cfg_base.ctrl.joint_space_id.uniform_arm_and_hand_gains:
                hand_ik_dof_count = self.cfg_ctrl['ik_body_dof_count'] - self.robot.arm.num_joints
                self.cfg_ctrl['ik_prop_gains'][:][ 0:hand_ik_dof_count] = \
                    self.cfg_base.ctrl.all.residual_prop_gain
                self.cfg_ctrl['ik_deriv_gains'][:][0:hand_ik_dof_count] = \
                    self.cfg_base.ctrl.all.residual_deriv_gain

            self.cfg_ctrl['do_inertial_comp'] = False  # originally True

        self.robot_dof_lower_limits = []
        self.robot_dof_upper_limits = []
        if self.cfg_ctrl['motor_ctrl_mode'] == 'gym':
            prop_gains = torch.cat((self.cfg_ctrl['ik_prop_gains'],
                                    self.cfg_ctrl['residual_prop_gains']), dim=-1).to('cpu')
            deriv_gains = torch.cat((self.cfg_ctrl['ik_deriv_gains'],
                                     self.cfg_ctrl['residual_deriv_gains']), dim=-1).to('cpu')
            # No tensor API for getting/setting actor DOF props; thus, loop required
            for env_ptr, robot_handle, prop_gain, deriv_gain in zip(self.env_ptrs, self.robot_handles, prop_gains,
                                                                     deriv_gains):
                robot_dof_props = self.gym.get_actor_dof_properties(env_ptr, robot_handle)
                self.robot_dof_lower_limits.append(robot_dof_props['lower'])
                self.robot_dof_upper_limits.append(robot_dof_props['upper'])
                robot_dof_props['driveMode'][:] = gymapi.DOF_MODE_POS
                robot_dof_props['stiffness'] = prop_gain
                robot_dof_props['damping'] = deriv_gain
                self.gym.set_actor_dof_properties(env_ptr, robot_handle, robot_dof_props)

        elif self.cfg_ctrl['motor_ctrl_mode'] == 'manual':
            # zero passive stiffness for ik dofs
            prop_gains = torch.cat(
                (torch.zeros_like(self.cfg_ctrl['ik_prop_gains']),
                 self.cfg_ctrl['residual_stiffness'] *
                 torch.ones_like(self.cfg_ctrl['residual_prop_gains'])),
                dim=-1).to('cpu')
            # zero passive damping for ik_dofs
            deriv_gains = torch.cat(
                (torch.zeros_like(self.cfg_ctrl['ik_deriv_gains']),
                 self.cfg_ctrl['residual_damping'] *
                 torch.ones_like(self.cfg_ctrl['residual_deriv_gains'])),
                dim=-1).to('cpu')

            # No tensor API for getting/setting actor DOF props; thus, loop required
            for env_ptr, robot_handle, prop_gain, deriv_gain in zip(
                    self.env_ptrs, self.robot_handles, prop_gains, deriv_gains):
                robot_dof_props = self.gym.get_actor_dof_properties(env_ptr, robot_handle)
                self.robot_dof_lower_limits.append(robot_dof_props['lower'])
                self.robot_dof_upper_limits.append(robot_dof_props['upper'])
                robot_dof_props['driveMode'][:] = gymapi.DOF_MODE_EFFORT
                robot_dof_props['stiffness'][:] = prop_gain
                robot_dof_props['damping'][:] = deriv_gain
                self.gym.set_actor_dof_properties(env_ptr, robot_handle, robot_dof_props)
        else:
            assert False
        self.robot_dof_lower_limits = to_torch(self.robot_dof_lower_limits,
                                               device=self.device)
        self.robot_dof_upper_limits = to_torch(self.robot_dof_upper_limits,
                                               device=self.device)

        if self.cfg_base.ctrl['neglect_attached_weights']:
            # No tensor API for getting/setting actor rigid body props; thus, loop required
            for env_ptr, robot_handle in zip(self.env_ptrs, self.robot_handles):
                robot_rigid_body_props = \
                    self.gym.get_actor_rigid_body_properties(
                        env_ptr, robot_handle)
                for rigid_body_idx, rigid_body_name in enumerate(
                        self.robot.model.get_asset_rigid_body_names()):

                    if rigid_body_name not in self.robot.arm.body_names:
                        robot_rigid_body_props[rigid_body_idx].mass = 1e-5

                self.gym.set_actor_rigid_body_properties(
                    env_ptr, robot_handle, robot_rigid_body_props,
                    recomputeInertia=True)

    def generate_ctrl_signals(self):
        """Get Jacobian. Set robot DOF position targets or DOF torques."""

        #print("self.ctrl_target_dof_pos:", self.ctrl_target_dof_pos)

        # Input targets of actuated DoFs into DoF targets
        self.ctrl_target_dof_pos[:, self.residual_actuated_dof_indices] = \
            self.ctrl_target_residual_actuated_dof_pos

        #print("self.ctrl_target_dof_pos:", self.ctrl_target_dof_pos)
        # Scale DoF targets from [-1, 1] to the DoF limits of the robot
        self.ctrl_target_dof_pos = scale(
            self.ctrl_target_dof_pos, self.robot_dof_lower_limits,
            self.robot_dof_upper_limits)

        #print("self.ctrl_target_dof_pos:", self.ctrl_target_dof_pos)
        # Enforce equalities specified in XML model
        ctrl_target_dof_pos = self.robot.model.joint_equality(
            self.ctrl_target_dof_pos)
        self.ctrl_target_residual_dof_pos = ctrl_target_dof_pos[
                                       :, self.ik_body_dof_count:]

        #print("ctrl_target_dof_pos:", ctrl_target_dof_pos)

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

    def enable_gravity(self, gravity_mag):
        """Enable gravity."""

        sim_params = self.gym.get_sim_params(self.sim)
        sim_params.gravity.z = -gravity_mag
        self.gym.set_sim_params(self.sim, sim_params)

    def disable_gravity(self):
        """Disable gravity."""

        sim_params = self.gym.get_sim_params(self.sim)
        sim_params.gravity.z = 0.0
        self.gym.set_sim_params(self.sim, sim_params)

    def export_scene(self, label):
        """Export scene to USD."""

        usd_export_options = gymapi.UsdExportOptions()
        usd_export_options.export_physics = False

        usd_exporter = self.gym.create_usd_exporter(usd_export_options)
        self.gym.export_usd_sim(usd_exporter, self.sim, label)
        sys.exit()

    def extract_poses(self):
        """Extract poses of all bodies."""

        if not hasattr(self, 'export_pos'):
            self.export_pos = []
            self.export_rot = []
            self.frame_count = 0

        pos = self.body_pos
        rot = self.body_quat

        self.export_pos.append(pos.cpu().numpy().copy())
        self.export_rot.append(rot.cpu().numpy().copy())
        self.frame_count += 1

        if len(self.export_pos) == self.max_episode_length:
            output_dir = self.__class__.__name__
            save_dir = os.path.join('usd', output_dir)
            os.makedirs(output_dir, exist_ok=True)

            print(f'Exporting poses to {output_dir}...')
            np.save(os.path.join(save_dir, 'body_position.npy'), np.array(self.export_pos))
            np.save(os.path.join(save_dir, 'body_rotation.npy'), np.array(self.export_rot))
            print('Export completed.')
            sys.exit()

    def _reset_robot(self, env_ids, apply_reset: bool = True) -> None:
        """Reset DOF states and DOF targets of robot.

            Args:
                apply_reset: (bool) Whether to set the DoF state tensor in this
                method. Setting the DoF state multiple times causes problems.
                Hence, environments that must also reset other DoFs should only
                call self.gym.set_dof_state_tensor_indexed once.
        """

        self.dof_pos[env_ids, :self.robot_dof_count] = torch.tensor(
            self.robot.model.initial_dof_pos, device=self.device
        ).unsqueeze(0).repeat(
            (len(env_ids), 1))  # shape = (len(env_ids), num_dofs)
        self.dof_vel[env_ids, :self.robot_dof_count] = 0.0


        # Reset control targets
        self.ctrl_target_ik_body_pos[env_ids] = \
            self.base_ctrl_target_ik_body_pos[env_ids].clone()
        self.ctrl_target_ik_body_quat[env_ids] = \
            self.base_ctrl_target_ik_body_quat[env_ids].clone()
        self.ctrl_target_dof_pos[env_ids] = \
            self.dof_pos[env_ids, :self.robot_dof_count]
        self.ctrl_target_residual_actuated_dof_pos[env_ids, :] = 0.

        if apply_reset:
            multi_env_ids_int32 = self.robot_actor_ids_sim[env_ids].flatten()
            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.dof_state),
                gymtorch.unwrap_tensor(multi_env_ids_int32),
                len(multi_env_ids_int32))

    def _reset_buffers(self, env_ids):
        """Reset buffers."""

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def _set_viewer_params(self):
        """Set viewer parameters."""

        cam_pos = gymapi.Vec3(-1.0, -1.0, 1.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _apply_actions_as_ctrl_targets(
            self,
            ik_body_pose_actions: torch.Tensor,
            residual_dof_actions: torch.Tensor,
            do_scale: bool
    ) -> None:
        """Apply actions from policy as pose targets of the ik_body and residual
        DoF targets.

        Args:
            ik_body_pose_actions: (torch.Tensor) Changes to the pose of the
                ik_body through position in and rotation around the three axes
                (shape: [num_envs, 6]).
            residual_dof_actions: (torch.Tensor) DoF position targets for
                remaining DoFs that are not used to position the ik_body, but
                come after it like joints of the fingers of a manipulator
                (shape: [num_envs, robot_dof_count - ik_body_dof_count]).
        """

        # Interpret actions as target pos displacements and set pos target
        pos_actions = ik_body_pose_actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg_base.ctrl.pos_action_scale, device=self.device))

        if self.cfg_base.ctrl.add_pose_actions_to == "pose":
            self.ctrl_target_ik_body_pos = self.ik_body_pos + pos_actions
        elif self.cfg_base.ctrl.add_pose_actions_to == "target":
            self.ctrl_target_ik_body_pos += pos_actions
        else:
            assert False

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = ik_body_pose_actions[:, 3:6]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg_base.ctrl.rot_action_scale, device=self.device))

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        if self.cfg_base.ctrl.clamp_rot:
            rot_actions_quat = torch.where(angle.unsqueeze(-1).repeat(1, 4) > self.cfg_base.ctrl.clamp_rot_thresh,
                                           rot_actions_quat,
                                           torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs,
                                                                                                         1))

        if self.cfg_base.ctrl.add_pose_actions_to == "pose":
            self.ctrl_target_ik_body_quat = torch_utils.quat_mul(rot_actions_quat, self.ik_body_quat)
        elif self.cfg_base.ctrl.add_pose_actions_to == "target":
            self.ctrl_target_ik_body_quat = torch_utils.quat_mul(rot_actions_quat, self.ctrl_target_ik_body_quat)
        else:
            assert False

        # Action refers to relative change in target joint angles: j_{t+1}^{target} = j_{t}^{target} + a+t * \delta t * c_{rel_change_scale}.
        # Note that j_t refers to the joint control space in the [-1, 1] interval and is not in
        # rad yet. Conversion to the proper bounds happens later.
        if self.cfg_base.ctrl.relative_residual_actions:
            self.ctrl_target_residual_actuated_dof_pos += \
                residual_dof_actions * self.cfg_base.sim.dt * \
                self.cfg_base.ctrl.relative_residual_target_change_scale

            self.ctrl_target_residual_actuated_dof_pos = torch.clamp(
                self.ctrl_target_residual_actuated_dof_pos, min=-1, max=1)

        # Action refers to absolute target joint angles: j_{t+1}^{target} = a_t
        else:
            self.ctrl_target_residual_actuated_dof_pos = residual_dof_actions

        # Target poses can be set in base config, i.e., to test inv. kinematics
        if self.cfg_base.debug.override_target_pose:
            assert len(self.cfg_base.debug.target_poses) > 0, \
                "override_target_pose is True, but no target_poses were given."
            target_pos, target_quat, start_time = [], [], [0,]
            for pos, quat, duration in self.cfg_base.debug.target_poses:
                target_pos.append(pos)
                target_quat.append(quat)
                start_time.append(start_time[-1] + duration)
            start_time.pop(-1)

            current_pose_idx = len(self.cfg_base.debug.target_poses) - 1

            elapsed_time_steps = self.progress_buf[0]

            while start_time[current_pose_idx] > elapsed_time_steps:
                current_pose_idx -= 1

            self.ctrl_target_ik_body_pos = torch.Tensor(
                target_pos[current_pose_idx]).unsqueeze(0).repeat(
                self.num_envs, 1).to(self.device)
            self.ctrl_target_ik_body_quat = torch.Tensor(
                target_quat[current_pose_idx]).unsqueeze(0).repeat(
                self.num_envs, 1).to(self.device)

        # Residual DoF target can be set in base config, i.e., to test joint
        # coupling or responsiveness of the manipulator
        if self.cfg_base.debug.override_residual_dof_target:
            assert len(self.cfg_base.debug.residual_dof_targets) > 0, \
                "override_residual_dof_target is True, but no " \
                "residual_dof_targets were given."
            target_dof_pos, start_time = [], [0, ]
            for dof_pos, duration in self.cfg_base.debug.residual_dof_targets:
                target_dof_pos.append(dof_pos)
                start_time.append(start_time[-1] + duration)
            start_time.pop(-1)

            current_dof_pos_idx = len(self.cfg_base.debug.residual_dof_targets) - 1
            elapsed_time_steps = self.progress_buf[0]

            while start_time[current_dof_pos_idx] > elapsed_time_steps:
                current_dof_pos_idx -= 1

            if self.cfg_base.ctrl.relative_residual_actions:
                self.ctrl_target_residual_actuated_dof_pos += \
                    torch.Tensor(target_dof_pos[current_dof_pos_idx]).unsqueeze(
                        0).repeat(self.num_envs, 1).to(self.device) * \
                    self.cfg_base.sim.dt * \
                    self.cfg_base.ctrl.relative_residual_target_change_scale

                self.ctrl_target_residual_actuated_dof_pos = torch.clamp(
                    self.ctrl_target_residual_actuated_dof_pos, min=-1, max=1)

            else:
                self.ctrl_target_residual_actuated_dof_pos = torch.Tensor(
                    target_dof_pos[current_dof_pos_idx]).unsqueeze(0).repeat(
                    self.num_envs, 1).to(self.device)

        self.generate_ctrl_signals()

    def _move_gripper_to_dof_pos(self, gripper_dof_pos, sim_steps=20):
        """Move gripper fingers to specified DOF position using controller."""

        delta_hand_pose = torch.zeros((self.num_envs, self.cfg_task.env.numActions),
                                      device=self.device)  # No hand motion
        self._apply_actions_as_ctrl_targets(delta_hand_pose, gripper_dof_pos,
                                            do_scale=False)

        # Step sim
        for _ in range(sim_steps):
            self.render()
            self.gym.simulate(self.sim)

    def _randomize_ik_body_pose(self, env_ids, sim_steps: int) -> None:
        """Move ik_body to random pose."""

        print("start randomizing ik_body_pose")

        # Set target pos to desired initial pos
        self.ctrl_target_ik_body_pos = torch.tensor(
            self.cfg_task.randomize.ik_body_pos_initial, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)

        # Add noise to initial pos
        ik_body_pos_noise = \
            2 * (torch.rand((self.num_envs, 3), dtype=torch.float32,
                            device=self.device) - 0.5)  # [-1, 1]
        ik_body_pos_noise = ik_body_pos_noise @ torch.diag(torch.tensor(
                self.cfg_task.randomize.ik_body_pos_noise, device=self.device))
        self.ctrl_target_ik_body_pos += ik_body_pos_noise

        # Set target rot to desired initial rot
        ctrl_target_ik_body_euler = torch.tensor(
            self.cfg_task.randomize.ik_body_euler_initial, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)

        ik_body_euler_noise = \
            2 * (torch.rand((self.num_envs, 3), dtype=torch.float32,
                            device=self.device) - 0.5)  # [-1, 1]
        ik_body_euler_noise = ik_body_euler_noise @ torch.diag(
            torch.tensor(self.cfg_task.randomize.ik_body_euler_noise,
                         device=self.device))

        ctrl_target_ik_body_euler += ik_body_euler_noise

        self.ctrl_target_ik_body_quat = torch_utils.quat_from_euler_xyz(
            ctrl_target_ik_body_euler[:, 0],
            ctrl_target_ik_body_euler[:, 1],
            ctrl_target_ik_body_euler[:, 2])

        # Step sim and render
        for _ in range(sim_steps):
            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()

            print("self.ik_body_pos:", self.ik_body_pos)

            pos_error, axis_angle_error = ctrl.get_pose_error(
                ik_body_pos=self.ik_body_pos,
                ik_body_quat=self.ik_body_quat,
                ctrl_target_ik_body_pos=self.ctrl_target_ik_body_pos,
                ctrl_target_ik_body_quat=self.ctrl_target_ik_body_quat,
                jacobian_type=self.cfg_ctrl['jacobian_type'],
                rot_error_type='axis_angle')

            delta_ik_body_pose = torch.cat(
                (pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions),
                                  device=self.device)
            actions[:, :6] = delta_ik_body_pose

            self._apply_actions_as_ctrl_targets(
                ik_body_pose_actions=actions[:, :6],
                residual_dof_actions=actions[:, 6:],
                do_scale=False)

            self.gym.simulate(self.sim)
            self.render()

        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])

        # Set DOF state
        multi_env_ids_int32 = self.robot_actor_ids_sim[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32))

        print("done randomizing ik_body pose")