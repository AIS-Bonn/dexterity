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

"""Dexterity: class for object on conveyor env.

Inherits object env class. Inherited by object on conveyor task
classes. Not directly executed.

Configuration defined in DexterityEnvObjectOnConveyor.yaml.
"""

import hydra
import os

from isaacgym import gymapi
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.dexterity.env.object import DexterityEnvObject
from isaacgymenvs.tasks.dexterity.env.schema_config_env import \
    DexteritySchemaConfigEnv
from isaacgym import gymtorch


class DexterityEnvObjectOnConveyor(DexterityEnvObject):

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

        config_path = 'task/DexterityEnvObjectOnConveyor.yaml'  # relative to cfg dir
        self.cfg_env = hydra.compose(config_name=config_path)
        self.cfg_env = self.cfg_env['task']  # strip superfluous nesting

        self.object_sets_asset_root = os.path.normpath(os.path.join(
            os.path.dirname(__file__),
            self.cfg_env['env']['object_sets_asset_root']))

    def _import_env_assets(self):
        """Set object assets options. Import objects."""
        # Import objects assets like in the single object task
        object_assets = super()._import_env_assets()

        # Import conveyor asset
        conveyor_asset_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..', '..',
                         'assets', 'dexterity', 'conveyor'))
        conveyor_asset_file = 'element.urdf'
        conveyor_options = gymapi.AssetOptions()
        conveyor_options.fix_base_link = True
        conveyor_options.use_mesh_materials = True
        conveyor_options.default_dof_drive_mode = gymapi.DOF_MODE_VEL
        conveyor_asset = self.gym.load_asset(self.sim, conveyor_asset_root,
                                        conveyor_asset_file, conveyor_options)
        return (object_assets, conveyor_asset)

    def _create_actors(self, lower: gymapi.Vec3, upper: gymapi.Vec3,
                       num_per_row: int, robot_asset, table_asset,
                       env_assets) -> None:
        object_assets, conveyor_asset = env_assets
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

        conveyor_pose = gymapi.Transform()
        conveyor_pose.p.x = 0.0
        conveyor_pose.p.y = -0.64
        conveyor_pose.p.z = 0.1

        self.env_ptrs = []
        self.object_handles = []  # Isaac Gym actors
        self.object_actor_ids_sim = []  # within-sim indices
        self.conveyor_handles = []  # Isaac Gym actors
        self.conveyor_actor_ids_sim = []  # within-sim indices

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
                    used_object.rigid_body_count + \
                    self.gym.get_asset_rigid_body_count(conveyor_asset)
                max_rigid_shapes = self.base_rigid_shapes + \
                    used_object.rigid_shape_count + \
                    self.gym.get_asset_rigid_shape_count(conveyor_asset)
                self.gym.begin_aggregate(env_ptr, max_rigid_bodies,
                                         max_rigid_shapes, True)

            # Create common actors (robot, table, cameras)
            actor_count = self.create_base_actors(
                env_ptr, i, actor_count, robot_asset, robot_pose,
                table_asset, table_pose)

            # Aggregate task-specific actors (objects, conveyor-belts)
            if self.cfg_base.sim.aggregate_mode == 1:
                max_rigid_bodies = used_object.rigid_body_count + \
                    self.gym.get_asset_rigid_body_count(conveyor_asset)
                max_rigid_shapes = used_object.rigid_shape_count + \
                    self.gym.get_asset_rigid_shape_count(conveyor_asset)
                self.gym.begin_aggregate(env_ptr, max_rigid_bodies,
                                         max_rigid_shapes, True)

            # Create object actor
            object_handle = self.gym.create_actor(
                env_ptr, used_object.asset, used_object.start_pose,
                used_object.name, i, 0, 2)
            self.object_actor_ids_sim.append(actor_count)
            self.object_handles.append(object_handle)
            actor_count += 1

            # Create conveyor-belt actor
            conveyor_handle = self.gym.create_actor(
                env_ptr, conveyor_asset, conveyor_pose, 'conveyor_belt', i, 0,
                3)
            for idx in range(
                    self.gym.get_asset_rigid_body_count(conveyor_asset)):
                self.gym.set_rigid_body_color(
                    env_ptr, conveyor_handle, idx, gymapi.MESH_VISUAL,
                    gymapi.Vec3(*self.cfg_env['env']['conveyor_color']))
            self.conveyor_actor_ids_sim.append(actor_count)
            self.conveyor_handles.append(conveyor_handle)
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
        self.conveyor_actor_ids_sim = torch.tensor(
            self.conveyor_actor_ids_sim, dtype=torch.int32, device=self.device)

        # For extracting root pos/quat
        self.object_actor_id_env = self.gym.find_actor_index(
            env_ptr, used_object.name, gymapi.DOMAIN_ENV)
        self.conveyor_actor_id_env = self.gym.find_actor_index(
            env_ptr, 'conveyor_belt', gymapi.DOMAIN_ENV)

        self.set_conveyor_dof_props()

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self.reset_conveyor_elements()

    def set_conveyor_dof_props(self):
        # No tensor API for getting/setting actor DOF props; thus, loop required
        for env_ptr, conveyor_handle in zip(
                self.env_ptrs, self.conveyor_handles):
            conveyor_dof_props = self.gym.get_actor_dof_properties(
                env_ptr, conveyor_handle)
            conveyor_dof_props['stiffness'][:] = 1e9
            conveyor_dof_props['damping'][:] = 1e5
            self.gym.set_actor_dof_properties(env_ptr, conveyor_handle,
                                              conveyor_dof_props)

    def set_conveyor_belt_in_motion(self):
        ctrl_target_dof_vel = torch.zeros((self.num_envs, self.num_dofs),
                                          device=self.device)
        ctrl_target_dof_vel[:, self.robot_dof_count:] = self.cfg_env["env"]["conveyor_speed"]

        self.gym.set_dof_velocity_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(ctrl_target_dof_vel),
            gymtorch.unwrap_tensor(self.conveyor_actor_ids_sim),
            len(self.conveyor_actor_ids_sim))

    def reset_conveyor_elements(self):
        conveyor_elements_pos = 0.16 * torch.arange(0, 9).unsqueeze(0).repeat(
            self.num_envs, 1).to(self.device) + self.dof_pos[:,
                                                self.robot_dof_count:] - 0.64
        conveyor_elements_out_of_reach = conveyor_elements_pos > 0.72

        if torch.any(conveyor_elements_out_of_reach):
            conveyor_dof_pos = self.dof_pos[:, self.robot_dof_count:].clone()
            conveyor_dof_pos = torch.where(
                conveyor_elements_out_of_reach,
                conveyor_dof_pos - 1.44, conveyor_dof_pos)
            self.dof_pos[:, self.robot_dof_count:] = conveyor_dof_pos
            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.dof_state),
                gymtorch.unwrap_tensor(self.conveyor_actor_ids_sim),
                len(self.conveyor_actor_ids_sim))

