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

"""Dexterity: Class for bin pick task.

Inherits object environment class and abstract task class (not enforced).
Can be executed with python train.py task=DexterityTaskBinPick
"""

import hydra
import omegaconf
import torch

from isaacgym import gymapi, gymtorch, torch_utils
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.dexterity.env.bin import DexterityEnvBin
from isaacgymenvs.tasks.dexterity.task.schema_config_task import DexteritySchemaConfigTask
from isaacgymenvs.tasks.dexterity.task.task_utils import *
from isaacgymenvs.tasks.dexterity.task.object_lift import DexterityTaskObjectLift


class DexterityTaskBinPick(DexterityEnvBin, DexterityTaskObjectLift):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless,
                 virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize environment superclass."""

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self.cfg = cfg
        self._get_task_yaml_params()
        self._acquire_task_tensors()
        self.parse_controller_spec()

        if self.cfg_task.sim.disable_gravity:
            self.disable_gravity()

        if self.viewer is not None:
            self._set_viewer_params()

        self.objects_dropped = False

    def _get_task_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='dexterity_schema_config_task', node=DexteritySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self.cfg)
        self.max_episode_length = self.cfg_task.rl.max_episode_length  # required instance var for VecTask

        asset_info_path = '../../assets/dexterity/object_sets/asset_info_object_sets.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_objects = hydra.compose(config_name=asset_info_path)
        self.asset_info_objects = self.asset_info_objects['']['']['']['']['']['']['assets']['dexterity']['object_sets']  # strip superfluous nesting

        ppo_path = 'train/DexterityTaskBinPickPPO.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo['train']  # strip superfluous nesting

        self.acquire_setup_params()

    def acquire_setup_params(self) -> None:
        # Find object drop pose and workspace extent based on setup.
        self.object_pos_drop = \
            self.cfg['randomize']['object_pos_drop'][self.cfg_env.env.setup]

        self.cfg_task.randomize.ik_body_pos_initial = \
            self.cfg['randomize']['ik_body_pos_initial'][self.cfg_env.env.setup]
        self.cfg_task.randomize.ik_body_euler_initial = \
            self.cfg['randomize']['ik_body_euler_initial'][
                self.cfg_env.env.setup]


        print("acquire_setup_params called")
        print('self.cfg_task.randomize.ik_body_pos_initial:', self.cfg_task.randomize.ik_body_pos_initial)

        #import time
        #time.sleep(1000)

    def _acquire_task_tensors(self):
        """Acquire tensors."""
        pass

    def _refresh_task_tensors(self):
        """Refresh tensors."""
        pass

    def _update_reset_buf(self):
        """Assign environments for reset if successful or failed."""
        self._compute_object_lifting_reset(self.target_object_pos,
                                           self.target_object_pos_initial)

    def _update_rew_buf(self):
        """Compute reward at current timestep."""
        self._compute_object_lifting_reward(
            self.target_object_pos, self.target_object_pos_initial)

    def reset_idx(self, env_ids):
        """Reset specified environments."""

        if self.objects_dropped:
            self._reset_object(env_ids)
        else:
            self._reset_robot(env_ids, reset_to="home",
                              randomize_ik_body_pose=False)
            self._drop_objects(
                env_ids,
                sim_steps=self.cfg_task.randomize.num_object_drop_steps)
            self.objects_dropped = True

        self._reset_goal(env_ids)
        self._reset_robot(env_ids, reset_to=self.cfg_env.env.setup + '_initial')
        self._reset_buffers(env_ids)

    def _reset_object(self, env_ids):
        """Reset root states of object."""

        for i, object_id in enumerate(self.object_actor_id_env):
            self.root_pos[env_ids, object_id] = \
                self.object_pos_initial[env_ids, i]
            self.root_quat[env_ids, object_id] = \
                self.object_quat_initial[env_ids, i]
            self.root_linvel[env_ids, object_id] = 0.0
            self.root_angvel[env_ids, object_id] = 0.0

        object_actor_ids_sim = self.object_actor_ids_sim[env_ids].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(object_actor_ids_sim),
            len(object_actor_ids_sim))

    def _drop_objects(self, env_ids, sim_steps: int):
        all_env_ids = env_ids.clone()
        objects_dropped_successfully = torch.zeros(
            (self.num_envs, self.cfg_env.env.num_objects),
            dtype=torch.bool, device=self.device)

        # Init buffers for the initial pose object will be reset to
        self.object_pos_initial = self.root_pos[
                                  :, self.object_actor_id_env].detach().clone()
        self.object_quat_initial = self.root_quat[
                                   :, self.object_actor_id_env].detach().clone()

        # Disable all object collisions initially
        self._disable_object_collisions(
            object_ids=range(self.cfg_env.env.num_objects))
        self._place_objects_before_bin()
        initial_dropping_sequence = True

        while not torch.all(objects_dropped_successfully):
            # loop through object actors per bin
            for i in range(self.cfg_env.env.num_objects):
                # Enable collisions for this object
                if initial_dropping_sequence:
                    self._enable_object_collisions(object_ids=[i])
                # Check for which env_ids this object must still be dropped
                env_ids = torch.masked_select(
                    all_env_ids, ~objects_dropped_successfully[:, i])

                if len(env_ids) > 0:
                    if self.cfg_base.debug.verbose:
                        print(f"Objects {i} must still be dropped in envs:", env_ids)

                    # Randomize drop position and orientation of object
                    object_pos_drop = self._get_random_drop_pos(env_ids)
                    object_quat_drop = self._get_random_drop_quat(env_ids)

                    # Set root state tensor of the simulation
                    self.root_pos[env_ids, self.object_actor_id_env[i]] = \
                        object_pos_drop
                    self.root_quat[env_ids, self.object_actor_id_env[i]] = \
                        object_quat_drop
                    self.root_linvel[env_ids, self.object_actor_id_env[i]] = 0.0
                    self.root_angvel[env_ids, self.object_actor_id_env[i]] = 0.0

                    object_actor_ids_sim = self.object_actor_ids_sim[env_ids, i]
                    self.gym.set_actor_root_state_tensor_indexed(
                        self.sim,
                        gymtorch.unwrap_tensor(self.root_state),
                        gymtorch.unwrap_tensor(object_actor_ids_sim),
                        len(object_actor_ids_sim))

                    # Step simulation to drop objects
                    for _ in range(sim_steps):
                        self.gym.simulate(self.sim)
                        self.render()
                        self.refresh_base_tensors()
                        self.refresh_env_tensors()
                        if len(self.cfg_base.debug.visualize) > 0 and not self.cfg[
                            'headless']:
                            self.gym.clear_lines(self.viewer)
                            self.draw_visualizations(self.cfg_base.debug.visualize)

            # Refresh tensor and set initial object poses
            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self.object_pos_initial[:] = self.object_pos.detach().clone()
            self.object_quat_initial[:] = self.object_quat.detach().clone()
            objects_dropped_successfully = self._object_in_bin(self.object_pos_initial)
            initial_dropping_sequence = False

    def _reset_goal(self, env_ids):
        """Choose new target object to be picked."""
        # Reset the materials of the old target objects
        if self.cfg_env['env']['highlight_target_object'] and not self.headless:
            for env_id in env_ids:
                old_target_object_handle = self.object_handles[env_id][
                    self.target_object_id[env_id]]
                self.gym.reset_actor_materials(
                    self.env_ptrs[env_id], old_target_object_handle,
                    gymapi.MESH_VISUAL)

        # Get random object id in scope of the number of objects
        target_object_id = torch.randint(
            self.cfg_env['env']['num_objects'], (len(env_ids),)).to(self.device)
        # Map id to actor id in the environment
        target_object_actor_id_env = \
            target_object_id + self.object_actor_id_env[0]

        self.target_object_id[env_ids] = target_object_id
        self.target_object_actor_id_env[env_ids] = target_object_actor_id_env

        # Set the color of the new target object
        if self.cfg_env['env']['highlight_target_object'] and not self.headless:
            for env_id in env_ids:
                target_object_handle = self.object_handles[env_id][
                    self.target_object_id[env_id]]

                self.gym.set_rigid_body_color(
                    self.env_ptrs[env_id], target_object_handle, 0,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(*self.cfg_env['env']['target_object_color']))

    def _reset_buffers(self, env_ids):
        """Reset buffers."""

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
