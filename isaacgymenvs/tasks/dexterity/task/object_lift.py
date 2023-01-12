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

"""Dexterity: Class for object lift task.

Inherits object environment class and abstract task class (not enforced).
Can be executed with python train.py task=DexterityTaskObjectLift
"""

import hydra
import omegaconf

from isaacgym import gymapi, gymtorch, torch_utils
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.dexterity.env.object import DexterityEnvObject
from isaacgymenvs.tasks.dexterity.task.schema_class_task import DexterityABCTask
from isaacgymenvs.tasks.dexterity.task.schema_config_task import DexteritySchemaConfigTask
from isaacgymenvs.tasks.dexterity.task.task_utils import *


class DexterityTaskObjectLift(DexterityEnvObject, DexterityABCTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless,
                 virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize environment superclass."""

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self.cfg = cfg
        self._get_task_yaml_params()
        self._acquire_task_tensors()
        self.parse_camera_spec()
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

        ppo_path = 'train/DexterityTaskObjectLiftPPO.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo['train']  # strip superfluous nesting

    def _acquire_task_tensors(self):
        """Acquire tensors."""
        pass

    def _refresh_task_tensors(self):
        """Refresh tensors."""
        pass

    def _update_reset_buf(self):
        """Assign environments for reset if successful or failed."""
        self._compute_object_lifting_reset(self.object_pos, self.object_pos_initial)

    def _compute_object_lifting_reset(self, object_pos, object_pos_initial):
        # If max episode length has been reached
        self.reset_buf[:] = torch.where(
            self.progress_buf[:] >= self.max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf)

        # If the object has been lifted to the target height
        object_height = object_pos[:, 2]
        object_height_initial = object_pos_initial[:, 2]
        object_lifted = (object_height - object_height_initial) > \
                        self.cfg_task.rl.target_height
        self.reset_buf[:] = torch.where(object_lifted,
                                        torch.ones_like(self.reset_buf),
                                        self.reset_buf)

    def _update_rew_buf(self):
        """Compute reward at current timestep."""
        self._compute_object_lifting_reward(
            self.object_pos, self.object_pos_initial)

    def _compute_object_lifting_reward(self, object_pos, object_pos_initial):
        """Compute object lifting reward at current timestep."""
        self.rew_buf[:] = 0.
        object_height = object_pos[:, 2]
        object_height_initial = object_pos_initial[:, 2]

        delta_target_height = torch.clamp(
            self.cfg_task.rl.target_height -
            (object_height - object_height_initial), min=0)
        delta_lift_off_height = torch.clamp(
            self.cfg_task.rl.lift_off_height -
            (object_height - object_height_initial), min=0)
        object_lifted = (object_height - object_height_initial) > \
                        self.cfg_task.rl.target_height

        reward_dict = {}
        for reward_term, scale in self.cfg_task.rl.reward.items():
            # Penalize distance of keypoint groups to object
            if reward_term.startswith(tuple(self.keypoint_dict.keys())):
                keypoint_group_name = reward_term.split('_')[0]
                assert reward_term.endswith('_dist_penalty'), \
                    f"Reward term {reward_term} depends on " \
                    f"keypoint_group {keypoint_group_name}, but is not " \
                    f"a distance penalty."
                keypoint_pos = getattr(self, keypoint_group_name + '_pos')
                object_pos_expanded = object_pos.unsqueeze(1).repeat(
                    1, keypoint_pos.shape[1], 1)
                keypoint_dist = torch.norm(
                    keypoint_pos - object_pos_expanded, dim=-1).mean(1)
                reward = hyperbole_rew(scale, keypoint_dist, c=0.1, pow=1)

            # Penalize large actions
            elif reward_term == 'action_penalty':
                action_norm = torch.norm(self.actions, p=2, dim=-1)
                reward = - action_norm * scale

            # Reward the height progress of the object towards lift-off
            elif reward_term == 'object_lift_off_reward':
                reward = \
                    hyperbole_rew(
                        scale, delta_lift_off_height, c=0.02, pow=1) - \
                    hyperbole_rew(
                        scale, torch.ones_like(delta_lift_off_height) *
                               self.cfg_task.rl.lift_off_height, c=0.02, pow=1)

            # Reward the height progress of the object towards target height
            elif reward_term == 'object_target_reward':
                reward = \
                    hyperbole_rew(
                        scale, delta_target_height, c=0.05, pow=1) - \
                    hyperbole_rew(
                        scale, torch.ones_like(delta_target_height) *
                               self.cfg_task.rl.target_height, c=0.05, pow=1)

            # Reward reaching the target height
            elif reward_term == 'success_bonus':
                reward = scale * object_lifted

            else:
                assert False, f"Unknown reward term {reward_term}."

            self.rew_buf[:] += reward
            reward_dict[reward_term] = reward.mean()
        self.log(reward_dict)

    def reset_idx(self, env_ids):
        """Reset specified environments."""

        if self.objects_dropped:
            self._reset_object(env_ids)
        else:
            self._drop_object(
                env_ids,
                sim_steps=self.cfg_task.randomize.num_object_drop_steps)
            self.objects_dropped = True

        self._reset_robot(env_ids)

        #self._randomize_ik_body_pose(
        #    env_ids,
        #    sim_steps=self.cfg_task.randomize.num_ik_body_initial_move_steps)

        self._reset_buffers(env_ids)

    def _reset_object(self, env_ids):
        """Reset root states of object."""
        self.root_pos[env_ids, self.object_actor_id_env] = \
            self.object_pos_initial[env_ids]
        self.root_quat[env_ids, self.object_actor_id_env] = \
            self.object_quat_initial[env_ids]
        self.root_linvel[env_ids, self.object_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.object_actor_id_env] = 0.0

        object_indices = self.object_actor_ids_sim[env_ids].to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(object_indices),
            len(object_indices))

    def _drop_object(self, env_ids, sim_steps: int):
        all_env_ids = env_ids.clone()
        object_dropped_successfully = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device)

        # Init buffers for the initial pose object will be reset to
        self.object_pos_initial = self.root_pos[
            all_env_ids, self.object_actor_id_env].detach().clone()
        self.object_quat_initial = self.root_quat[
            all_env_ids, self.object_actor_id_env].detach().clone()

        while not torch.all(object_dropped_successfully):
            # Check for which env_ids we need to drop the object
            env_ids = torch.masked_select(
                all_env_ids, ~object_dropped_successfully)

            if self.cfg_base.debug.verbose:
                print("Objects must still be dropped in envs:",
                      env_ids.cpu().numpy())

            # Randomize drop position and orientation of object
            object_pos_drop = self._get_random_drop_pos(env_ids)
            object_quat_drop = self._get_random_drop_quat(env_ids)

            # Set root state tensor of the simulation
            self.root_pos[env_ids, self.object_actor_id_env] = object_pos_drop
            self.root_quat[env_ids, self.object_actor_id_env] = object_quat_drop
            self.root_linvel[env_ids, self.object_actor_id_env] = 0.0
            self.root_angvel[env_ids, self.object_actor_id_env] = 0.0

            object_actor_ids_sim = self.object_actor_ids_sim[env_ids]
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_state),
                gymtorch.unwrap_tensor(object_actor_ids_sim),
                len(object_actor_ids_sim))

            # Step simulation to drop objects
            for _ in range(sim_steps):
                self.gym.simulate(self.sim)
                self.render()
                if len(self.cfg_base.debug.visualize) > 0 and not self.cfg[
                    'headless']:
                    self.gym.clear_lines(self.viewer)
                    self.draw_visualizations(self.cfg_base.debug.visualize)

            # Refresh tensor and set initial object poses
            self.refresh_base_tensors()
            self.object_pos_initial[env_ids] = self.root_pos[
                env_ids, self.object_actor_id_env].detach().clone()
            self.object_quat_initial[env_ids] = self.root_quat[
                env_ids, self.object_actor_id_env].detach().clone()

            object_dropped_successfully = self._object_in_workspace(
                self.object_pos_initial)
