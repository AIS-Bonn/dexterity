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

"""Dexterity: Class for hammer drive nail task.

Inherits hammer environment class and abstract task class (not enforced).
Can be executed with python train.py task=DexterityTaskHammerDriveNail
"""

import hydra
import omegaconf

from isaacgym import gymapi, gymtorch, torch_utils
from isaacgym.torch_utils import *
import isaacgymenvs.tasks.dexterity.dexterity_control as ctrl
from isaacgymenvs.tasks.dexterity.env.hammer import DexterityEnvHammer
from isaacgymenvs.tasks.dexterity.task.schema_class_task import DexterityABCTask
from isaacgymenvs.tasks.dexterity.task.schema_config_task import \
    DexteritySchemaConfigTask
from isaacgymenvs.tasks.dexterity.task.task_utils import *
import os


class DexterityTaskHammerDriveNail(DexterityEnvHammer, DexterityABCTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless,
                 virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize environment superclass."""

        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

        self.cfg = cfg
        self._get_task_yaml_params(task_name="HammerDriveNail")
        self._acquire_task_tensors()
        self.parse_controller_spec()

        if self.cfg_task.sim.disable_gravity:
            self.disable_gravity()

        if self.viewer is not None:
            self._set_viewer_params()

        self.hammers_dropped = False

    def _update_reset_buf(self):
        """Assign environments for reset if successful or failed."""

        # If max episode length has been reached
        self.reset_buf[:] = torch.where(
            self.progress_buf[:] >= self.max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf)

        # Log exponentially weighted moving average (EWMA) of the success rate
        if "success_rate_ewma" in self.cfg_base.logging.keys():
            if not hasattr(self, "success_rate_ewma"):
                self.success_rate_ewma = 0.
            num_resets = torch.sum(self.reset_buf)
            # Update success rate if resets have actually occurred
            if num_resets > 0:
                num_successes = torch.sum(self.succeeded_once)
                curr_success_rate = num_successes / num_resets
                alpha = (num_resets / self.num_envs) * \
                        self.cfg_base.logging.success_rate_ewma.alpha
                self.success_rate_ewma = alpha * curr_success_rate + (
                        1 - alpha) * self.success_rate_ewma
                self.log({"success_rate_ewma": self.success_rate_ewma})

        #nail_depth = self.dof_pos[:, -1]
        #nail_driven = nail_depth < self.cfg_task.rl.target_nail_depth

        #self.reset_buf[:] = torch.where(
        #    nail_driven,
        #    torch.ones_like(self.reset_buf),
        #    self.reset_buf)

    def _update_rew_buf(self):
        """Compute reward at current timestep."""
        tool_grasping_reward, tool_grasping_reward_terms, ik_body_pose_reached, all_keypoints_reached, lifted = self._compute_tool_grasping_reward()

        self.rew_buf[:] = tool_grasping_reward

        if self.cfg_task.ablation == 'disable_demo_guidance':
            self.rew_buf[:] *= 0.

        # Get distance to nail position: Δx
        nail_pos_dist = torch.norm(
            self.nail_pos - self.hammer_pos, p=2, dim=1)

        # Get depth of the nail
        nail_depth = self.dof_pos[:, -1]
        nail_driven = nail_depth < self.cfg_task.rl.target_nail_depth

        self.succeeded_once = torch.logical_or(self.succeeded_once,
                                               nail_driven)

        reward_terms = {}
        for reward_term, scale in self.cfg_task.rl.reward.items():
            # Penalize large actions
            if reward_term == 'action_penalty':
                squared_action_norm = torch.linalg.norm(
                    self.actions, dim=-1)
                reward = - squared_action_norm * scale

            elif reward_term == 'nail_dist_penalty':
                reward = self.tool_picked_up_once * hyperbole_rew(
                    scale, nail_pos_dist, c=0.05, pow=1)

            # Reward progress towards target position
            elif reward_term == 'nail_depth_reward':
                reward = scale * self.tool_picked_up_once * -nail_depth

            # Reward reaching the target pose
            elif reward_term == 'success_bonus':
                reward = scale * self.tool_picked_up_once * nail_driven

            else:
                continue

            self.rew_buf[:] += reward
            reward_terms["reward_terms/" + reward_term] = reward.mean()

        if "reward_terms" in self.cfg_base.logging.keys():
            reward_terms = {**reward_terms, **tool_grasping_reward_terms}
            self.log(reward_terms)

        print("reward_terms:", reward_terms)
        print("self.tool_picked_up_once:", self.tool_picked_up_once)

    def reset_idx(self, env_ids):
        """Reset specified environments."""

        if self.hammers_dropped:
            self.reset_tool(env_ids, apply_reset=True)
        else:
            self.drop_tool(
                env_ids,
                sim_steps=self.cfg_task.randomize.num_hammer_drop_steps)
            self.hammers_dropped = True

        self._reset_robot(env_ids, apply_reset=False)
        self._reset_nail(env_ids)

        if self.cfg_task.ablation not in ['disable_pre_grasp_pose', 'disable_demo_guidance']:
            self.move_to_curriculum_pose(env_ids, sim_steps=500)

        self._reset_buffers(env_ids)

    def _reset_nail(self, env_ids):
        # Randomize root state of nail
        nail_noise_xy = 2 * (
                torch.rand((self.num_envs, 2), dtype=torch.float32,
                           device=self.device) - 0.5)  # [-1, 1]
        nail_noise_xy = nail_noise_xy @ torch.diag(
            torch.tensor(
                self.cfg_task.randomize.nail_pos_xy_initial_noise,
                device=self.device))
        self.root_pos[env_ids, self.nail_actor_id_env, 0] = \
            self.cfg_task.randomize.nail_pos_xy_initial[0] + \
            nail_noise_xy[env_ids, 0]
        self.root_pos[env_ids, self.nail_actor_id_env, 1] = \
            self.cfg_task.randomize.nail_pos_xy_initial[1] + \
            nail_noise_xy[env_ids, 1]
        self.root_pos[env_ids, self.nail_actor_id_env, 2] = 0.1

        hammer_and_nail_indices = torch.cat(
            [self.nail_actor_ids_sim[env_ids],
             self.hammer_actor_ids_sim[env_ids]]).to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(hammer_and_nail_indices),
            len(hammer_and_nail_indices))

        # reset DoF state of nail
        self.dof_pos[env_ids, -1] = 0.0
        self.dof_vel[env_ids, -1] = 0.0
        # dof state tensor should only be set once. Hence, setting nail and
        # robot together here.
        robot_and_nail_indices = torch.cat(
            [self.robot_actor_ids_sim[env_ids],
             self.nail_actor_ids_sim[env_ids]]).to(torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(robot_and_nail_indices),
            len(robot_and_nail_indices))
