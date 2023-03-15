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

"""Dexterity: Class for mug hang task.

Inherits drill environment class and abstract task class (not enforced).
Can be executed with python train.py task=DexterityTaskMugHang
"""

import hydra
import omegaconf
import os

import torch
from isaacgym import gymapi, gymtorch, torch_utils
from isaacgym.torch_utils import *
import isaacgymenvs.tasks.dexterity.dexterity_control as ctrl
from isaacgymenvs.tasks.dexterity.env.mug import DexterityEnvMug
from isaacgymenvs.tasks.dexterity.task.schema_class_task import DexterityABCTask
from isaacgymenvs.tasks.dexterity.task.schema_config_task import \
    DexteritySchemaConfigTask
from isaacgymenvs.tasks.dexterity.task.task_utils import *


class DexterityTaskMugPutOnShelf(DexterityEnvMug, DexterityABCTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless,
                 virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize environment superclass."""

        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

        self.cfg = cfg
        self._get_task_yaml_params()
        self._acquire_task_tensors()
        self.parse_controller_spec()

        if self.cfg_task.sim.disable_gravity:
            self.disable_gravity()

        if self.viewer is not None:
            self._set_viewer_params()

        self.mugs_dropped = False

    def _get_task_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='dexterity_schema_config_task', node=DexteritySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self.cfg)
        self.max_episode_length = self.cfg_task.rl.max_episode_length  # required instance var for VecTask

        ppo_path = 'train/DexterityTaskMugPutOnShelfPPO.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo['train']  # strip superfluous nesting

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

        # If the mug has been placed on the shelf
        #self.reset_buf[:] = torch.where(
        #    self.on_shelf,
        #    torch.ones_like(self.reset_buf),
        #    self.reset_buf)

    def _update_rew_buf(self):
        """Compute reward at current timestep."""
        tool_grasping_reward, tool_grasping_reward_terms, ik_body_pose_reached, all_keypoints_reached, tool_picked_up = self._compute_tool_grasping_reward()

        self.rew_buf[:] = tool_grasping_reward

        if self.cfg_task.ablation == 'disable_demo_guidance':
            self.rew_buf[:] *= 0.

        # Get distance to shelf position: Δx
        shelf_pos_dist = torch.norm(
            self.to_shelf_pos, p=2, dim=1)
        shelf_pos_dist_initial = torch.norm(self.mug_pos_initial - self.shelf_pos, p=2, dim=1)
        # Get smallest angle to shelf orientation: Δθ
        shelf_angle_dist = 2.0 * torch.asin(
            torch.clamp(torch.norm(self.to_shelf_quat[:, 0:3],
                                   p=2, dim=-1), max=1.0))

        # Check whether the mug has been placed on the shelf
        target_pos_reached = \
            shelf_pos_dist < 0.04
        target_angle_reached = \
            shelf_angle_dist < 0.4
        self.on_shelf[:] = torch.logical_and(
            target_pos_reached, target_angle_reached)

        self.succeeded_once = torch.logical_or(self.succeeded_once,
                                               self.on_shelf)

        reward_terms = {}
        for reward_term, scale in self.cfg_task.rl.reward.items():
            # Penalize large actions
            if reward_term == 'action_penalty':
                squared_action_norm = torch.linalg.norm(
                    self.actions, dim=-1)
                reward = - squared_action_norm * scale

            elif reward_term == 'shelf_pose_matching':
                alpha = 10.
                beta = 1.
                reward = scale * self.tool_picked_up_once * all_keypoints_reached * torch.exp(
                    -alpha * shelf_pos_dist - beta * shelf_angle_dist)

            # Reward progress towards target position
            elif reward_term == 'shelf_pos_dist_penalty':
                reward = torch.logical_and(all_keypoints_reached, tool_picked_up) * (hyperbole_rew(scale, shelf_pos_dist, c=0.25, pow=1) - hyperbole_rew(scale, shelf_pos_dist_initial, c=0.25, pow=1))

            # Reward progress towards target orientation
            elif reward_term == 'shelf_quat_dist_penalty':
                reward = torch.logical_and(
                    all_keypoints_reached, tool_picked_up) * \
                         hyperbole_rew(
                             scale, shelf_angle_dist, c=0.1, pow=1)

            # Reward reaching the target pose
            elif reward_term == 'success_bonus':
                reward = scale * self.on_shelf

            else:
                continue

            self.rew_buf[:] += reward
            reward_terms["reward_terms/" + reward_term] = reward.mean()

        if "reward_terms" in self.cfg_base.logging.keys():
            reward_terms = {**reward_terms, **tool_grasping_reward_terms}
            self.log(reward_terms)
            #print("reward_terms:", reward_terms)
            #print("self.rew_buf:", self.rew_buf)

    def reset_idx(self, env_ids):
        """Reset specified environments."""

        if self.mugs_dropped:
            self.reset_tool(env_ids, apply_reset=True)
        else:
            self.drop_tool(
                env_ids,
                sim_steps=self.cfg_task.randomize.num_mug_drop_steps)
            self.mugs_dropped = True

        self._reset_robot(env_ids, apply_reset=True)

        if self.cfg_task.ablation not in ['disable_pre_grasp_pose', 'disable_demo_guidance']:
            self.move_to_curriculum_pose(env_ids, sim_steps=500)

        self._reset_buffers(env_ids)
