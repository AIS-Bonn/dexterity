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

"""Dexterity: Class for drill pick-and-place task.

Inherits drill environment class and abstract task class (not enforced).
Can be executed with python train.py task=DexterityTaskDrillPickAndPlace
"""

import hydra
import omegaconf
from isaacgym import gymapi, gymtorch, torch_utils
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.dexterity.env.drill import DexterityEnvDrill
from isaacgymenvs.tasks.dexterity.task.schema_class_task import DexterityABCTask
from isaacgymenvs.tasks.dexterity.task.schema_config_task import \
    DexteritySchemaConfigTask
from isaacgymenvs.tasks.dexterity.task.task_utils import *

from isaacgymenvs.tasks.dexterity.env.object import randomize_rotation


class DexterityTaskDrillPickAndPlace(DexterityEnvDrill, DexterityABCTask):

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

        self.drills_dropped = False

    def _get_task_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='dexterity_schema_config_task', node=DexteritySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self.cfg)
        self.max_episode_length = self.cfg_task.rl.max_episode_length  # required instance var for VecTask

        ppo_path = 'train/DexterityTaskDrillPickAndPlacePPO.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo['train']  # strip superfluous nesting

    def _update_reset_buf(self):
        """Assign environments for reset if successful or failed."""

        # If max episode length has been reached
        self.reset_buf[:] = torch.where(
            self.progress_buf[:] >= self.max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf)

        # If the drill reaches the target pose
        # Get distance to target position: Δx
        drill_target_pos_dist = torch.norm(
            self.drill_pos - self.drill_target_pos, p=2, dim=1)
        # Get smallest angle to target orientation: Δθ
        drill_target_quat_dist = quat_mul(
            self.drill_quat, quat_conjugate(self.drill_target_quat))
        drill_target_angle_dist = 2.0 * torch.asin(
            torch.clamp(torch.norm(drill_target_quat_dist[:, 0:3],
                                   p=2, dim=-1), max=1.0))
        # Check whether the target pose has been reached
        target_pos_reached = \
            drill_target_pos_dist < self.cfg_task.rl.target_pos_threshold
        target_angle_reached = \
            drill_target_angle_dist < self.cfg_task.rl.target_angle_threshold
        target_pose_reached = torch.logical_and(
            target_pos_reached, target_angle_reached)
        self.reset_buf[:] = torch.where(target_pose_reached,
                                        torch.ones_like(self.reset_buf),
                                        self.reset_buf)

    def _update_rew_buf(self):
        """Compute reward at current timestep."""
        tool_grasping_reward, tool_grasping_reward_terms, ik_body_pose_reached, all_keypoints_reached, tool_picked_up = self._compute_tool_grasping_reward()
        self.rew_buf[:] = tool_grasping_reward

        # Get distance to target position: Δx
        drill_target_pos_dist = torch.norm(
            self.to_drill_target_pos, p=2, dim=1)

        # Get smallest angle to target orientation: Δθ
        drill_target_angle_dist = 2.0 * torch.asin(
            torch.clamp(torch.norm(self.to_drill_target_quat[:, 0:3],
                                   p=2, dim=-1), max=1.0))

        # Check whether the target pose has been reached
        target_pos_reached = \
            drill_target_pos_dist < self.cfg_task.rl.target_pos_threshold
        target_angle_reached = \
            drill_target_angle_dist < self.cfg_task.rl.target_angle_threshold
        target_pose_reached = torch.logical_and(
            target_pos_reached, target_angle_reached)

        reward_terms = {}
        for reward_term, scale in self.cfg_task.rl.reward.items():
            # Penalize large actions
            if reward_term == 'action_penalty':
                squared_action_norm = torch.linalg.norm(
                    self.actions, dim=-1)
                reward = - squared_action_norm * scale

            # Reward matching the target pose.
            elif reward_term == 'target_pose_matching':
                alpha = 5.
                beta = 0.5
                reward = scale * all_keypoints_reached * torch.exp(
                    -alpha * drill_target_pos_dist - beta * drill_target_angle_dist)

            # Reward progress towards target position
            elif reward_term == 'target_pos_dist_penalty':
                reward = torch.logical_and(
                    ik_body_pose_reached, all_keypoints_reached) * \
                         hyperbole_rew(
                        scale, drill_target_pos_dist, c=0.05, pow=1)

            # Reward progress towards target orientation
            elif reward_term == 'target_quat_dist_penalty':
                reward = torch.logical_and(
                    ik_body_pose_reached, all_keypoints_reached) * \
                         hyperbole_rew(
                             scale, drill_target_angle_dist, c=0.05, pow=1)

            # Reward reaching the target pose
            elif reward_term == 'success_bonus':
                reward = scale * target_pose_reached

            else:
                continue

            self.rew_buf[:] += reward
            reward_terms["reward_terms/" + reward_term] = reward.mean()

        if "reward_terms" in self.cfg_base.logging.keys():
            reward_terms = {**reward_terms, **tool_grasping_reward_terms}
            self.log(reward_terms)
            print("reward_terms:", reward_terms)
            print("self.rew_buf:", self.rew_buf)

    def reset_idx(self, env_ids):
        """Reset specified environments."""

        if self.drills_dropped:
            self.reset_tool(env_ids, apply_reset=True)
        else:
            self.drop_tool(
                env_ids,
                sim_steps=self.cfg_task.randomize.num_drill_drop_steps)
            self.drills_dropped = True

        self._reset_robot(env_ids, apply_reset=True)

        if self.cfg_task.randomize.move_to_pre_grasp_pose:
            self.move_to_curriculum_pose(env_ids, sim_steps=500)

        self._reset_target_pose(env_ids, apply_reset=True)
        self._reset_buffers(env_ids)

    def _reset_target_pose(self, env_ids, apply_reset: bool = True) -> None:
        # Randomize target position of drill
        drill_pos_target = self._get_random_target_pos(env_ids)
        drill_quat_target = self._get_random_target_quat(env_ids)

        # Set root state tensor of the simulation
        self.root_pos[env_ids, self.drill_site_actor_id_env] = drill_pos_target
        self.root_quat[env_ids, self.drill_site_actor_id_env] = \
            drill_quat_target

        # Reset pose of drill and drill-site.
        if apply_reset:
            drill_indices = self.drill_actor_ids_sim[env_ids].to(torch.int32)
            drill_site_indices = self.drill_site_actor_ids_sim[env_ids].to(
                torch.int32)
            indices = torch.cat([drill_indices, drill_site_indices])
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_state),
                gymtorch.unwrap_tensor(indices),
                len(indices))

    def _get_random_target_pos(self, env_ids) -> torch.Tensor:
        drill_pos_target = torch.tensor(
            self.cfg_task.randomize.drill_pos_target, device=self.device
        ).unsqueeze(0).repeat(len(env_ids), 1)
        drill_pos_target_noise = \
            2 * (torch.rand((len(env_ids), 3), dtype=torch.float32,
                            device=self.device) - 0.5)  # [-1, 1]
        drill_pos_target_noise = drill_pos_target_noise @ torch.diag(torch.tensor(
            self.cfg_task.randomize.drill_pos_target_noise, device=self.device))
        drill_pos_target += drill_pos_target_noise
        return drill_pos_target

    def _get_random_target_quat(self, env_ids) -> torch.Tensor:
        drill_quat_target = torch.tensor(
            [[0, 0, 0, 1]], dtype=torch.float,
            device=self.device).repeat(len(env_ids), 1)

        x_unit_tensor = to_torch(
            [1, 0, 0], dtype=torch.float, device=self.device).repeat(
            (len(env_ids), 1))
        y_unit_tensor = to_torch(
            [0, 1, 0], dtype=torch.float, device=self.device).repeat(
            (len(env_ids), 1))
        rand_floats = torch_rand_float(
            -0.1, 0.1, (len(env_ids), 2), device=self.device)
        drill_quat_target = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1], x_unit_tensor, y_unit_tensor)
        return drill_quat_target

    def visualize_ik_body_demo_pose(self, env_id: int, axis_length: float = 0.3
                                    ) -> None:
        self.visualize_body_pose("ik_body_demo", env_id, axis_length)
