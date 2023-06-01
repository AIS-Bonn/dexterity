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
from isaacgymenvs.tasks.dexterity.task.sim2real_utils import CalibrationUtils
import isaacgymenvs.tasks.dexterity.dexterity_control as ctrl


class DexterityTaskObjectLift(DexterityEnvObject, DexterityABCTask, CalibrationUtils):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless,
                 virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize environment superclass."""

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self.cfg = cfg
        self._get_task_yaml_params()
        self.acquire_setup_params()
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

        ppo_path = 'train/DexterityTaskObjectLiftPPO.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo['train']  # strip superfluous nesting

    def acquire_setup_params(self) -> None:
        # Find object drop pose and workspace extent based on setup.
        self.object_pos_drop = \
            self.cfg['randomize']['object_pos_drop'][self.cfg_env.env.setup]
        self.workspace_extent_xy = \
            self.cfg['randomize']['workspace_extent_xy'][self.cfg_env.env.setup]

        self.cfg_task.randomize.ik_body_pos_initial = \
            self.cfg['randomize']['ik_body_pos_initial'][self.cfg_env.env.setup]
        self.cfg_task.randomize.ik_body_euler_initial = \
            self.cfg['randomize']['ik_body_euler_initial'][self.cfg_env.env.setup]

    def _acquire_task_tensors(self):
        """Acquire tensors."""
        self.object_lifted = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _refresh_task_tensors(self):
        """Refresh tensors."""
        pass

    def pre_physics_step(self, actions):
        if 'calibrate' in self.cfg_task.keys() and self.cfg_task['calibrate']:
            actions = self.run_calibration_procedure(actions)
        super().pre_physics_step(actions)

    def _update_reset_buf(self):
        """Assign environments for reset if successful or failed."""
        self._compute_object_lifting_reset(self.object_pos, self.object_pos_initial)

    def _compute_object_lifting_reset(self, object_pos, object_pos_initial, log_object_wise_success: bool = True):
        # If max episode length has been reached
        self.reset_buf[:] = torch.where(
            self.progress_buf[:] >= self.max_episode_length - 1,
            torch.ones_like(self.reset_buf),
            self.reset_buf)

        self._compute_object_lifting_success_rate(object_pos, object_pos_initial, log_object_wise_success)
        
    def _compute_object_lifting_success_rate(self, object_pos, object_pos_initial, log_object_wise_success: bool = True):
        # Success is defined as lifting the object above the target height in the course of the episode.
        object_height = object_pos[:, 2]
        object_height_initial = object_pos_initial[:, 2]
        object_lifted = (object_height - object_height_initial) > \
                        self.cfg_task.rl.target_height
        self.object_lifted[:] = torch.logical_or(object_lifted, self.object_lifted)                      
        

        # Log exponentially weighted moving average (EWMA) of the success rate
        if "success_rate_ewma" in self.cfg_base.logging.keys():
            if not hasattr(self, "success_rate_ewma"):
                self.success_rate_ewma = 0.
            num_resets = torch.sum(self.reset_buf)
            # Update success rate if resets have actually occurred
            if num_resets > 0:
                num_successes = torch.sum(self.object_lifted)
                curr_success_rate = num_successes / num_resets
                alpha = (num_resets / self.num_envs) * \
                    self.cfg_base.logging.success_rate_ewma.alpha
                self.success_rate_ewma = alpha * curr_success_rate + (
                        1 - alpha) * self.success_rate_ewma
                self.log({"success_rate_ewma/overall": self.success_rate_ewma})

                # Log exponentially weighted moving average (EWMA) of the
                # object-wise success rate
                if log_object_wise_success:
                    for i, obj in enumerate(self.objects):
                        if not hasattr(self, obj.name + "_success_rate_ewma"):
                            setattr(self, obj.name + "_success_rate_ewma", 0.)
                        num_resets = torch.sum(self.reset_buf[i::len(self.objects)])
                        if num_resets > 0:
                            num_successes = torch.sum(self.object_lifted[i::len(self.objects)])
                            curr_success_rate = num_successes / num_resets
                            alpha = (num_resets * len(self.objects) / self.num_envs) * \
                                    self.cfg_base.logging.success_rate_ewma.alpha
                            setattr(
                                self, obj.name + "_success_rate_ewma",
                                alpha * curr_success_rate + (1 - alpha) * getattr(
                                    self, obj.name + "_success_rate_ewma"))
                            self.log({"success_rate_ewma/" + obj.name: getattr(
                                self, obj.name + "_success_rate_ewma")})

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

        reward_terms = {}
        for reward_term, scale in self.cfg_task.rl.reward.items():
            # Penalize distance of keypoint groups to object
            if reward_term.startswith(tuple(self.keypoint_dict.keys())):
                if reward_term.endswith('_dist_penalty'):
                    keypoint_group_name = reward_term[:-len('_dist_penalty')]
                    keypoint_pos = getattr(self, keypoint_group_name + '_pos')
                    object_pos_expanded = object_pos.unsqueeze(1).repeat(
                        1, keypoint_pos.shape[1], 1)
                    keypoint_dist = torch.norm(
                        keypoint_pos - object_pos_expanded, dim=-1).mean(dim=-1)
                    reward = -scale * keypoint_dist
                elif reward_term.endswith('_proximity'):
                    keypoint_group_name = reward_term[:-len('_proximity')]
                    keypoint_pos = getattr(self, keypoint_group_name + '_pos')
                    object_pos_expanded = object_pos.unsqueeze(1).repeat(
                        1, keypoint_pos.shape[1], 1)
                    keypoint_dist = torch.norm(
                        keypoint_pos - object_pos_expanded, dim=-1).mean(dim=-1)
                    in_proximity = keypoint_dist < 0.15
                    reward = torch.where(in_proximity, 0. * keypoint_dist, -scale * keypoint_dist)
                else:
                    assert False

            # Penalize large actions
            elif reward_term == 'action_penalty':
                squared_action_norm = torch.linalg.norm(
                    self.actions, dim=-1)
                reward = - squared_action_norm * scale

            # Reward the height progress of the object towards lift-off
            elif reward_term == 'object_lift_off_reward':
                reward = \
                    hyperbole_rew(
                        scale, delta_lift_off_height, c=0.02, pow=1) - \
                    hyperbole_rew(
                        scale, torch.ones_like(delta_lift_off_height) *
                               self.cfg_task.rl.lift_off_height, c=0.02, pow=1)
                reward = torch.clamp(reward, min=0)

            # Reward the height progress of the object towards target height
            elif reward_term == 'object_target_reward':
                reward = scale * (object_height - object_height_initial)

            # Reward reaching the target height
            elif reward_term == 'success_bonus':
                reward = float(scale) * object_lifted

            else:
                assert False, f"Unknown reward term {reward_term}."

            self.rew_buf[:] += reward
            reward_terms["reward_terms/" + reward_term] = reward.mean()
        if "reward_terms" in self.cfg_base.logging.keys():
            self.log(reward_terms)
            #print(reward_terms)

    def reset_idx(self, env_ids):
        """Reset specified environments."""

        if self.objects_dropped:
            self._reset_object(env_ids)
        else:
            self._reset_robot(env_ids, reset_to="home",
                              randomize_ik_body_pose=False)
            self._drop_object(
                env_ids,
                sim_steps=self.cfg_task.randomize.num_object_drop_steps)
            self.objects_dropped = True

        self._reset_robot(env_ids, reset_to=self.cfg_env.env.setup + '_initial')

        # Initialize SAM segmentation at the start of each episode.
        if any(obs.startswith("detected_pointcloud") for obs in self.cfg["env"]["observations"]):
            self._reset_segmentation_tracking(env_ids)

        self.object_lifted[env_ids] = False
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



    def _randomize_ik_body_pose(self, env_ids, sim_steps: int) -> None:
        """Move ik_body to random pose."""

        if self.cfg_task.randomize.move_to_pregrasp_pose:
            self._move_to_pregrasp_pose(env_ids, sim_steps, self.object_pos)
        else:
            super()._randomize_ik_body_pose(env_ids, sim_steps)


    def _move_to_pregrasp_pose(self, env_ids, sim_steps: int, object_pos: torch.Tensor) -> None:

        self.gym.set_dof_position_target_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.ctrl_target_dof_pos),
            gymtorch.unwrap_tensor(self.robot_actor_ids_sim),
            len(self.robot_actor_ids_sim))

        # Set target pos to desired initial pos
        target_ik_body_pos = object_pos.clone()
        target_ik_body_pos[:, 2] += 0.2
        target_ik_body_pos[:, 1] -= 0.075

        target_ik_body_euler = torch.tensor(
            self.cfg_task.randomize.ik_body_euler_initial, device=self.device
        ).unsqueeze(0).repeat(self.num_envs, 1)

        target_ik_body_quat = torch_utils.quat_from_euler_xyz(
            target_ik_body_euler[:, 0],
            target_ik_body_euler[:, 1],
            target_ik_body_euler[:, 2])

        self.initial_ik_body_quat = target_ik_body_quat.clone()

        # Step sim and render
        for rand_step in range(sim_steps):
            self.refresh_base_tensors()
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)

            # On the initial step after resetting, the ik_body is still in the
            # wrong pose. Hence, we do not assume a pos or quat error.
            if self.cfg_base.ctrl.add_pose_actions_to == 'pose':
                current_pos = self.ik_body_pos
                current_quat = self.ik_body_quat
            elif self.cfg_base.ctrl.add_pose_actions_to == 'target':
                current_pos = self.ctrl_target_ik_body_pos
                current_quat = self.ctrl_target_ik_body_quat
            else:
                assert False

            pos_error, axis_angle_error = ctrl.get_pose_error(
                ik_body_pos=current_pos,
                ik_body_quat=current_quat,
                ctrl_target_ik_body_pos=target_ik_body_pos,
                ctrl_target_ik_body_quat=target_ik_body_quat,
                jacobian_type=self.cfg_ctrl['jacobian_type'],
                rot_error_type='axis_angle')

            delta_ik_body_pose = torch.cat(
                (pos_error, axis_angle_error), dim=-1)
            actions = torch.zeros((self.num_envs, self.cfg_task.env.numActions),
                                  device=self.device)
            actions[:, :6] = delta_ik_body_pose

            # Open hand.
            actions[:, 6] = -1.
            actions[:, 7] = -1.
            actions[:, 8:] = 1.

            if rand_step > 0:
                self._apply_actions_as_ctrl_targets(
                    ik_body_pose_actions=actions[:, :6],
                    residual_dof_actions=actions[:, 6:],
                    do_scale=False)

            self.gym.simulate(self.sim)
            self.render()
            if len(self.cfg_base.debug.visualize) > 0 and not self.cfg[
                'headless']:
                self.gym.clear_lines(self.viewer)
                self.draw_visualizations(self.cfg_base.debug.visualize)