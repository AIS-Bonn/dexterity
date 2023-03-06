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
import os

import torch
from isaacgym import gymapi, gymtorch, torch_utils
from isaacgym.torch_utils import *
import isaacgymenvs.tasks.dexterity.dexterity_control as ctrl
from isaacgymenvs.tasks.dexterity.env.drill import DexterityEnvDrill
from isaacgymenvs.tasks.dexterity.task.schema_class_task import DexterityABCTask
from isaacgymenvs.tasks.dexterity.task.schema_config_task import \
    DexteritySchemaConfigTask
from isaacgymenvs.tasks.dexterity.task.task_utils import *


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

    def _acquire_task_tensors(self):
        """Acquire tensors."""
        self._acquire_demo_pose()

    def _refresh_task_tensors(self):
        """Refresh tensors."""
        self._refresh_demo_pose()

    def _acquire_demo_pose(self):
        demo_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..',
                         '..', 'assets', 'dexterity', 'tools',
                         'drills', 'train', 'black_and_decker_unknown',
                         self.robot.manipulator.model_name + '_demo_pose.npz'))
        if os.path.isfile(demo_path):
            self.demo_pose_npz = np.load(demo_path)
            self.demo_pose = {}
            for k, v in self.demo_pose_npz.items():
                # Store demo pose of keypoints relative to the drill
                # (shape: [num_envs, num_keypoints, 3 or 4]
                if k.startswith(tuple(self.keypoint_dict.keys())):
                    self.demo_pose[k] = torch.from_numpy(v).to(
                        self.device, dtype=torch.float32).repeat(
                        self.num_envs, 1, 1)
                    setattr(self, k, self.demo_pose[k].clone())

                # Store ik_body pose and residual dof pos
                # (shape: [num_envs, 3 or 4 or num_residual_actuated_dofs]
                else:
                    self.demo_pose[k] = torch.from_numpy(v).to(
                        self.device, dtype=torch.float32).repeat(
                        self.num_envs, 1)
                    setattr(self, k, self.demo_pose[k].clone())

                # Store relative positions and rotations
                if hasattr(self, k.replace('_demo', '')):
                    if '_pos' in k:
                        setattr(self, 'to_' + k,
                                self.demo_pose[k].clone() - getattr(
                                    self, k.replace('_demo', '')))
                    elif '_quat' in k:
                        setattr(self, 'to_' + k, quat_mul(
                                self.demo_pose[k].clone(), quat_conjugate(getattr(
                                    self, k.replace('_demo', '')))))

    def _refresh_demo_pose(self):
        if hasattr(self, "demo_pose"):
            for k, v in self.demo_pose.items():
                # Transform values relative to the drill to absolute values via
                # the current drill pose

                # Transform keypoint poses
                if k.startswith(tuple(self.keypoint_dict.keys())):
                    num_keypoints = v.shape[1]
                    if "_pos" in k:
                        getattr(self, k)[:] = \
                            self.drill_pos.unsqueeze(1).repeat(
                                1, num_keypoints, 1) + quat_apply(
                                self.drill_quat.unsqueeze(1).repeat(
                                    1,  num_keypoints, 1), v)
                    elif "_quat" in k:
                        getattr(self, k)[:] = quat_mul(
                            self.drill_quat.unsqueeze(1).repeat(
                                1, num_keypoints, 1), v)

                # Transform ik_body pose
                elif k.startswith("ik_body"):
                    if "_pos" in k:
                        getattr(self, k)[:] = self.drill_pos + quat_apply(
                            self.drill_quat, v)
                    elif "_quat" in k:
                        getattr(self, k)[:] = quat_mul(
                            self.drill_quat, v)

                # Update relative positions and rotations
                if hasattr(self, k.replace('_demo', '')):
                    if '_pos' in k:
                        getattr(self, 'to_' + k)[:] = getattr(self, k) - getattr(
                        self, k.replace('_demo', ''))

                    elif '_quat' in k:
                        getattr(self, 'to_' + k)[:] = quat_mul(
                            getattr(self, k), quat_conjugate(getattr(
                            self, k.replace('_demo', ''))))

    def compute_observations(self):
        """Compute observations."""

        obs_tensors = []
        for observation in self.cfg_task.env.observations:
            # Flatten body dimension for keypoint observations
            if any(k in observation for k in self.keypoint_dict.keys()):
                obs_tensors.append(getattr(self, observation).flatten(1, 2))
            else:
                obs_tensors.append(getattr(self, observation))

        self.obs_buf = torch.cat(obs_tensors, dim=-1)  # shape = (num_envs, num_observations)
        return self.obs_buf

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
        self.rew_buf[:] = 0.

        # Get drill height: Δh
        # delta_lift_off_height is clamped between the initial difference
        # (self.cfg_task.rl.lift_off_height) and the target (0).
        drill_height = self.drill_pos[:, 2]
        drill_height_initial = self.drill_pos_initial[:, 2]
        delta_lift_off_height = torch.clamp(
            self.cfg_task.rl.lift_off_height -
            (drill_height - drill_height_initial),
            min=0, max=self.cfg_task.rl.lift_off_height)

        # Get distance to target position: Δx
        drill_target_pos_dist = torch.norm(
            self.to_drill_target_pos, p=2, dim=1)
        #drill_target_pos_dist_initial = torch.norm(
        #    self.drill_pos_initial - self.drill_target_pos, p=2, dim=1)

        # Get smallest angle to target orientation: Δθ
        drill_target_angle_dist = 2.0 * torch.asin(
            torch.clamp(torch.norm(self.to_drill_target_quat[:, 0:3],
                                   p=2, dim=-1), max=1.0))
        #drill_target_quat_dist_initial = quat_mul(
        #    self.drill_quat_initial,
        #    quat_conjugate(self.drill_target_quat))
        #drill_target_angle_dist_initial = 2.0 * torch.asin(
        #    torch.clamp(torch.norm(
        #        drill_target_quat_dist_initial[:, 0:3],
        #        p=2, dim=-1), max=1.0))

        # Check whether the target pose has been reached
        target_pos_reached = \
            drill_target_pos_dist < self.cfg_task.rl.target_pos_threshold
        target_angle_reached = \
            drill_target_angle_dist < self.cfg_task.rl.target_angle_threshold
        target_pose_reached = torch.logical_and(
            target_pos_reached, target_angle_reached)

        # Get distance to demo ik_body position: Δx
        ik_body_demo_pos_dist = torch.norm(
            self.ik_body_pos - self.ik_body_demo_pos, p=2, dim=1)
        # Get smallest angle to demo ik_body orientation: Δθ
        ik_body_demo_quat_dist = quat_mul(
            self.ik_body_quat, quat_conjugate(self.ik_body_demo_quat))
        ik_body_demo_angle_dist = 2.0 * torch.asin(
            torch.clamp(torch.norm(ik_body_demo_quat_dist[:, 0:3],
                                   p=2, dim=-1), max=1.0))
        # Check whether demo ik_body pose has been reached
        ik_body_pose_reached = torch.logical_and(
            ik_body_demo_pos_dist < 0.045,
            ik_body_demo_angle_dist < 0.4)
        close_to_ik_body_pose = sq_hyperbole_rew(
            1.0, 6 * ik_body_demo_pos_dist) * sq_hyperbole_rew(
            1.0, 1 * ik_body_demo_angle_dist)



        # Check whether the thumb is under the drill
        thumb_to_drill = self.fingertips_pos[:, 4] - self.drill_pos
        thumb_to_drill_tool_coordinates = quat_rotate_inverse(
            self.drill_quat, thumb_to_drill)
        thumb_under_drill = thumb_to_drill_tool_coordinates[:, 1] > 0.

        # Check whether the target dof pos has been reached
        dof_pos_dist = torch.abs(
            self.residual_actuated_dof_demo_pos -
            self.dof_pos[:, self.residual_actuated_dof_indices])
        dof_pos_reached = torch.all(dof_pos_dist < 0.4, dim=1)

        reward_terms = {}
        for reward_term, scale in self.cfg_task.rl.reward.items():
            # Penalize distance of keypoint groups to pre-recorded pose
            if reward_term.startswith(tuple(self.keypoint_dict.keys())):
                keypoint_group_name = '_'.join(reward_term.split('_')[:-2])
                keypoint_dist = torch.norm(
                    getattr(self, 'to_' + keypoint_group_name + '_demo_pos'),
                    dim=2)

                mean_keypoint_dist = keypoint_dist.mean(1)

                # This is a distance penalty to the demonstrated keypoint pose
                if reward_term.endswith('dist_penalty'):
                    # keypoint reward is only used once the ik_body_pose is reached
                    #reward = 0.025 * ik_body_pose_reached * torch.sum(keypoint_dist < 0.04, dim=-1)
                    reward = ik_body_pose_reached * exponential_rew(
                        scale, mean_keypoint_dist, c=8)

                # This term rewards lift-off if the keypoint pose is satisfied
                elif reward_term.endswith('based_liftoff'):
                    keypoint_pose_reached = torch.all(
                        keypoint_dist < 0.045, dim=1)
                    reward = keypoint_pose_reached * exponential_rew(
                        scale, delta_lift_off_height, c=5)

                else:
                    assert False

            elif reward_term == 'liftoff_reward':
                delta_ik_body_pos_height = torch.clamp(
                    self.cfg_task.rl.lift_off_height - (self.ik_body_pos[:, 2] -0.1), min=0.)
                #print("dof_pos_reached:", dof_pos_reached)
                #print("ik_body_pose_reached:", ik_body_pose_reached)
                reward = torch.logical_and(ik_body_pose_reached, dof_pos_reached) * exponential_rew(
                    scale, delta_ik_body_pos_height, c=5)

            elif reward_term == 'around_handle_reward':
                reward = scale * thumb_to_drill_tool_coordinates[:, 1]
                #reward = scale * ik_body_pose_reached * thumb_under_drill

            elif reward_term == 'ik_body_pos_dist_penalty':
                reward = -scale * ik_body_demo_pos_dist

            elif reward_term == 'ik_body_quat_dist_penalty':
                reward = -scale * ik_body_demo_angle_dist

            elif reward_term == 'ik_body_pose_reached_bonus':
                reward = scale * ik_body_pose_reached

            elif reward_term == 'dof_dist_penalty':
                residual_actuated_dof_open_pos = torch.Tensor([[-1.571, 0, 0, 0, 0]]).repeat(self.num_envs, 1).to(self.device)
                close_to_ik_body_pose_expanded = close_to_ik_body_pose.unsqueeze(1).repeat(1, 5)
                #print("close_to_ik_body_pose.shape:", close_to_ik_body_pose.shape)
                #print("self.residual_actuated_dof_demo_pos.shape:", self.residual_actuated_dof_demo_pos.shape)
                #print("residual_actuated_dof_open_pos.shape:", residual_actuated_dof_open_pos.shape)
                residual_actuated_dof_target_pos = close_to_ik_body_pose_expanded * self.residual_actuated_dof_demo_pos + (1 - close_to_ik_body_pose_expanded) * residual_actuated_dof_open_pos

                residual_actuated_dof_dist = torch.abs(
                    residual_actuated_dof_target_pos -
                    self.dof_pos[:, self.residual_actuated_dof_indices])
                mean_residual_actuated_dof_dist = \
                    residual_actuated_dof_dist.mean(dim=-1)
                reward = scale * -mean_residual_actuated_dof_dist

            elif reward_term == 'dof_pos_reached_bonus':
                reward = scale * dof_pos_reached * ik_body_pose_reached


            # Penalize large actions
            elif reward_term == 'action_penalty':
                squared_action_norm = torch.linalg.norm(
                    self.actions, dim=-1)
                reward = - squared_action_norm * scale

            # Reward progress towards target position
            elif reward_term == 'target_pos_dist_penalty':
                reward = torch.logical_and(
                    ik_body_pose_reached, dof_pos_reached) * \
                         hyperbole_rew(
                        scale, drill_target_pos_dist, c=0.05, pow=1)

            # Reward progress towards target orientation
            elif reward_term == 'target_quat_dist_penalty':
                reward = torch.logical_and(
                    ik_body_pose_reached, dof_pos_reached) * \
                         hyperbole_rew(
                             scale, drill_target_angle_dist, c=0.05, pow=1)

            # Reward reaching the target pose
            elif reward_term == 'success_bonus':
                reward = scale * target_pose_reached

            else:
                assert False, f"Unknown reward term {reward_term}."

            self.rew_buf[:] += reward
            reward_terms["reward_terms/" + reward_term] = reward.mean()

        if "reward_terms" in self.cfg_base.logging.keys():
            self.log(reward_terms)

        print("reward_terms:", reward_terms)

    def reset_idx(self, env_ids):
        """Reset specified environments."""

        if self.drills_dropped:
            self._reset_drill(env_ids, apply_reset=False)
        else:
            self._drop_drill(
                env_ids,
                sim_steps=self.cfg_task.randomize.num_drill_drop_steps)
            self.drills_dropped = True

        self._reset_robot(env_ids, apply_reset=True)

        self._reset_target_pose(env_ids, apply_reset=True)

        # self._randomize_ik_body_pose(
        #    env_ids,
        #    sim_steps=self.cfg_task.randomize.num_ik_body_initial_move_steps)

        self._reset_buffers(env_ids)

    def _reset_drill(self, env_ids, apply_reset: bool = True):
        """Reset root states of the drill."""

        self.root_pos[env_ids, self.drill_actor_id_env] = \
            self.drill_pos_initial[env_ids]
        self.root_quat[env_ids, self.drill_actor_id_env] = \
            self.drill_quat_initial[env_ids]
        self.root_linvel[env_ids, self.drill_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.drill_actor_id_env] = 0.0

        # Set actor root state tensor
        if apply_reset:
            drill_indices = self.drill_actor_ids_sim[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_state),
                gymtorch.unwrap_tensor(drill_indices),
                len(drill_indices))

    def _drop_drill(self, env_ids, sim_steps: int):

        if self.cfg_base.debug.verbose:
            print("Drills must still be dropped in envs:",
                  env_ids.cpu().numpy())

        # Randomize drop position of drill
        drill_pos_drop = self._get_random_drop_pos(env_ids)
        drill_quat_drop = torch.tensor(
            [[-0.707, 0, 0, 0.707]], dtype=torch.float,
            device=self.device).repeat(self.num_envs, 1)
        #drill_quat_drop = self._get_random_drop_quat(env_ids)

        # Set root state tensor of the simulation
        self.root_pos[env_ids, self.drill_actor_id_env] = drill_pos_drop
        self.root_quat[env_ids, self.drill_actor_id_env] = drill_quat_drop
        self.root_linvel[env_ids, self.drill_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.drill_actor_id_env] = 0.0

        drill_indices = self.drill_actor_ids_sim[env_ids].to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(drill_indices),
            len(drill_indices))

        # Step simulation to drop drills
        for _ in range(sim_steps):
            self.gym.simulate(self.sim)
            self.render()
            if len(self.cfg_base.debug.visualize) > 0 \
                    and not self.cfg['headless']:
                self.refresh_base_tensors()
                self.refresh_env_tensors()
                self._refresh_task_tensors()
                self.gym.clear_lines(self.viewer)
                self.draw_visualizations(self.cfg_base.debug.visualize)

        # Refresh tensor and set initial object poses
        self.refresh_base_tensors()
        self.drill_pos_initial = self.root_pos[
            :, self.drill_actor_id_env].detach().clone()
        self.drill_quat_initial = self.root_quat[
            :, self.drill_actor_id_env].detach().clone()

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

    def _reset_buffers(self, env_ids):
        """Reset buffers."""

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def visualize_ik_body_demo_pose(self, env_id: int, axis_length: float = 0.3
                                    ) -> None:
        self.visualize_body_pose("ik_body_demo", env_id, axis_length)
