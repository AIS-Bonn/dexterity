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
        self._get_task_yaml_params()
        self._acquire_task_tensors()
        self.parse_controller_spec()

        if self.cfg_task.sim.disable_gravity:
            self.disable_gravity()

        if self.viewer is not None:
            self._set_viewer_params()

        self.hammers_dropped = False

    def _get_task_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='dexterity_schema_config_task', node=DexteritySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self.cfg)
        self.max_episode_length = self.cfg_task.rl.max_episode_length  # required instance var for VecTask

        ppo_path = 'train/DexterityTaskHammerDriveNailPPO.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo['train']  # strip superfluous nesting

    def _acquire_demo_pose(self):
        demo_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..',
                         '..', 'assets', 'dexterity', 'tools',
                         'hammers', 'canonical',
                         self.robot.manipulator.model_name + '_demo_pose.npz'))
        if os.path.isfile(demo_path):
            demo_pose_npz = np.load(demo_path)
            self.demo_pose = {}
            for k, v in demo_pose_npz.items():
                if k.startswith(tuple(self.keypoint_dict.keys())):
                    self.demo_pose[k + '_demo_relative'] = \
                        torch.from_numpy(v).to(self.device)
                    setattr(self, k + '_demo',
                            torch.from_numpy(v).to(self.device).repeat(
                                self.num_envs, 1, 1))

    def _refresh_demo_pose(self):
        if hasattr(self, "demo_pose"):
            for k, v in self.demo_pose.items():
                getattr(self, k[:-9])[:] = \
                    self.hammer_pos.unsqueeze(1).repeat(1, v.shape[1], 1) + \
                    quat_apply(
                    self.hammer_quat.unsqueeze(1).repeat(1, v.shape[1], 1),
                    v.repeat(self.num_envs, 1, 1))

    def _acquire_task_tensors(self):
        """Acquire tensors."""
        self._acquire_demo_pose()

    def _refresh_task_tensors(self):
        """Refresh tensors."""
        self._refresh_demo_pose()

    def compute_observations(self):
        """Compute observations."""

        obs_tensors = []
        for observation in self.cfg_task.env.observations:
            # Flatten body dimension for keypoint observations
            if observation.startswith(tuple(self.keypoint_dict.keys())):
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

    def _update_rew_buf(self):
        """Compute reward at current timestep."""
        self.rew_buf[:] = 0.
        nail_depth = self.dof_pos[:, -1]
        nail_driven = nail_depth < self.cfg_task.rl.target_nail_depth

        reward_dict = {}
        for reward_term, scale in self.cfg_task.rl.reward.items():
            # Penalize distance of keypoint groups to object
            if reward_term.startswith(tuple(self.keypoint_dict.keys())):
                keypoint_group_name = '_'.join(reward_term.split('_')[:-3])
                assert reward_term.endswith('_dist_penalty'), \
                    f"Reward term {reward_term} depends on " \
                    f"keypoint_group {keypoint_group_name}, but is not " \
                    f"a distance penalty."

                assert reward_term.split('_')[-3] == "imitation", \
                    f"Reward term {reward_term} should be an imitation loss."

                assert f'{keypoint_group_name}_pos' in self.cfg['env']['observations'], \
                    f"Cannot use imitation loss on keypoint group " \
                    f"{keypoint_group_name} poses if " \
                    f"'{keypoint_group_name}_pos' is not part of the " \
                    f"observations."

                keypoint_pos = getattr(self, keypoint_group_name + '_pos')
                keypoint_pos_demo = getattr(self, keypoint_group_name + '_pos_demo')

                keypoint_dist_demo = torch.sum(torch.norm(
                    keypoint_pos - keypoint_pos_demo, dim=2), dim=1)
                reward = -keypoint_dist_demo * scale

            # Penalize large actions
            elif reward_term == 'action_penalty':
                action_norm = torch.norm(self.actions, p=2, dim=-1)
                reward = - action_norm * scale

            # Reward driving the nail
            elif reward_term == 'nail_depth_reward':
                reward = -nail_depth * self.cfg_task.rl.reward.nail_depth_reward

            # Reward reaching the target height
            elif reward_term == 'success_bonus':
                reward = scale * nail_driven

            else:
                assert False, f"Unknown reward term {reward_term}."

            self.rew_buf[:] += reward
            reward_dict[reward_term] = reward.mean()
        self.log(reward_dict)
        #print("reward_dict:", reward_dict)

    def reset_idx(self, env_ids):
        """Reset specified environments."""

        if self.hammers_dropped:
            self._reset_hammer(env_ids, apply_reset=False)
        else:
            self._drop_hammer(
                env_ids,
                sim_steps=self.cfg_task.randomize.num_hammer_drop_steps)
            self.hammers_dropped = True

        self._reset_robot(env_ids, apply_reset=False)
        self._reset_nail(env_ids)

        #self._randomize_ik_body_pose(
        #    env_ids,
        #    sim_steps=self.cfg_task.randomize.num_ik_body_initial_move_steps)

        self._reset_buffers(env_ids)

    def _reset_hammer(self, env_ids, apply_reset: bool = True):
        """Reset root states of the hammer."""

        self.root_pos[env_ids, self.hammer_actor_id_env] = \
            self.hammer_pos_initial[env_ids]
        self.root_quat[env_ids, self.hammer_actor_id_env] = \
            self.hammer_quat_initial[env_ids]
        self.root_linvel[env_ids, self.hammer_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.hammer_actor_id_env] = 0.0

        # Set actor root state tensor
        if apply_reset:
            hammer_indices = self.hammer_actor_ids_sim[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_state),
                gymtorch.unwrap_tensor(hammer_indices),
                len(hammer_indices))

    def _drop_hammer(self, env_ids, sim_steps: int):
        all_env_ids = env_ids.clone()
        hammer_dropped_successfully = torch.zeros(
            (self.num_envs,), dtype=torch.bool, device=self.device)

        # Init buffers for the initial pose object will be reset to
        self.hammer_pos_initial = self.root_pos[
            all_env_ids, self.hammer_actor_id_env].detach().clone()
        self.hammer_quat_initial = self.root_quat[
            all_env_ids, self.hammer_actor_id_env].detach().clone()

        while not torch.all(hammer_dropped_successfully):
            # Check for which env_ids we need to drop the object
            env_ids = torch.masked_select(
                all_env_ids, ~hammer_dropped_successfully)

            if self.cfg_base.debug.verbose:
                print("Hammers must still be dropped in envs:",
                      env_ids.cpu().numpy())

            # Randomize drop position and orientation of object
            hammer_pos_drop = self._get_random_drop_pos(env_ids)
            hammer_quat_drop = self._get_random_drop_quat(env_ids)

            # Set root state tensor of the simulation
            self.root_pos[env_ids, self.hammer_actor_id_env] = hammer_pos_drop
            self.root_quat[env_ids, self.hammer_actor_id_env] = hammer_quat_drop
            self.root_linvel[env_ids, self.hammer_actor_id_env] = 0.0
            self.root_angvel[env_ids, self.hammer_actor_id_env] = 0.0

            hammer_indices = self.hammer_actor_ids_sim[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_state),
                gymtorch.unwrap_tensor(hammer_indices),
                len(hammer_indices))

            # Step simulation to drop hammers
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
            self.hammer_pos_initial[env_ids] = self.root_pos[
                env_ids, self.hammer_actor_id_env].detach().clone()
            self.hammer_quat_initial[env_ids] = self.root_quat[
                env_ids, self.hammer_actor_id_env].detach().clone()

            hammer_dropped_successfully = self._hammer_in_workspace(
                self.hammer_pos_initial)

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

    def _reset_buffers(self, env_ids):
        """Reset buffers."""

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
