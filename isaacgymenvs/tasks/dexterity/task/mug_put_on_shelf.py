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


class DexterityTaskMugHang(DexterityEnvMug, DexterityABCTask):

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

        ppo_path = 'train/DexterityTaskMugHangPPO.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo['train']  # strip superfluous nesting

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

        tool_grasping_reward, reward_terms = self._compute_tool_grasping_reward()

        if "reward_terms" in self.cfg_base.logging.keys():
            self.log(reward_terms)

        self.rew_buf[:] = tool_grasping_reward

    def reset_idx(self, env_ids):
        """Reset specified environments."""

        if self.mugs_dropped:
            self.reset_tool(env_ids, apply_reset=False)
        else:
            self.drop_tool(
                env_ids,
                sim_steps=self.cfg_task.randomize.num_mug_drop_steps)
            self.mugs_dropped = True

        self._reset_robot(env_ids, apply_reset=True)
        self._reset_buffers(env_ids)
