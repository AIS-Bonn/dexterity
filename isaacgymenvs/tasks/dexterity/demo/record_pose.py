# Copyright (c) 2018-2022, NVIDIA Corporation
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

import datetime
import isaacgym

import os
from omegaconf import OmegaConf
from hydra import compose, initialize
import gym

from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
import isaacgymenvs
from isaacgymenvs.tasks.dexterity.demo.src.vr_vec_task import VRVecTask
import numpy as np
from pynput import keyboard
from isaacgym.torch_utils import *
from isaacgymenvs.utils import torch_jit_utils

space_pressed = False


def on_press(key):
    if key == keyboard.Key.space:
        global space_pressed
        space_pressed = True


def record_keypoint_pose(task: str = "DexterityTaskMugPutOnShelf"):
    initialize(config_path="../../../cfg/")
    cfg = compose(config_name="config",
                  overrides=["num_envs=1", "headless=False", "sim_device=cpu",
                             f"task={task}", "seed=3",
                             "task.rl.max_episode_length=1000"])

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{cfg.wandb_name}_{time_str}"

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    if cfg.task.name == "DexterityTaskDrillPickAndPlace":
        tool_name = "drill"
    elif cfg.task.name == "DexterityTaskHammerDriveNail":
        tool_name = "hammer"
    elif cfg.task.name == "DexterityTaskMugPutOnShelf":
        tool_name = "mug"
    elif cfg.task.name == "DexterityTaskObjectLift":
        tool_name = "object"
    elif cfg.task.name == "DexterityTaskBinPick":
        tool_name = "target_object"
    else:
        assert False

    # set numpy formatting for printing only
    set_np_formatting()

    rank = int(os.getenv("LOCAL_RANK", "0"))

    # sets seed. if seed is -1 will pick a random one
    cfg.seed += rank
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=rank)

    if cfg.wandb_activate and rank == 0:
        # Make sure to install WandB if you actually use this.
        import wandb

        run = wandb.init(
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            entity=cfg.wandb_entity,
            config=cfg_dict,
            sync_tensorboard=True,
            name=run_name,
            resume="allow",
            monitor_gym=True,
        )

    experiment_dir = os.path.join('runs', cfg.train.params.config.name)
    os.makedirs(experiment_dir, exist_ok=True)

    def create_env_thunk(**kwargs):
        envs = isaacgymenvs.make(
            cfg.seed,
            cfg.task_name,
            cfg.task.env.numEnvs,
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )
        if cfg.capture_video:
            envs.is_vector_env = True
            envs = gym.wrappers.RecordVideo(
                envs,
                f"videos/{run_name}",
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        return envs

    env = create_env_thunk()

    # Check for keyboard events
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()

    # Wrap env to interact with VR environment
    env = VRVecTask(env)

    done = False
    obs = env.reset()
    act = torch.zeros((1, env.cfg["env"]["numActions"])).to(env.device)
    relative_keypoints = {}
    while True:
        global space_pressed
        if space_pressed:
            relative_ik_body_pos_in_world_coordinates = env.ik_body_pos - getattr(env, tool_name + "_pos").clone()
            relative_ik_body_pos_in_tool_coordinates = quat_rotate_inverse(
                getattr(env, tool_name + '_quat'),
                relative_ik_body_pos_in_world_coordinates)
            relative_ik_body_quat = quat_mul(quat_conjugate(getattr(env, tool_name + '_quat')), env.ik_body_quat)

            np_save_dict = {
                "ik_body_demo_pos": relative_ik_body_pos_in_tool_coordinates.cpu().numpy(),
                "ik_body_demo_quat": relative_ik_body_quat.cpu().numpy(),
                "residual_actuated_dof_demo_pos": env.dof_pos[:, env.residual_actuated_dof_indices]
            }

            for keypoint_group in env.keypoint_dict.keys():
                relative_pos_in_world_coordinates = getattr(env, keypoint_group + "_pos")[0] - getattr(env, tool_name + "_pos").clone()[0]
                relative_pos_in_tool_coordinates = quat_rotate_inverse(getattr(env, tool_name + '_quat').unsqueeze(1).repeat(1, getattr(env, keypoint_group + "_pos").shape[1], 1)[0], relative_pos_in_world_coordinates).unsqueeze(0)
                relative_keypoints[keypoint_group] = relative_pos_in_tool_coordinates
                np_save_dict[keypoint_group + "_demo_pos"] = relative_pos_in_tool_coordinates.cpu().numpy()

                relative_quat = quat_mul(
                    quat_conjugate(getattr(env, tool_name + '_quat').unsqueeze(0).repeat(1, getattr(env, keypoint_group + "_quat").shape[1], 1)),
                    getattr(env, keypoint_group + "_quat"))
                np_save_dict[keypoint_group + "_demo_quat"] = relative_quat.cpu().numpy()

            asset_root = os.path.normpath(
                os.path.join(os.path.dirname(__file__), '..', '..', '..', '..',
                             'assets', 'dexterity', 'tools', tool_name))
            asset_file = env.cfg_env['env']['canonical']
            tool_dir = os.path.dirname(os.path.join(asset_root, asset_file))
            np.savez(os.path.join(tool_dir, env.robot.manipulator.model_name +
                                  '_demo_pose.npz'),
                     **np_save_dict)
            print("Pose saved to ", os.path.join(
                tool_dir, env.robot.manipulator.model_name + '_demo_pose.npz'))
            space_pressed = False

        obs, rew, done, info = env.step(act)

        for k, v in relative_keypoints.items():
            if k + "_pose" in env.cfg_base.debug.visualize:
                kp_pos = getattr(env, tool_name + "_pos").unsqueeze(1).repeat(
                    1, v.shape[1], 1) + quat_apply(
                    getattr(env, tool_name + "_quat").unsqueeze(
                    1).repeat(1, v.shape[1], 1), v)
                env.visualize_recorded_keypoint_pose(kp_pos, 0)

    # dump config dict
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))

    if cfg.wandb_activate and rank == 0:
        wandb.finish()


if __name__ == "__main__":
    record_keypoint_pose()
