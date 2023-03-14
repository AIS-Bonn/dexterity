import isaacgym
from isaacgym import gymapi
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.dexterity.base.base import DexterityBase
from isaacgymenvs.tasks.dexterity.env.schema_class_env import DexterityABCEnv
from isaacgymenvs.tasks.dexterity.env.schema_config_env import \
    DexteritySchemaConfigEnv

from isaacgymenvs.tasks.dexterity.env.tool_utils import DexterityCategory
from .object import randomize_rotation
from isaacgymenvs.tasks.dexterity.task.task_utils import *
from isaacgym import gymapi, gymtorch, torch_utils
from isaacgymenvs.tasks.dexterity.task.schema_config_task import \
    DexteritySchemaConfigTask
import omegaconf
import math
from typing import *
import isaacgymenvs.tasks.dexterity.dexterity_control as ctrl

import hydra
import os


class DexterityEnvToolUse(DexterityBase, DexterityABCEnv):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless,
                 virtual_screen_capture, force_render, visual_tracking=False,
                 tool_category="drill"):
        """Initialize instance variables. Initialize environment superclass.
        Acquire tensors."""
        self.visual_tracking = visual_tracking
        self.shapes_estimated = False
        self.tool_category = tool_category
        self._get_env_yaml_params()
        self.parse_camera_spec(cfg)

        super().__init__(cfg, rl_device, sim_device, graphics_device_id,
                         headless, virtual_screen_capture, force_render)

        self.acquire_base_tensors()  # defined in superclass
        self._acquire_env_tensors()
        self.refresh_base_tensors()  # defined in superclass
        self.refresh_env_tensors()

    def _get_env_yaml_params(self):
        """Initialize instance variables from YAML files."""
        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='dexterity_schema_config_env',
                 node=DexteritySchemaConfigEnv)

        config_path = f'task/DexterityEnv{self.tool_category.capitalize()}.yaml'  # relative to cfg dir
        self.cfg_env = hydra.compose(config_name=config_path)
        self.cfg_env = self.cfg_env['task']  # strip superfluous nesting

    def _get_task_yaml_params(self, task_name: str):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='dexterity_schema_config_task', node=DexteritySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self.cfg)
        self.max_episode_length = self.cfg_task.rl.max_episode_length  # required instance var for VecTask

        ppo_path = f'train/DexterityTask{task_name}PPO.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo['train']  # strip superfluous nesting

    def _import_tool_assets(self):
        asset_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '../..', '..', '..',
                         'assets', 'dexterity', 'tools', self.tool_category))
        source_asset_file = self.cfg_env['env']['canonical']
        target_asset_files = self.cfg_env['env'][self.tool_category + "s"]

        source_asset_file += '/rigid.urdf'
        target_asset_files = [t + '/rigid.urdf' for t in target_asset_files]

        demo_path = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..',
                         '..', 'assets', 'dexterity', 'tools',
                         self.tool_category, self.cfg_env['env']['canonical'],
                         self.robot.manipulator.model_name + '_demo_pose.npz'))

        self.category_space = DexterityCategory(
            self.gym, self.sim, asset_root, source_asset_file,
            target_asset_files, demo_path)

        load_path = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..',
                     'tasks', 'dexterity', 'demo', 'category_space',
                     self.tool_category + '_category'))

        if os.path.isfile(load_path + '_latent_space.csv'):
            print(f"Loading category space from {load_path} ...")
            self.category_space.from_file(load_path, test=self.visual_tracking)

        else:
            print(f"Load path {load_path} not found. Building shape space ...")
            self.category_space.build()
            self.category_space.to_file(load_path)

        return self.category_space.source_instance, self.category_space.target_instances

    def _get_random_drop_pos(self, env_ids) -> torch.Tensor:
        pos_drop = torch.tensor(
            getattr(self.cfg_task.randomize, f"{self.tool_category}_pos_drop"),
            device=self.device).unsqueeze(0).repeat(len(env_ids), 1)
        pos_drop_noise = \
            2 * (torch.rand((len(env_ids), 3), dtype=torch.float32,
                            device=self.device) - 0.5)  # [-1, 1]
        pos_drop_noise = pos_drop_noise @ torch.diag(torch.tensor(
            getattr(self.cfg_task.randomize,
                    f"{self.tool_category}_pos_drop_noise"),
            device=self.device))
        pos_drop += pos_drop_noise
        return pos_drop

    def _get_random_drop_quat(self, env_ids) -> torch.Tensor:
        x_unit_tensor = to_torch(
            [1, 0, 0], dtype=torch.float, device=self.device).repeat(
            (len(env_ids), 1))
        y_unit_tensor = to_torch(
            [0, 1, 0], dtype=torch.float, device=self.device).repeat(
            (len(env_ids), 1))
        rand_floats = torch_rand_float(
            -1.0, 1.0, (len(env_ids), 2), device=self.device)
        quat_drop = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1], x_unit_tensor, y_unit_tensor)
        return quat_drop

    def _get_random_drop_quat_zaxis(self, env_ids, z_angle_range: float) -> torch.Tensor:
        yaw = 2 * (torch.rand(len(env_ids), device=self.device) - 0.5) * z_angle_range
        roll = torch.zeros_like(yaw)
        pitch = torch.zeros_like(yaw)
        quat = quat_from_euler_xyz(roll, pitch, yaw)
        return quat

    def _acquire_env_tensors(self):
        """Acquire and wrap tensors. Create views."""

        self.tool_picked_up_once = torch.zeros(self.num_envs,
                                               dtype=torch.bool,
                                               device=self.device)
        self.succeeded_once = torch.zeros(self.num_envs,
                                               dtype=torch.bool,
                                               device=self.device)

        # Acquire tool pose and velocities
        tool_actor_id_env = getattr(self, f"{self.tool_category}_actor_id_env")
        setattr(self, f"{self.tool_category}_pos",
                self.root_pos[:, tool_actor_id_env, 0:3])
        setattr(self, f"{self.tool_category}_quat",
                self.root_quat[:, tool_actor_id_env, 0:4])
        setattr(self, f"{self.tool_category}_linvel",
                self.root_linvel[:, tool_actor_id_env, 0:3])
        setattr(self, f"{self.tool_category}_angvel",
                self.root_angvel[:, tool_actor_id_env, 0:3])

    def refresh_env_tensors(self):
        pass

    def _compute_proprioceptive_observations(self):
        # Estimate and overwrite tool pose
        if self.visual_tracking:
            if not self.shapes_estimated:
                self.estimate_observed_shapes()
                self.shapes_estimated = True

            tool_pos, tool_quat = self.estimate_observed_poses()



            #setattr(self, f"{self.tool_category}_pos",
            #        0.0 * getattr(self, f"{self.tool_category}_pos"))

        # Keep ground truth pose
        else:
            pass

        super()._compute_proprioceptive_observations()

    def get_segmented_and_tool_pointclouds(self, cameras: Tuple[str]):
        segmented_pointclouds = []
        for camera_name in cameras:
            segmented_pointclouds.append(self.obs_dict['image'][camera_name])

        tool_pointclouds = []
        for env_id in range(self.num_envs):
            tool_pointcloud = []
            for camera_idx in range(len(segmented_pointclouds)):
                is_on_tool = (segmented_pointclouds[camera_idx][
                              env_id, :, 3] == 2.0).unsqueeze(-1).repeat(1, 4)
                camera_tool_pointcloud = torch.masked_select(
                    segmented_pointclouds[camera_idx][env_id],
                    is_on_tool).view(-1, 4)[:, :3]
                tool_pointcloud.append(camera_tool_pointcloud)
            tool_pointcloud = torch.cat(tool_pointcloud)
            tool_pointclouds.append(tool_pointcloud)
        return segmented_pointclouds, tool_pointclouds

    def estimate_observed_shapes(
            self,
            cameras: Tuple[str] = ('left_sideview', ),
            visualize: bool = True
    ) -> None:

        segmented_pointclouds, tool_pointclouds = \
            self.get_segmented_and_tool_pointclouds(cameras)

        if visualize:
            import plotly.graph_objects as go
            env_id_to_show = 0
            # Visualize segmented pointclouds of the individual cameras
            scene_scatters = []
            for i, camera_name in enumerate(cameras):
                scene_scatters.append(go.Scatter3d(
                    x=segmented_pointclouds[i][env_id_to_show, :, 0].cpu().numpy(),
                    y=segmented_pointclouds[i][env_id_to_show, :, 1].cpu().numpy(),
                    z=segmented_pointclouds[i][env_id_to_show, :, 2].cpu().numpy(),
                    mode='markers',
                    marker=dict(color=segmented_pointclouds[i][env_id_to_show, :, 3].cpu().numpy(), size=3),
                    name=camera_name
                ))
            tool_scatter = go.Scatter3d(
                x=tool_pointclouds[env_id_to_show][:, 0].cpu().numpy(),
                y=tool_pointclouds[env_id_to_show][:, 1].cpu().numpy(),
                z=tool_pointclouds[env_id_to_show][:, 2].cpu().numpy(),
                mode='markers',
                marker=dict(size=3),
                name='tool_pointcloud'
            )

            pointcloud_fig = go.Figure(data=[tool_scatter, ] + scene_scatters, )
            pointcloud_fig.update_layout(
                scene=dict(
                    xaxis=dict(range=[-0.75, 0.75], ),
                    yaxis=dict(range=[-0.75, 0.75], ),
                    zaxis=dict(range=[-0.05, 1.45], ), ),
            )
            pointcloud_fig.show()

        self.category_space.fit_shape_to_observed(
            tool_pointclouds, getattr(self, f"{self.tool_category}_pos"),
            getattr(self, f"{self.tool_category}_quat"))

    def estimate_observed_poses(
            self,
            cameras: Tuple[str] = ('left_sideview', 'right_sideview'),
            visualize: bool = False
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        segmented_pointclouds, tool_pointclouds = \
            self.get_segmented_and_tool_pointclouds(cameras)

        tool_pos, tool_quat = self.category_space.fit_pose_to_observed(
            tool_pointclouds)

        print("Ground truth tool pos:", getattr(self, f"{self.tool_category}_pos"))
        print("Ground truth tool quat:", getattr(self, f"{self.tool_category}_quat"))

        print("Estimated tool pos:", tool_pos)
        print("Estimated tool quat:", tool_quat)

        return tool_pos, tool_quat

    def _import_env_assets(self):
        pass

    def _acquire_task_tensors(self):
        """Acquire tensors."""
        self._acquire_demo_pose()
        self._acquire_synthetic_pointclouds()

    def _refresh_task_tensors(self):
        """Refresh tensors."""
        self._refresh_demo_pose()
        self._refresh_synthetic_pointclouds()

    def _acquire_demo_pose(self):
        collected_demo_poses = {k: [] for k in self.category_space.target_instances[0].demo_dict.keys()}

        demo_poses = []
        for tool_instance in [self.category_space.source_instance, ] + self.category_space.target_instances:
            demo_poses.append(tool_instance.demo_dict)
            for k in collected_demo_poses.keys():
                if self.cfg_task.ablation == 'disable_grasp_pose_generalization':
                    collected_demo_poses[k].append(self.category_space.source_instance.demo_dict[k])
                else:
                    collected_demo_poses[k].append(tool_instance.demo_dict[k])

        for k, v in collected_demo_poses.items():
            collected_demo_poses[k] = np.stack(v)

        num_repeats = math.ceil(self.num_envs / (len(self.category_space.target_instances) + 1))

        self.demo_pose = {}
        for k, v in collected_demo_poses.items():
            # Store demo pose of keypoints relative to the tool
            # (shape: [num_envs, num_keypoints, 3 or 4]
            if k.startswith(tuple(self.keypoint_dict.keys())):
                self.demo_pose[k] = torch.from_numpy(v).to(
                    self.device, dtype=torch.float32).repeat(
                    num_repeats, 1, 1)[:self.num_envs]
                setattr(self, k, self.demo_pose[k].clone())

            # Store ik_body pose and residual dof pos
            # (shape: [num_envs, 3 or 4 or num_residual_actuated_dofs]
            else:
                self.demo_pose[k] = torch.from_numpy(v).to(
                    self.device, dtype=torch.float32).repeat(
                    num_repeats, 1)[:self.num_envs]
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

        self.demo_pose['pre_grasp_pos'] = self.demo_pose['ik_body_demo_pos'].clone()
        self.demo_pose['pre_grasp_quat'] = self.demo_pose['ik_body_demo_quat'].clone()
        self.demo_pose['pre_grasp_pos'][:] += quat_apply(
                            self.demo_pose['pre_grasp_quat'], torch.Tensor([[-0.05, 0., 0.07]]).repeat(self.num_envs, 1).to(self.device))

        self.pre_grasp_pos = self.demo_pose['pre_grasp_pos'].clone()
        self.pre_grasp_quat = self.demo_pose['pre_grasp_quat'].clone()

    def _acquire_synthetic_pointclouds(self):
        synthetic_pointclouds = []
        for tool_instance in [self.category_space.source_instance, ] + self.category_space.target_instances:
            synthetic_pointclouds.append(tool_instance.synthetic_pointcloud)

        synthetic_pointclouds = np.stack(synthetic_pointclouds)
        num_repeats = math.ceil(self.num_envs / (len(self.category_space.target_instances) + 1))

        self.synthetic_pointclouds_object_coords = torch.from_numpy(synthetic_pointclouds).to(
                self.device, dtype=torch.float32).repeat(num_repeats, 1, 1)[:self.num_envs]
        self.synthetic_pointcloud_pos = torch.zeros_like(self.synthetic_pointclouds_object_coords)

    def _refresh_synthetic_pointclouds(self):
        num_points = self.synthetic_pointcloud_pos.shape[1]
        self.synthetic_pointcloud_pos[:] = getattr(self, f"{self.tool_category}_pos").unsqueeze(1).repeat(
                1, num_points, 1) + quat_apply(getattr(self, f"{self.tool_category}_quat").unsqueeze(1).repeat(1, num_points, 1), 
                        self.synthetic_pointclouds_object_coords)

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
                            getattr(self, f"{self.tool_category}_pos").unsqueeze(1).repeat(
                                1, num_keypoints, 1) + quat_apply(
                                getattr(self, f"{self.tool_category}_quat").unsqueeze(1).repeat(
                                    1,  num_keypoints, 1), v)
                    elif "_quat" in k:
                        getattr(self, k)[:] = quat_mul(
                            getattr(self, f"{self.tool_category}_quat").unsqueeze(1).repeat(
                                1, num_keypoints, 1), v)

                # Transform ik_body pose
                elif k.startswith("ik_body") or k.startswith("pre_grasp"):
                    if "_pos" in k:
                        getattr(self, k)[:] = getattr(self, f"{self.tool_category}_pos") + quat_apply(
                            getattr(self, f"{self.tool_category}_quat"), v)
                    elif "_quat" in k:
                        getattr(self, k)[:] = quat_mul(
                            getattr(self, f"{self.tool_category}_quat"), v)

                # Update relative positions and rotations
                if hasattr(self, k.replace('_demo', '')) and 'pre_grasp' not in k:
                    if '_pos' in k:
                        getattr(self, 'to_' + k)[:] = getattr(self, k) - getattr(
                        self, k.replace('_demo', ''))

                    elif '_quat' in k:
                        getattr(self, 'to_' + k)[:] = quat_mul(
                            getattr(self, k), quat_conjugate(getattr(
                            self, k.replace('_demo', ''))))

    def _compute_tool_grasping_reward(self):
        """Compute tool_grasping reward at current timestep."""
        tool_grasping_reward = torch.zeros_like(self.rew_buf)

        # Get distance to demo ik_body position: Δx
        ik_body_demo_pos_dist = torch.norm(
            self.to_ik_body_demo_pos, p=2, dim=1)
        # Get the smallest angle to demo ik_body orientation: Δθ
        ik_body_demo_angle_dist = 2.0 * torch.asin(
            torch.clamp(torch.norm(self.to_ik_body_demo_quat[:, 0:3],
                                   p=2, dim=-1), max=1.0))
        # Check whether demo ik_body pose has been reached
        ik_body_pose_reached = torch.logical_and(
            ik_body_demo_pos_dist < 0.03,
            ik_body_demo_angle_dist < 0.2)
        close_to_ik_body_pose = sq_hyperbole_rew(
            1.0, 6 * ik_body_demo_pos_dist) * sq_hyperbole_rew(
            1.0, 1 * ik_body_demo_angle_dist)

        # Check whether the target dof pos has been reached
        dof_pos_dist = torch.abs(self.to_residual_actuated_dof_demo_pos)
        dof_pos_reached = torch.all(dof_pos_dist < 0.4, dim=1)

        # Check whether the tool has been picked up
        synthetic_pointcloud_height = self.synthetic_pointcloud_pos[..., 2]
        tool_height = torch.min(synthetic_pointcloud_height, dim=1)[0]

        delta_lift_off_height = torch.clamp(
            self.cfg_task.rl.lift_off_height - tool_height, min=0)
        tool_picked_up = tool_height > self.cfg_task.rl.lift_off_height

        self.tool_picked_up_once = torch.logical_or(self.tool_picked_up_once, tool_picked_up)

        self.log({'tool_picked_up': tool_picked_up.float().mean().item()})

        keypoints_reached_threshold = 0.05
        keypoint_dist = torch.norm(
            getattr(self, 'to_' + 'hand_bodies' + '_demo_pos'),
            dim=2)
        all_keypoints_reached = torch.all(
            keypoint_dist < keypoints_reached_threshold, dim=1)

        reward_terms = {}
        for reward_term, scale in self.cfg_task.rl.reward.items():
            # Penalize distance of keypoint groups to pre-recorded pose.
            if reward_term.startswith(tuple(self.keypoint_dict.keys())):
                keypoint_group_name = '_'.join(reward_term.split('_')[:-2])
                if reward_term.endswith('_dist_penalty'):
                    keypoint_dist = torch.norm(
                    getattr(self, 'to_' + keypoint_group_name + '_demo_pos'),
                        dim=2)
                    keypoint_dist_rews = 1.0 / (0.004 + keypoint_dist ** 2)
                    keypoint_dist_rews *= keypoint_dist_rews
                    keypoint_dist_rews = torch.where(keypoint_dist <= 0.02, keypoint_dist_rews * 2, keypoint_dist_rews)
                    mean_keypoint_dist_rew = keypoint_dist_rews.mean(dim=1)
                    reward = scale * mean_keypoint_dist_rew * ik_body_pose_reached

                    self.log({'mean_keypoint_dist': keypoint_dist.mean().item()})

                elif reward_term.endswith('reached_bonus'):
                    all_keypoints_reached_closer = torch.all(keypoint_dist < 0.5 * keypoints_reached_threshold, dim=1)
                    self.log({'keypoints_reached_threshold': keypoints_reached_threshold})
                    self.log({'all_keypoints_reached': all_keypoints_reached.float().mean().item()})
                    self.log({'all_keypoints_reached_closer': all_keypoints_reached_closer.float().mean().item()})
                    reward = scale * (all_keypoints_reached.float() + all_keypoints_reached_closer.float()) * ik_body_pose_reached
                else:
                    assert False

            # Reward lifting the tool off the ground.
            elif reward_term == 'liftoff_reward':
                reward = ik_body_pose_reached * all_keypoints_reached * (
                        hyperbole_rew(scale, delta_lift_off_height, c=0.02) - hyperbole_rew(
                    scale, self.cfg_task.rl.lift_off_height *
                           torch.ones_like(delta_lift_off_height), c=0.02))

            # Penalize distance to demonstrated ik_body position.
            elif reward_term == 'ik_body_pos_dist_penalty':
                ik_body_pos_dist_rew = 1.0 / (0.02 + ik_body_demo_pos_dist)
                #ik_body_pos_dist_rew *= ik_body_pos_dist_rew
                ik_body_pos_dist_rew = torch.where(ik_body_demo_pos_dist <= 0.04, ik_body_pos_dist_rew * 2, ik_body_pos_dist_rew)
                reward = scale * ik_body_pos_dist_rew

            # Penalize distance to demonstrated ik_body orientation.
            elif reward_term == 'ik_body_quat_dist_penalty':
                ik_body_pos_dist_rew = 2.0 / (0.02 + ik_body_demo_pos_dist)

                ik_body_quat_dist_rew = 1.0 / (
                            0.02 + (0.25 * ik_body_demo_angle_dist))
                #ik_body_quat_dist_rew *= ik_body_quat_dist_rew
                ik_body_quat_dist_rew = torch.where(
                    ik_body_demo_angle_dist <= 0.2,
                    ik_body_quat_dist_rew * 2,
                    ik_body_quat_dist_rew)
                reward = scale * ik_body_quat_dist_rew * 0.01 * ik_body_pos_dist_rew

            # Award a bonus for reaching the demonstrated ik_body pose.
            elif reward_term == 'ik_body_pose_success_bonus':
                reward = scale * ik_body_pose_reached

            # Penalize distance to target DoF position.
            elif reward_term == 'dof_dist_penalty':
                residual_actuated_dof_open_pos = torch.Tensor(
                    [[-1.571, 0, 0, 0, 0]]).repeat(self.num_envs, 1).to(
                    self.device)
                close_to_ik_body_pose_expanded = close_to_ik_body_pose.unsqueeze(
                    1).repeat(1, 5)
                residual_actuated_dof_target_pos = close_to_ik_body_pose_expanded * self.residual_actuated_dof_demo_pos + (
                            1 - close_to_ik_body_pose_expanded) * residual_actuated_dof_open_pos

                residual_actuated_dof_dist = torch.abs(
                    residual_actuated_dof_target_pos -
                    self.dof_pos[:, self.residual_actuated_dof_indices])
                mean_residual_actuated_dof_dist = \
                    residual_actuated_dof_dist.mean(dim=-1)
                reward = exponential_rew(scale, mean_residual_actuated_dof_dist,
                                         c=5) * close_to_ik_body_pose

            # Award a bonus for reaching the demonstrated DoF position.
            elif reward_term == 'dof_pos_success_bonus':
                reward = scale * dof_pos_reached * ik_body_pose_reached

            else:
                continue

            tool_grasping_reward += reward
            reward_terms["reward_terms/" + reward_term] = reward.mean()
        #print("reward_terms:", reward_terms)
        #print("tool_grasping_reward:", tool_grasping_reward)
        #print("self.tool_picked_up_once:", self.tool_picked_up_once)
        return tool_grasping_reward, reward_terms, ik_body_pose_reached, all_keypoints_reached, tool_picked_up

    def reset_tool(self, env_ids, apply_reset: bool = True):
        """Reset root states of the tools."""
        tool_actor_id_env = getattr(self, f"{self.tool_category}_actor_id_env")
        self.root_pos[env_ids, tool_actor_id_env] = \
            getattr(self, f"{self.tool_category}_pos_initial")[env_ids]
        self.root_quat[env_ids, tool_actor_id_env] = \
            getattr(self, f"{self.tool_category}_quat_initial")[env_ids]
        self.root_linvel[env_ids, tool_actor_id_env] = 0.0
        self.root_angvel[env_ids, tool_actor_id_env] = 0.0


        self.tool_picked_up_once[env_ids] = False
        self.succeeded_once[env_ids] = False

        # Set actor root state tensor
        if apply_reset:
            tool_indices = getattr(self, f"{self.tool_category}_actor_ids_sim")[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.root_state),
                gymtorch.unwrap_tensor(tool_indices),
                len(tool_indices))

    def drop_tool(self, env_ids, sim_steps: int):

        if self.cfg_base.debug.verbose:
            print(f"{self.tool_category}s must still be dropped in envs:",
                  env_ids.cpu().numpy())
        tool_actor_id_env = getattr(self, f"{self.tool_category}_actor_id_env")

        # Randomize drop position of tool
        tool_pos_drop = self._get_random_drop_pos(env_ids)

        # Randomize or set drop orientation of tool
        drop_quat = getattr(self.cfg_task.randomize, f"{self.tool_category}_quat_drop")
        tool_quat_drop = torch.tensor(
            [drop_quat], dtype=torch.float,
            device=self.device).repeat(self.num_envs, 1)
        randomized_drop_quat = self._get_random_drop_quat_zaxis(env_ids, z_angle_range=0.5)
        tool_quat_drop = quat_mul(randomized_drop_quat, tool_quat_drop)

        # Set root state tensor of the simulation
        self.root_pos[env_ids, tool_actor_id_env] = tool_pos_drop
        self.root_quat[env_ids, tool_actor_id_env] = tool_quat_drop
        self.root_linvel[env_ids, tool_actor_id_env] = 0.0
        self.root_angvel[env_ids, tool_actor_id_env] = 0.0

        tool_indices = getattr(self, f"{self.tool_category}_actor_ids_sim")[env_ids].to(torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state),
            gymtorch.unwrap_tensor(tool_indices),
            len(tool_indices))

        # Step simulation to drop the tools
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
        setattr(self, f"{self.tool_category}_pos_initial",
                self.root_pos[:, tool_actor_id_env].detach().clone())
        setattr(self, f"{self.tool_category}_quat_initial",
                self.root_quat[:, tool_actor_id_env].detach().clone())

    def move_to_curriculum_pose(self, env_ids, sim_steps: int) -> None:
        """Move ik_body to random pose."""
        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()

        residual_actuated_dof_open_pos = torch.Tensor(
            [[-1.571, 0, 0, 0, 0]]).repeat(self.num_envs, 1).to(
            self.device)

        # Step sim and render
        for _ in range(50):
            pos_error, axis_angle_error = ctrl.get_pose_error(
                self.ik_body_pos, self.ik_body_quat, self.pre_grasp_pos,
                self.pre_grasp_quat, jacobian_type='geometric',
                rot_error_type='axis_angle')

            actions = torch.zeros(
                (self.num_envs, self.cfg_task.env.numActions),
                device=self.device)
            actions[:, 0:3] = pos_error
            actions[:, 3:6] = axis_angle_error
            actions[:, 6:] = residual_actuated_dof_open_pos - self.dof_pos[:, self.residual_actuated_dof_indices]
            actions = torch.clamp(actions, -1, 1)

            self._apply_actions_as_ctrl_targets(
                ik_body_pose_actions=actions[:, 0:6],
                residual_dof_actions=actions[:, 6:],
                do_scale=False)

            self.gym.simulate(self.sim)
            self.render()
            if len(self.cfg_base.debug.visualize) > 0 and self.num_envs < 64:
                self.gym.clear_lines(self.viewer)
                self.draw_visualizations(self.cfg_base.debug.visualize)

            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()
            self.gym.fetch_results(self.sim, True)

            if len(self.cfg_base.debug.visualize) > 0 and self.num_envs < 10:
                self.gym.clear_lines(self.viewer)
                self.draw_visualizations(self.cfg_base.debug.visualize)

        for i in range(100):
            progress = i / 75
            progress = min(1.0, progress)
            pos_error, axis_angle_error = ctrl.get_pose_error(
                self.ik_body_pos, self.ik_body_quat, (1 - progress) * self.pre_grasp_pos + progress * self.ik_body_demo_pos,
                self.ik_body_demo_quat, jacobian_type='geometric',
                rot_error_type='axis_angle')

            actions = torch.zeros(
                (self.num_envs, self.cfg_task.env.numActions),
                device=self.device)
            actions[:, 0:3] = 1 * pos_error
            actions[:, 3:6] = 1 * axis_angle_error

            dof_targets = (1 - 0.5*progress) * residual_actuated_dof_open_pos + 0.5*progress * self.residual_actuated_dof_demo_pos
            actions[:, 6:] = dof_targets - self.dof_pos[:, self.residual_actuated_dof_indices]
            actions = torch.clamp(actions, -1, 1)

            self._apply_actions_as_ctrl_targets(
                ik_body_pose_actions=actions[:, 0:6],
                residual_dof_actions=actions[:, 6:],
                do_scale=False)

            self.gym.simulate(self.sim)
            self.render()

            self.refresh_base_tensors()
            self.refresh_env_tensors()
            self._refresh_task_tensors()
            self.gym.fetch_results(self.sim, True)

            if len(self.cfg_base.debug.visualize) > 0 and self.num_envs < 64:
                self.gym.clear_lines(self.viewer)
                self.draw_visualizations(self.cfg_base.debug.visualize)

        self.dof_vel[env_ids, :] = torch.zeros_like(self.dof_vel[env_ids])

        # Set DOF state
        multi_env_ids_int32 = self.robot_actor_ids_sim[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(multi_env_ids_int32),
            len(multi_env_ids_int32))

    def visualize_ik_body_demo_pose(self, env_id: int, axis_length: float = 0.3
                                    ) -> None:
        self.visualize_body_pose("ik_body_demo", env_id, axis_length)

    def visualize_synthetic_pointcloud_pos(self, env_id: int) -> None:
        self.visualize_body_pos("synthetic_pointcloud", env_id)

    def visualize_pre_grasp_pose(self, env_id: int, axis_length: float = 0.3
                                    ) -> None:
        self.visualize_body_pose("pre_grasp", env_id, axis_length)
