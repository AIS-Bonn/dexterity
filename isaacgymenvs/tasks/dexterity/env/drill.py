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

"""Dexterity: class for drill env.

Inherits base class and abstract environment class. Inherited by drill task
classes. Not directly executed.

Configuration defined in DexterityEnvDrill.yaml.
"""

import hydra
import numpy as np
import os
from isaacgym.torch_utils import *
from typing import *

from isaacgym import gymapi
from isaacgymenvs.tasks.dexterity.base.base import DexterityBase
from isaacgymenvs.tasks.dexterity.env.schema_class_env import DexterityABCEnv
from isaacgymenvs.tasks.dexterity.env.schema_config_env import \
    DexteritySchemaConfigEnv

from urdfpy import URDF
import trimesh
import mujoco
import mujoco_viewer
from scipy.spatial.transform import Rotation as R
from isaacgymenvs.tasks.dexterity.xml.robot import DexterityRobot
import nevergrad as ng


class DexterityManipulatorSim:
    """Helper class that creates a MuJoCo physics simulation of the manipulator
    and  optimizes its ik_body pose and joint angles to match a generalized
    keypoint demonstration."""

    def __init__(self, manipulator_robot: DexterityRobot,
                 canonical_ik_body_pos: np.array,
                 canonical_ik_body_quat: np.array,
                 canonical_ctrl: np.array,
                 canonical_keypoint_pos: np.array,
                 target_keypoint_pos: np.array,
                 keypoints: str,
                 show: bool = True) -> None:
        self._manipulator_robot = manipulator_robot

        self._canonical_ik_body_pos = canonical_ik_body_pos
        self._canonical_ik_body_quat = canonical_ik_body_quat
        self._canonical_ctrl = canonical_ctrl

        self._canonical_ik_body_quat_ig = torch.Tensor(
            [self._canonical_ik_body_quat[1], self._canonical_ik_body_quat[2],
             self._canonical_ik_body_quat[3], self._canonical_ik_body_quat[0]])

        self._canonical_keypoint_pos = canonical_keypoint_pos
        self._target_keypoint_pos = target_keypoint_pos
        self._keypoints_group = keypoints
        self._show = show
        self._sim_step = 0
        self._init_mj_model()

        self._keypoint_site_names = []
        for site_name in self._manipulator_robot.model.site_names:
            split_name = site_name.split("-")
            if len(split_name) == 3 and split_name[0] == "keypoint" \
                    and split_name[1] == self._keypoints_group:
                self._keypoint_site_names.append(site_name)

    def _init_mj_model(self) -> None:
        with self._manipulator_robot.model.as_xml(
                './dexterity_manipulator_sim.xml'):
            tmp_model = mujoco.MjModel.from_xml_path(
                './dexterity_manipulator_sim.xml')

        data = mujoco.MjData(tmp_model)
        mujoco.mj_step(tmp_model, data)
        tracker_pos_offset = data.body('tracker').xpos.copy()
        tracker_quat_offset = data.body('tracker').xquat.copy()

        self._manipulator_robot.model.add_sites(
            self._canonical_keypoint_pos, "canonical_keypoint_pos",
            rgba="1 0 0 1")
        self._manipulator_robot.model.add_sites(
            self._target_keypoint_pos, "target_keypoint_pos",
            rgba="0.8 0.8 0 1")
        self._manipulator_robot.model.add_mocap(
            pos=tracker_pos_offset, quat=tracker_quat_offset)
        self._manipulator_robot.model.add_freejoint()
        #manipulator_model.model.add_coordinate_system()

        with self._manipulator_robot.model.as_xml(
                './dexterity_manipulator_sim.xml'):
            self._mj_model = mujoco.MjModel.from_xml_path(
                './dexterity_manipulator_sim.xml')
        self._mj_data = mujoco.MjData(self._mj_model)
        if self._show:
            self._mj_viewer = mujoco_viewer.MujocoViewer(
                self._mj_model, self._mj_data)

    def step(self, ik_body_pos: np.array = None, ik_body_quat: np.array = None,
              ctrl: np.array = None) -> None:
        # Set ik_body mocap to target pose
        if ik_body_pos is not None:
            self._mj_data.mocap_pos[0] = ik_body_pos
        if ik_body_quat is not None:
            self._mj_data.mocap_quat[0] = ik_body_quat

        # Set control to target DoF positions
        if ctrl is not None:
            self._mj_data.ctrl = ctrl

        # Step simulation
        mujoco.mj_step(self._mj_model, self._mj_data)
        self._sim_step += 1
        if self._show:
            self._mj_viewer.render()

    def loss(self, delta_ik_body_pos: np.array, delta_ik_body_quat: np.array,
             delta_ctrl: np.array, sim_steps: int = 128) -> float:


        # Get absolute ik_body pose from deltas
        ik_body_pos = self._canonical_ik_body_pos + delta_ik_body_pos
        ik_body_quat = quat_mul(self._canonical_ik_body_quat_ig, torch.from_numpy(delta_ik_body_quat).to(torch.float32) + torch.Tensor([0., 0, 0, 1])).numpy()
        ctrl = self._canonical_ctrl + delta_ctrl

        for i in range(sim_steps):
            self.step(ik_body_pos, ik_body_quat, ctrl)

        print("self._keypoint_site_names:", self._keypoint_site_names)

        resulting_keypoint_pos = []
        for site_name in self._keypoint_site_names:
            resulting_keypoint_pos.append(self._mj_data.site(site_name).xpos)
        resulting_keypoint_pos = np.array(resulting_keypoint_pos)

        print("self._target_keypoint_pos:", self._target_keypoint_pos)

        print("resulting_keypoint_pos:", resulting_keypoint_pos)

        loss = np.linalg.norm(self._target_keypoint_pos - resulting_keypoint_pos, axis=1).mean()
        print("loss:", loss)

        return loss

    def get_rigid_body_poses(self) -> Dict[str, Any]:
        rigid_body_poses = {}
        for body_name in self._manipulator_robot.model.body_names:
            rigid_body_poses[body_name] = {}
            rigid_body_poses[body_name]["pos"] = \
                self._mj_data.body(body_name).xpos
            rigid_body_poses[body_name]["quat"] = \
                self._mj_data.body(body_name).xquat
        return rigid_body_poses

    def move_to_canonical_pose(self) -> None:
        for i in range(128):
            self.step(self._canonical_ik_body_pos,
                      self._canonical_ik_body_quat,
                      self._canonical_ctrl)

    def optimize_for_keypoint_pose(self) -> None:
        self.move_to_canonical_pose()
        parameterization = ng.p.Instrumentation(
            delta_ik_body_pos=ng.p.Array(shape=(3,)).set_bounds(-0.1, 0.1),
            delta_ik_body_quat=ng.p.Array(shape=(4,)).set_bounds(-1, 1),
            delta_ctrl=ng.p.Array(
                shape=(self._manipulator_robot.model.num_actions,)).set_bounds(
                -0.5, 0.5)
        )

        optimizer = ng.optimizers.NGOpt(parameterization, budget=32)
        recommendation = optimizer.minimize(self.loss)
        print("recommendation:", recommendation)


class DexterityTool:
    """Helper class that wraps tool assets to make information about the
    geometry, etc. more easily available."""
    def __init__(self, gym, sim, asset_root, asset_file,
                 cfg_robot: List[str]) -> None:
        self._gym = gym
        self._asset_root = asset_root
        self._asset_file = asset_file
        self._cfg_robot = cfg_robot

        # Load default asset options
        self._asset_options = gymapi.AssetOptions()
        self._asset_options.override_com = True
        self._asset_options.override_inertia = True
        self._asset_options.vhacd_enabled = True  # Enable convex decomposition
        self._asset_options.vhacd_params = gymapi.VhacdParams()
        self._asset_options.vhacd_params.resolution = 1000000

        # Create IsaacGym asset
        self._asset = gym.load_asset(sim, asset_root, asset_file,
                                     self._asset_options)

        # Set object name based on asset file name
        self.name = asset_file.split('/')[-1].split('.')[0]

        # Initialize information about tool geometry and canonical geometry
        self._urdf = URDF.load(os.path.join(asset_root, asset_file))
        self._collision_mesh = trimesh.util.concatenate(
            [link.collision_mesh for link in self._urdf.links])

        self._canonical_urdf = URDF.load(
            os.path.join(asset_root, 'canonical/canonical.urdf'))
        self._canonical_collision_mesh = trimesh.util.concatenate(
            [link.collision_mesh for link in self._canonical_urdf.links])

        # Initialize corresponding demo_pose
        self._demo_pose = self._generalize_demo_pose(keypoints='hand_bodies')

    def _generalize_demo_pose(self, keypoints: str):
        self._load_manipulator_model()
        self._load_canonical_demo(keypoints)

        # No need to generalize pose for canonical model, for which the original
        # demonstration has been recorded
        if self._asset_file == 'canonical/canonical.urdf':
            return self._canonical_keypoint_pos

        # Sample point clouds for this tool and the canonical tool
        self._canonical_points = self.sample_canonical_pointcloud()
        self._tool_points = self.sample_pointcloud()


        '''
        # Sample point clouds for this tool and the canonical tool
        self._canonical_points = self.sample_canonical_pointcloud()
        self._tool_points = self.sample_pointcloud()

        # Create trimesh scene
        scene = trimesh.scene.Scene()

        # Add canonical and tool point cloud to scene
        scene.add_geometry(self._canonical_points)
        scene.add_geometry(self._tool_points)

        # Add manipulator meshes to scene
        for body_name, body_meshes in self._manipulator_bodies_trimesh.items():
            for body_collision_mesh in body_meshes["collision"]:
                body_offset_torch = torch.from_numpy(body_collision_mesh['pos_offset']).unsqueeze(0).to(torch.float32)
                body_quat = torch.Tensor(
                    [[manipulator_rigid_body_poses[body_name]['quat'][1],
                      manipulator_rigid_body_poses[body_name]['quat'][2],
                      manipulator_rigid_body_poses[body_name]['quat'][3],
                      manipulator_rigid_body_poses[body_name]['quat'][0]]])
                body_offset = quat_apply(body_quat, body_offset_torch)[
                    0].numpy()

                transform = self.transformation_matrix(
                    manipulator_rigid_body_poses[body_name][
                        'pos'] + body_offset,
                    manipulator_rigid_body_poses[body_name]['quat'])
                scene.add_geometry(body_collision_mesh["trimesh"], transform=transform)

        # Add markers for the keypoints to the scene
        for keypoint_idx in range(self._canonical_keypoint_pos.shape[0]):
            pos = self._canonical_keypoint_pos[keypoint_idx]
            quat = self._canonical_keypoint_quat[keypoint_idx]
            transform = self.transformation_matrix(pos, quat)
            axis_marker = trimesh.creation.axis(
                transform=transform, origin_size=0.002,
                axis_radius=0.001, axis_length=0.025)
            scene.add_geometry(axis_marker)

        from trimesh import viewer
        viewer = viewer.SceneViewer(scene)
        
        '''

        from pycpd import DeformableRegistration, RigidRegistration

        use_rigid_registration = True
        if use_rigid_registration:
            self._registration = RigidRegistration(
                X=np.array(self._tool_points.vertices),
                Y=np.array(self._canonical_points.vertices))
        else:
            self._registration = DeformableRegistration(
                X=np.array(self._tool_points.vertices),
                Y=np.array(self._canonical_points.vertices), beta=1, alpha=3)

        # Call registration callback with undeformed point clouds
        self._registration_callback(iteration=0, error=0,
                                    Y=np.array(self._canonical_points.vertices))

        TY, registration_params = self._registration.register(
            callback=self._registration_callback)

        # TODO: Adjust parameters for non-rigid registration

        # TODO: Apply the same transformation that the point-cloud undergoes to the canonical demo keypoints and visualize them in the callback

        # TODO: Write an optimizer that adjusts the ik_body pose and controls to best match these new keypoints

        # TODO: Take the keypoints from the MuJoCo model after the last optimization step as the generalized grasp pose

        # TODO: Potentially add regularizers, such as point cloud intersection, etc.

        import time
        time.sleep(1000)

    def _registration_callback(self, iteration: int, error: float,
                               Y: np.array, **kwargs) -> None:
        scene = trimesh.scene.Scene()
        print("iteration:", iteration)
        print("error:", error)

        # Add data / tool points to the scene
        scene.add_geometry(self._tool_points)

        # Add the current deformed/fitted version of the canonical point cloud
        count = 4096
        colors = np.array([[255, 0, 100, 255]]).repeat(count, axis=0)
        deformed_canonical_points = trimesh.points.PointCloud(Y, colors=colors)
        scene.add_geometry(deformed_canonical_points)

        # Add the current deformed/fitted version of the keypoint
        deformed_keypoint_pos = self._registration.transform_point_cloud(
            self._canonical_keypoint_pos)
        deformed_keypoint_quat = self._canonical_keypoint_quat

        for keypoint_idx in range(self._canonical_keypoint_pos.shape[0]):
            transform = self.transformation_matrix(
                deformed_keypoint_pos[keypoint_idx],
                deformed_keypoint_quat[keypoint_idx])
            axis_marker = trimesh.creation.axis(
                transform=transform, origin_size=0.002,
                axis_radius=0.001, axis_length=0.025)
            scene.add_geometry(axis_marker)

        from copy import deepcopy
        # Initialize MuJoCo simulation of the manipulator robot
        manipulator_sim = DexterityManipulatorSim(
            deepcopy(self._manipulator_model),
            self._canonical_ik_body_pos_mujoco,
            self._canonical_ik_body_quat_mujoco,
            self._canonical_ctrl,
            self._canonical_keypoint_pos,
            deformed_keypoint_pos, keypoints='hand_bodies', show=False)

        if iteration == 0:
            manipulator_sim.move_to_canonical_pose()

        else:
            manipulator_sim.optimize_for_keypoint_pose()

        manipulator_rigid_body_poses = manipulator_sim.get_rigid_body_poses()

        for body_name, body_meshes in self._manipulator_bodies_trimesh.items():
            for body_collision_mesh in body_meshes["collision"]:
                body_offset_torch = torch.from_numpy(
                    body_collision_mesh['pos_offset']).unsqueeze(0).to(
                    torch.float32)
                body_quat = torch.Tensor(
                    [[manipulator_rigid_body_poses[body_name]['quat'][1],
                      manipulator_rigid_body_poses[body_name]['quat'][2],
                      manipulator_rigid_body_poses[body_name]['quat'][3],
                      manipulator_rigid_body_poses[body_name]['quat'][0]]])
                body_offset = quat_apply(body_quat, body_offset_torch)[
                    0].numpy()

                transform = self.transformation_matrix(
                    manipulator_rigid_body_poses[body_name][
                        'pos'] + body_offset,
                    manipulator_rigid_body_poses[body_name]['quat'])
                scene.add_geometry(body_collision_mesh["trimesh"],
                                   transform=transform)

        from trimesh import viewer
        window = viewer.SceneViewer(scene)

    def _load_manipulator_model(self):
        """Loads a model of the used robot without the robotic arm and stores
        information required for visualization."""
        model_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '..', '..', '..', '..',
                         'assets', 'dexterity'))
        assert self._cfg_robot[-1] == 'vive_tracker/tracker.xml', \
            f"Generalization of grasp poses relies on VIVE tracker as the " \
            f"ik_body, which is not present in the robot configuration " \
            f"{self._cfg_robot}."

        # Create model without robot arm
        self._manipulator_model = DexterityRobot(
            model_root, self._cfg_robot[1:])

        # Save trimesh meshes of manipulator bodies, as they are required in
        # the visualization later
        self._manipulator_bodies_trimesh = \
            self._manipulator_model.model.bodies_as_trimesh()

    def _load_canonical_demo(self, keypoints: str):
        """Loads the demonstration (for the manipulator used) for the canonical
        tool."""

        # Find and load the correct demonstration
        canonical_demo_path = os.path.join(
            self._asset_root, 'canonical',
            self._manipulator_model.manipulator.model_name + '_demo_pose.npz')
        assert os.path.isfile(canonical_demo_path), \
            f"Tried to load canonical demo pose for " \
            f"{self._manipulator_model.manipulator.model_name}, but " \
            f"{canonical_demo_path} was not found."
        canonical_demo_pose_dict = np.load(canonical_demo_path)

        # Load arrays from demo file
        self._canonical_keypoint_pos = canonical_demo_pose_dict[
            keypoints + '_demo_pos'][0]
        self._canonical_keypoint_quat = canonical_demo_pose_dict[
            keypoints + '_demo_quat'][0]
        self._canonical_ik_body_pos = canonical_demo_pose_dict[
            'ik_body_demo_pos'][0]
        self._canonical_ik_body_quat = canonical_demo_pose_dict[
            'ik_body_demo_quat'][0]
        self._canonical_ctrl = canonical_demo_pose_dict[
            'residual_actuated_demo_dofs'][0]

        # Convert pose of ik_body for MuJoCo
        self._canonical_ik_body_pos_mujoco = self._canonical_ik_body_pos
        # Reorder to match [w, x, y, z] quaternion convention of MuJoCo
        self._canonical_ik_body_quat_mujoco = np.array(
            [self._canonical_ik_body_quat[3], self._canonical_ik_body_quat[0],
             self._canonical_ik_body_quat[1], self._canonical_ik_body_quat[2]])

    def _create_mujoco_model(self, manipulator_model,
                             canonical_keypoint_demo_pos):
        with manipulator_model.model.as_xml('./robot_drill_test.xml'):
            tmp_model = mujoco.MjModel.from_xml_path('./robot_drill_test.xml')

        data = mujoco.MjData(tmp_model)
        mujoco.mj_step(tmp_model, data)
        tracker_pos_offset = data.body('tracker').xpos.copy()
        tracker_quat_offset = data.body('tracker').xquat.copy()

        manipulator_model.model.add_sites(canonical_keypoint_demo_pos)
        manipulator_model.model.add_mocap(
            pos=tracker_pos_offset, quat=tracker_quat_offset)
        manipulator_model.model.add_freejoint()
        #manipulator_model.model.add_coordinate_system()

        with manipulator_model.model.as_xml('./robot_drill_test.xml'):
            mj_model = mujoco.MjModel.from_xml_path('./robot_drill_test.xml')
        return mj_model

    def transformation_matrix(self, pos, quat, reorder_quat: bool = True) -> np.array:
        t = np.zeros((4, 4))
        t[0, 3] = pos[0]
        t[1, 3] = pos[1]
        t[2, 3] = pos[2]
        t[3, 3] = 1
        if reorder_quat:
            r = R.from_quat(np.array([quat[1], quat[2], quat[3], quat[0]]))
        else:
            r = R.from_quat(quat)
        t[0:3, 0:3] = r.as_matrix()
        return t

    @property
    def asset(self):
        return self._asset

    @property
    def asset_options(self) -> gymapi.AssetOptions:
        return self._asset_options

    @asset_options.setter
    def asset_options(self, asset_options: gymapi.AssetOptions) -> None:
        self._asset_options = asset_options

    @property
    def rigid_body_count(self) -> int:
        return self._gym.get_asset_rigid_body_count(self._asset)

    @property
    def rigid_shape_count(self) -> int:
        return self._gym.get_asset_rigid_shape_count(self._asset)

    @property
    def start_pose(self) -> gymapi.Transform:
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(0, 0, 0.5)
        return start_pose

    def sample_pointcloud(self, count: int = 4096) -> np.ndarray:
        points = np.array(trimesh.sample.sample_surface(
            self._collision_mesh, count=count)[0]).astype(np.float)
        colors = np.array([[255, 255, 0, 255]]).repeat(count, axis=0)
        return trimesh.points.PointCloud(points, colors=colors)

    def sample_canonical_pointcloud(self, count: int = 4096) -> np.ndarray:
        points = np.array(trimesh.sample.sample_surface(
            self._canonical_collision_mesh, count=count)[0]).astype(np.float)
        colors = np.array([[255, 0, 255, 255]]).repeat(count, axis=0)
        return trimesh.points.PointCloud(points, colors=colors)\


class DexterityEnvDrill(DexterityBase, DexterityABCEnv):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless,
                 virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize environment superclass.
        Acquire tensors."""

        self._get_env_yaml_params()

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

        config_path = 'task/DexterityEnvDrill.yaml'  # relative to cfg dir
        self.cfg_env = hydra.compose(config_name=config_path)
        self.cfg_env = self.cfg_env['task']  # strip superfluous nesting

    def create_envs(self):
        """Set env options. Import assets. Create actors."""

        lower = gymapi.Vec3(-self.cfg_base.env.env_spacing,
                            -self.cfg_base.env.env_spacing, 0.0)
        upper = gymapi.Vec3(self.cfg_base.env.env_spacing,
                            self.cfg_base.env.env_spacing,
                            self.cfg_base.env.env_spacing)
        num_per_row = int(np.sqrt(self.num_envs))

        robot_asset, table_asset = self.import_robot_assets()
        drill_assets, drill_site_assets = self._import_env_assets()

        self._create_actors(lower, upper, num_per_row, robot_asset, table_asset,
                            drill_assets, drill_site_assets)

    def _import_env_assets(self):
        """Set object assets options. Import objects."""
        return self._import_drill_assets()

    def _import_drill_assets(self):
        asset_root = os.path.normpath(
            os.path.join(os.path.dirname(__file__), '../..', '..', '..',
                         'assets', 'dexterity', 'tools', 'drills'))
        asset_files = self.cfg_env['env']['drills']
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False

        # While I have not specified asset inertia manually yet
        asset_options.override_com = True
        asset_options.override_inertia = True

        asset_options.vhacd_enabled = True  # Enable convex decomposition
        asset_options.vhacd_params = gymapi.VhacdParams()
        asset_options.vhacd_params.resolution = 1000000

        site_options = gymapi.AssetOptions()
        site_options.fix_base_link = True

        drill_assets, drill_site_assets = [], []
        for asset_file in asset_files:
            #drill = DexterityTool(self.gym, self.sim, asset_root, asset_file, self.cfg_base.env.robot)
            drill_asset = self.gym.load_asset(self.sim, asset_root, asset_file,
                                              asset_options)
            drill_assets.append(drill_asset)
            drill_site_asset = self.gym.load_asset(
                self.sim, asset_root,
                os.path.join(os.path.dirname(asset_file), "site.urdf"),
                site_options)
            drill_site_assets.append(drill_site_asset)
        return drill_assets, drill_site_assets

    def _create_actors(self, lower: gymapi.Vec3, upper: gymapi.Vec3,
                       num_per_row: int, robot_asset, table_asset,
                       drill_assets, drill_site_assets) -> None:
        robot_pose = gymapi.Transform()
        robot_pose.p.x = -self.cfg_base.env.robot_depth
        robot_pose.p.y = 0.0
        robot_pose.p.z = 0.0
        robot_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        table_pose = gymapi.Transform()
        table_pose.p.x = 0.0
        table_pose.p.y = 0.0
        table_pose.p.z = self.cfg_base.env.table_height * 0.5
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        drill_pose = gymapi.Transform()
        drill_site_pose = gymapi.Transform()

        self.env_ptrs = []
        self.robot_handles = []
        self.table_handles = []
        self.drill_handles = []
        self.drill_site_handles = []
        self.shape_ids = []
        self.robot_actor_ids_sim = []  # within-sim indices
        self.table_actor_ids_sim = []  # within-sim indices
        self.drill_actor_ids_sim = []  # within-sim indices
        self.drill_site_actor_ids_sim = []  # within-sim indices
        actor_count = 0

        drill_count = len(self.cfg_env['env']['drills'])

        for i in range(self.num_envs):
            # Create new env
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # Loop through all used drills
            drill_idx = i % drill_count
            used_drill = drill_assets[drill_idx]
            used_drill_site = drill_site_assets[drill_idx]

            # Aggregate all actors
            if self.cfg_base.sim.aggregate_mode > 1:
                max_rigid_bodies = \
                    self.gym.get_asset_rigid_body_count(used_drill) + \
                    self.gym.get_asset_rigid_body_count(used_drill_site) + \
                    self.robot.rigid_body_count + \
                    int(self.cfg_base.env.has_table)
                max_rigid_shapes = \
                    self.gym.get_asset_rigid_shape_count(used_drill) + \
                    self.gym.get_asset_rigid_shape_count(used_drill_site) + \
                    self.robot.rigid_shape_count + \
                    int(self.cfg_base.env.has_table)
                self.gym.begin_aggregate(env_ptr, max_rigid_bodies,
                                         max_rigid_shapes, True)

            # Create robot actor
            # collision_filter=-1 to use asset collision filters in XML model
            robot_handle = self.gym.create_actor(
                env_ptr, robot_asset, robot_pose, 'robot', i, -1, 0)
            self.robot_actor_ids_sim.append(actor_count)
            self.robot_handles.append(robot_handle)
            # Enable force sensors for robot
            self.gym.enable_actor_dof_force_sensors(env_ptr, robot_handle)
            actor_count += 1

            # Create table actor
            if self.cfg_base.env.has_table:
                table_handle = self.gym.create_actor(
                    env_ptr, table_asset, table_pose, 'table', i, 0, 1)
                self.table_actor_ids_sim.append(actor_count)
                actor_count += 1

                # Set table shape properties
                table_shape_props = self.gym.get_actor_rigid_shape_properties(
                    env_ptr, table_handle)
                table_shape_props[0].friction = self.cfg_base.env.table_friction
                table_shape_props[0].rolling_friction = 0.0  # default = 0.0
                table_shape_props[0].torsion_friction = 0.0  # default = 0.0
                table_shape_props[0].restitution = 0.0  # default = 0.0
                table_shape_props[0].compliance = 0.0  # default = 0.0
                table_shape_props[0].thickness = 0.0  # default = 0.0
                self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle,
                                                          table_shape_props)
                self.table_handles.append(table_handle)

            # Aggregate task-specific actors (objects)
            if self.cfg_base.sim.aggregate_mode == 1:
                max_rigid_bodies = \
                    self.gym.get_asset_rigid_body_count(used_drill) + \
                    self.gym.get_asset_rigid_body_count(used_drill_site)
                max_rigid_shapes = \
                    self.gym.get_asset_rigid_shape_count(used_drill) + \
                    self.gym.get_asset_rigid_shape_count(used_drill_site)
                self.gym.begin_aggregate(env_ptr, max_rigid_bodies,
                                         max_rigid_shapes, True)

            # Create drill actor
            drill_handle = self.gym.create_actor(
                env_ptr, used_drill, drill_pose, 'drill', i, 0, 2)
            self.drill_actor_ids_sim.append(actor_count)
            self.drill_handles.append(drill_handle)
            actor_count += 1

            # Create drill site actor (used to visualize target pose)
            drill_site_handle = self.gym.create_actor(
                env_ptr, used_drill_site, drill_site_pose, 'drill_site', i, 0,
                0)
            self.drill_site_actor_ids_sim.append(actor_count)
            self.drill_site_handles.append(drill_site_handle)
            for rigid_body_idx in range(
                    self.gym.get_asset_rigid_body_count(used_drill_site)):
                self.gym.set_rigid_body_color(
                    env_ptr, drill_site_handle, rigid_body_idx,
                    gymapi.MESH_VISUAL,
                    gymapi.Vec3(*self.cfg_env['env']['drill_target_color']))
            actor_count += 1

            # Finish aggregation group
            if self.cfg_base.sim.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.env_ptrs.append(env_ptr)

        self.num_actors = int(actor_count / self.num_envs)  # per env
        self.num_bodies = self.gym.get_env_rigid_body_count(env_ptr)  # per env
        self.num_dofs = self.gym.get_env_dof_count(env_ptr)  # per env

        # For setting targets
        self.robot_actor_ids_sim = torch.tensor(
            self.robot_actor_ids_sim, dtype=torch.int32, device=self.device)
        self.drill_actor_ids_sim = torch.tensor(
            self.drill_actor_ids_sim, dtype=torch.int32, device=self.device)
        self.drill_site_actor_ids_sim = torch.tensor(
            self.drill_site_actor_ids_sim, dtype=torch.int32,
            device=self.device)

        # For extracting root pos/quat
        self.drill_actor_id_env = self.gym.find_actor_index(
            env_ptr, 'drill', gymapi.DOMAIN_ENV)
        self.drill_site_actor_id_env = self.gym.find_actor_index(
            env_ptr, 'drill_site', gymapi.DOMAIN_ENV)

    def _acquire_env_tensors(self):
        """Acquire and wrap tensors. Create views."""
        self.drill_pos = self.root_pos[:, self.drill_actor_id_env, 0:3]
        self.drill_quat = self.root_quat[:, self.drill_actor_id_env, 0:4]
        self.drill_linvel = self.root_linvel[:, self.drill_actor_id_env, 0:3]
        self.drill_angvel = self.root_angvel[:, self.drill_actor_id_env, 0:3]

        self.drill_target_pos = self.root_pos[:, self.drill_site_actor_id_env, 0:3]
        self.drill_target_quat = self.root_quat[:, self.drill_site_actor_id_env, 0:4]

    def refresh_env_tensors(self):
        """Refresh tensors."""
        # NOTE: Tensor refresh functions should be called once per step, before
        # setters.
        # NOTE: Since the drill_pos, drill_quat, etc. are obtained from the
        # root state tensor through regular slicing, they are views rather than
        # separate tensors and hence don't have to be updated separately.
        pass

    def _get_random_drop_pos(self, env_ids) -> torch.Tensor:
        drill_pos_drop = torch.tensor(
            self.cfg_task.randomize.drill_pos_drop, device=self.device
        ).unsqueeze(0).repeat(len(env_ids), 1)
        drill_pos_drop_noise = \
            2 * (torch.rand((len(env_ids), 3), dtype=torch.float32,
                            device=self.device) - 0.5)  # [-1, 1]
        drill_pos_drop_noise = drill_pos_drop_noise @ torch.diag(torch.tensor(
            self.cfg_task.randomize.drill_pos_drop_noise, device=self.device))
        drill_pos_drop += drill_pos_drop_noise
        return drill_pos_drop

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
        return drill_quat_target

    def visualize_drill_pose(self, env_id: int, axis_length: float = 0.3
                             ) -> None:
        self.visualize_body_pose("drill", env_id, axis_length)

    def visualize_drill_target_pose(self, env_id: int, axis_length: float = 0.3
                                    ) -> None:
        self.visualize_body_pose("drill_target", env_id, axis_length)
