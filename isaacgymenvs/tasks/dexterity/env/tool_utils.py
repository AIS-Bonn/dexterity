from isaacgym import gymapi
from isaacgym.torch_utils import *
from isaacgymenvs.tasks.dexterity.xml.robot import DexterityRobot

from functools import partial
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import os
import pandas as pd
import pycpd
import threading
import torch
import torch.nn as nn
from tqdm import tqdm
import trimesh
from trimesh import viewer
from typing import *
from urdfpy import URDF
import mujoco
import mujoco_viewer
import nevergrad as ng
from scipy.spatial.transform import Rotation as R
from plotly.subplots import make_subplots
import plotly.graph_objects as go

torch.backends.cuda.matmul.allow_tf32 = False


MANIPULATOR_MODEL = "right_schunk_sih_hand"


class DexterityPointCloud:
    def __init__(
            self,
            points: np.array,
            rgb: Tuple[int, int, int]
    ) -> None:
        self._points = points
        self._rgba = np.array([rgb + (255, )]).repeat(points.shape[0], axis=0)

    @property
    def trimesh(self) -> trimesh.PointCloud:
        return trimesh.points.PointCloud(self._points, colors=self._rgba)

    @property
    def points(self) -> np.array:
        return self._points

    @points.setter
    def points(self, value: np.array) -> None:
        self._points = value


class DexterityDeformableRegistration(pycpd.DeformableRegistration):
    def __init__(
            self,
            target_pc: DexterityPointCloud,
            source_pc: DexterityPointCloud,
            alpha: float,
            beta: float,
            max_iterations: int,
            show: bool = True
    ) -> None:
        super().__init__(X=target_pc.points, Y=source_pc.points, alpha=alpha,
                         beta=beta, max_iterations=max_iterations,
                         tolerance=0.0)
        self.target_pc = target_pc
        self.source_pc = source_pc

        viewer_thread = threading.Thread(
            target=partial(self.trimesh_viewer_thread, show=show))
        viewer_thread.start()
        self.register(callback=self.update_source_points)
        viewer_thread.join()

    def update_source_points(self, iteration: int, error: float, X: np.array, Y: np.array) -> None:
        self.source_pc.points = Y

    def trimesh_viewer_thread(self, show: bool) -> None:
        if show:
            scene = trimesh.scene.Scene()
            scene.add_geometry(self.target_pc.trimesh, node_name='target')
            trimesh_viewer = viewer.SceneViewer(
                scene, callback=self.update_source_in_viewer)

    def update_source_in_viewer(self, scene: trimesh.Scene) -> None:
        scene.delete_geometry('source')
        scene.add_geometry(self.source_pc.trimesh, node_name='source')


class DexterityRigidRegistration(pycpd.RigidRegistration):
    def __init__(
            self,
            target_pc: DexterityPointCloud,
            source_pc: DexterityPointCloud,
            max_iterations: int,
            show: bool = True
    ) -> None:
        super().__init__(X=target_pc.points, Y=source_pc.points,
                         max_iterations=max_iterations)
        self.target_pc = target_pc
        self.source_pc = source_pc

        viewer_thread = threading.Thread(
            target=partial(self.trimesh_viewer_thread, show=show))
        viewer_thread.start()
        self.register(callback=self.update_source_points)
        viewer_thread.join()

    def update_source_points(self, iteration: int, error: float, X: np.array, Y: np.array) -> None:
        self.source_pc.points = Y

    def trimesh_viewer_thread(self, show: bool) -> None:
        if show:
            scene = trimesh.scene.Scene()
            scene.add_geometry(self.target_pc.trimesh, node_name='target')
            trimesh_viewer = viewer.SceneViewer(
                scene, callback=self.update_source_in_viewer)

    def update_source_in_viewer(self, scene: trimesh.Scene) -> None:
        scene.delete_geometry('source')
        scene.add_geometry(self.source_pc.trimesh, node_name='source')


class DexterityJointSpaceOptimization:
    def __init__(
            self,
            manipulator_robot: DexterityRobot,
            demo_pose: Dict[str, np.array],
            target_keypoints: np.array,
            keypoint_group: str = 'hand_bodies',
            show: bool = True
    ) -> None:
        self.manipulator_robot = manipulator_robot
        self.demo_pose = demo_pose
        self.target_keypoints = target_keypoints
        self.keypoints_group = keypoint_group

        self.ik_body_quat_torch = torch.from_numpy(self.demo_pose['ik_body_demo_quat']).to(torch.float32)
        self.ik_body_quat = self._to_mujoco_quat(demo_pose['ik_body_demo_quat'])

        self.show = show
        self.sim_step = 0
        self._init_mj_model()

        self._keypoint_site_names = []
        for site_name in self.manipulator_robot.model.site_names:
            split_name = site_name.split("-")
            if len(split_name) == 3 and split_name[0] == "keypoint" \
                    and split_name[1] == self.keypoints_group:
                self._keypoint_site_names.append(site_name)

    def _to_mujoco_quat(self, quat: np.array) -> np.array:
        return np.array([quat[3], quat[0], quat[1], quat[2]])

    def _init_mj_model(self) -> None:
        with self.manipulator_robot.model.as_xml(
                './dexterity_joint_space_optimization.xml'):
            tmp_model = mujoco.MjModel.from_xml_path(
                './dexterity_joint_space_optimization.xml')

        data = mujoco.MjData(tmp_model)
        mujoco.mj_step(tmp_model, data)
        tracker_pos_offset = data.body('tracker').xpos.copy()
        tracker_quat_offset = data.body('tracker').xquat.copy()

        self.manipulator_robot.model.add_sites(
            self.demo_pose[self.keypoints_group + '_demo_pos'], "canonical_keypoint_pos",
            rgba="1 0 0 1")
        self.manipulator_robot.model.add_sites(
            self.target_keypoints, "target_keypoint_pos",
            rgba="0.8 0.8 0 1")
        self.manipulator_robot.model.add_mocap(
            pos=tracker_pos_offset, quat=tracker_quat_offset)
        self.manipulator_robot.model.add_freejoint()
        #manipulator_model.model.add_coordinate_system()

        with self.manipulator_robot.model.as_xml(
                './dexterity_joint_space_optimization.xml'):
            self._mj_model = mujoco.MjModel.from_xml_path(
                './dexterity_joint_space_optimization.xml')
        self._mj_data = mujoco.MjData(self._mj_model)
        if self.show:
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
        self.sim_step += 1
        if self.show:
            self._mj_viewer.render()

    def axis_angle_to_quat(self, roll, pitch, yaw):
        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(
            roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(
            roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(
            roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(
            roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
        return np.array([qx, qy, qz, qw])

    def loss(self, delta_ik_body_pos: np.array, delta_ik_body_rot: np.array,
             delta_ctrl: np.array, sim_steps: int = 128) -> float:

        delta_ik_body_quat = self.axis_angle_to_quat(
            delta_ik_body_rot[0], delta_ik_body_rot[1], delta_ik_body_rot[2])

        # Get absolute ik_body pose from deltas
        ik_body_pos = self.demo_pose['ik_body_demo_pos'] + delta_ik_body_pos
        ik_body_quat = quat_mul(self.ik_body_quat_torch, torch.from_numpy(delta_ik_body_quat).to(torch.float32)).numpy()
        ik_body_quat = np.array([ik_body_quat[3], ik_body_quat[0], ik_body_quat[1], ik_body_quat[2]])

        self.ctrl = self.demo_pose['residual_actuated_dof_demo_pos'] + delta_ctrl

        for i in range(sim_steps):
            self.step(ik_body_pos, ik_body_quat, self.ctrl)

        resulting_keypoint_pos = self.get_keypoint_pos()

        loss = np.linalg.norm(
            self.target_keypoints - resulting_keypoint_pos, axis=1).mean()
        return loss

    def get_rigid_body_poses(self) -> Dict[str, Any]:
        rigid_body_poses = {}
        for body_name in self.manipulator_robot.model.body_names:
            rigid_body_poses[body_name] = {}
            rigid_body_poses[body_name]["pos"] = \
                self._mj_data.body(body_name).xpos
            rigid_body_poses[body_name]["quat"] = \
                self._mj_data.body(body_name).xquat
        return rigid_body_poses

    def get_keypoint_pos(self) -> np.array:
        keypoint_pos = []
        for site_name in self._keypoint_site_names:
            keypoint_pos.append(self._mj_data.site(site_name).xpos)
        return np.array(keypoint_pos)

    def get_keypoint_quat(self) -> np.array:
        keypoint_quat = []
        for site_name in self._keypoint_site_names:
            keypoint_quat.append(np.array([0, 0, 0, 1.]))
        return np.array(keypoint_quat)

    def get_actuated_dof_pos(self) -> np.array:
        return self.ctrl

    def get_ik_body_pos(self) -> np.array:
        return self._mj_data.mocap_pos[0]

    def get_ik_body_quat(self) -> np.array:
        mj_quat = self._mj_data.mocap_quat[0]
        return np.array([mj_quat[1], mj_quat[2], mj_quat[3], mj_quat[0]])

    def move_to_canonical_pose(self, sim_steps: int = 256) -> None:
        for i in range(sim_steps):
            self.step(self.demo_pose['ik_body_demo_pos'],
                      self.ik_body_quat,
                      self.demo_pose['residual_actuated_dof_demo_pos'])

    def optimize_joints_for_keypoints(self, budget: int = 128) -> None:
        self.move_to_canonical_pose()

        parameterization = ng.p.Instrumentation(
            delta_ik_body_pos=ng.p.Array(shape=(3,)).set_mutation(
                sigma=0.01).set_bounds(-0.05, 0.05),
            delta_ik_body_rot=ng.p.Array(shape=(3,)).set_mutation(
                sigma=0.1).set_bounds(-0.5, 0.5),
            delta_ctrl=ng.p.Array(
                shape=(self.manipulator_robot.model.num_actions,)).set_mutation(
                sigma=0.1).set_bounds(-0.5, 0.5))

        optimizer = ng.optimizers.NGOpt(parameterization, budget=budget)
        recommendation = optimizer.minimize(self.loss)


class DexterityInstance:
    def __init__(
            self,
            gym,
            sim,
            asset_root: str,
            asset_file: str,
            vhacd_resolution: int = 100000
    ) -> None:
        self.gym = gym
        self.sim = sim
        self.urdf = URDF.load(os.path.join(asset_root, asset_file))
        self.collision_mesh = self.assemble_collision_mesh()
        self.name = os.path.basename(os.path.dirname(os.path.join(asset_root, asset_file)))
        self._latent_shape_params = None
        self._demo_dict = None

        # Load default asset options
        self._asset_options = gymapi.AssetOptions()
        self._asset_options.fix_base_link = False
        self._asset_options.override_com = True
        self._asset_options.override_inertia = True

        #self._asset_options.max_linear_velocity = 10.0
        #self._asset_options.max_angular_velocity = 10.0

        self._asset_options.vhacd_enabled = True  # Enable convex decomposition
        self._asset_options.vhacd_params = gymapi.VhacdParams()
        self._asset_options.vhacd_params.resolution = vhacd_resolution

        # Create IsaacGym asset
        self._asset = gym.load_asset(sim, asset_root, asset_file,
                                     self._asset_options)


        rigid_shape_props = gym.get_asset_rigid_shape_properties(self._asset)
        for prop in rigid_shape_props:
            pass
            #print("prop.compliance:", prop.compliance)
            #print("prop.contact_offset:", prop.contact_offset)
            #print("prop.friction:", prop.friction)
            #print("prop.rest_offset:", prop.rest_offset)
            #print("prop.restitution:", prop.restitution)
            #print("prop.rolling_friction:", prop.rolling_friction)
            #print("prop.thickness:", prop.thickness)
            #print("prop.torsion_friction:", prop.torsion_friction)
        gym.set_asset_rigid_shape_properties(self._asset, rigid_shape_props)

    def assemble_collision_mesh(self) -> trimesh.Trimesh:
        collision_meshes = []
        for link in self.urdf.links:
            if link.collision_mesh is not None:
                collision_meshes.append(link.collision_mesh)
        return trimesh.util.concatenate(collision_meshes)

    def sample_pointcloud(
            self,
            count: int = 1024,
            rgb: Tuple[int, int, int] = (0, 0, 0)
    ) -> DexterityPointCloud:
        points = np.array(trimesh.sample.sample_surface(
            self.collision_mesh, count=count)[0]).astype(float)
        return DexterityPointCloud(points, rgb=rgb)

    @property
    def latent_shape_params(self) -> torch.Tensor:
        return self._latent_shape_params

    @latent_shape_params.setter
    def latent_shape_params(self, value: torch.Tensor) -> None:
        self._latent_shape_params = value

    @property
    def demo_dict(self) -> Dict[str, Any]:
        return self._demo_dict

    @demo_dict.setter
    def demo_dict(self, value: Dict[str, Any]) -> None:
        self._demo_dict = value

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
        return self.gym.get_asset_rigid_body_count(self._asset)

    @property
    def rigid_shape_count(self) -> int:
        return self.gym.get_asset_rigid_shape_count(self._asset)

    @property
    def synthetic_pointcloud(self) -> np.array:
        count = 64
        return np.array(trimesh.sample.sample_surface(self.collision_mesh, count=count)[0]).astype(float)


class DexterityCategory:
    def __init__(
            self,
            gym,
            sim,
            asset_root: str,
            source_file: str,
            target_files: List[str],
            demo_path: os.path,
            alpha: float = 2.0,
            beta: float = 3.0,
            max_iterations: int = 100,
            num_latents: int = 4,
            normalize: bool = True,
            disable_generalization: bool = False
    ) -> None:
        self.gym = gym
        self.sim = sim
        self.asset_root = asset_root
        self.source_file = source_file
        self.target_files = target_files
        self.demo_path = demo_path
        self.alpha = alpha
        self.beta = beta
        self.max_iterations = max_iterations
        self.num_latents = num_latents
        self.normalize = normalize
        self.disable_generalization = disable_generalization

    def build(self) -> None:
        if 'drill' in self.asset_root:
            self.tool_category = 'drill'
        elif 'mug' in self.asset_root:
            self.tool_category = 'mug'
        elif 'hammer' in self.asset_root:
            self.tool_category = 'hammer'
        else:
            assert False, self.asset_root

        # Load demo pose
        self.demo_pose_npz = np.load(self.demo_path)
        self.demo_pose = {}
        for k, v in self.demo_pose_npz.items():
            self.demo_pose[k] = v[0]

        #self.load_canonical_demonstration(self.asset_root, self.source_file)
        self.acquire_instances(self.asset_root, self.source_file,
                               self.target_files)
        self.deformation_design_matrix = self.acquire_deformation_fields(
            self.alpha, self.beta, self.max_iterations)
        self.find_principal_deformations(num_latents=self.num_latents)

        d = {
            'latent_shape_params': [],
            'first_principal_component': [],
            'second_principal_component': [],
            'name': []
        }
        for instance in [self.source_instance, ] + self.target_instances:
            d['name'].append(instance.name)
            d['first_principal_component'].append(
                instance.latent_shape_params[0].cpu().numpy())
            d['second_principal_component'].append(
                instance.latent_shape_params[1].cpu().numpy())
            d['latent_shape_params'].append(instance.latent_shape_params.cpu().numpy())

        self.latent_space_df = pd.DataFrame(data=d)

        self.source_gaussian_kernel = pycpd.gaussian_kernel(
            self.source_pc.points, self.beta, self.source_pc.points)

    def to_file(self, save_path) -> None:
        # Create DataFrame from latent params of target instances
        self.latent_space_df.to_csv(save_path + '_latent_space.csv')

        np.save(save_path + '_principal_deformations.npy',
                self.principal_deformations.cpu().numpy())
        np.save(save_path + '_deformation_mean.npy',
                self.deformation_mean.cpu().numpy())
        np.save(save_path + '_deformation_std.npy',
                self.deformation_std.cpu().numpy())
        np.save(save_path + '_source_points.npy', self.source_pc.points)
        np.savez(save_path + "_demo_pose_canonical.npz", **self.demo_pose)

        for instance in [self.source_instance, ] + self.target_instances:
            np.savez(save_path + "_demo_pose_" + instance.name + ".npz", **instance.demo_dict)
            np.save(save_path + "_points_" + instance.name + ".npy", instance.sample_pointcloud(rgb=(255, 0, 0)).points)

    def from_file(self, load_path: str, load_instances: bool = True, test: bool = False):
        if 'drill' in load_path:
            self.tool_category = 'drill'
        elif 'mug' in load_path:
            self.tool_category = 'mug'
        elif 'hammer' in load_path:
            self.tool_category = 'hammer'
        else:
            assert False, load_path

        self.latent_space_df = pd.read_csv(load_path + '_latent_space.csv')
        self.demo_pose = np.load(load_path + "_demo_pose_canonical.npz")
        self.principal_deformations = torch.from_numpy(
            np.load(load_path + '_principal_deformations.npy')).to(torch.float32)
        self.deformation_mean = torch.from_numpy(
            np.load(load_path + '_deformation_mean.npy')).to(
            torch.float32)
        self.deformation_std = torch.from_numpy(
            np.load(load_path + '_deformation_std.npy')).to(
            torch.float32)
        self.source_pc = DexterityPointCloud(
            points=np.load(load_path + '_source_points.npy'), rgb=(0, 0, 255))

        self.source_gaussian_kernel = pycpd.gaussian_kernel(
            self.source_pc.points, self.beta, self.source_pc.points)

        self.instance_pointclouds = {}
        for instance_name in self.latent_space_df['name']:
            print("loading pointcloud for instance with name:", instance_name)
            points = np.load(load_path + "_points_" + instance_name + ".npy")
            self.instance_pointclouds[instance_name] = points

        if load_instances:
            self.acquire_instances(self.asset_root, self.source_file,
                                   self.target_files)

            for i, instance in enumerate([self.source_instance, ] + self.target_instances):
                if not test:
                    assert instance.name == self.latent_space_df['name'][i], \
                        "During training, the loaded instances should match " \
                        "the observed instances."

                if self.disable_generalization:
                    latent_shape_params = torch.zeros(self.num_latents)
                    demo_dict = self.demo_pose

                else:
                    latent_shape_params = self.latent_space_df['latent_shape_params'][i]
                    demo_dict = np.load(load_path + "_demo_pose_" + instance.name + ".npz")

                # Set demonstration and latent shape parameters of instance
                instance.demo_dict = demo_dict
                instance.latent_shape_params = latent_shape_params

    def acquire_instances(self, asset_root: str, source_file: str, target_files: List[str]) -> None:
        if self.tool_category == 'drill':
            vhacd_resolution = 512000
        else:
            vhacd_resolution = 100000

        # Create DexterityInstances from the asset files
        self.target_instances = []
        pbar = tqdm([source_file, ] + target_files)
        for i, instance_file in enumerate(pbar):
            pbar.set_description(f"Acquiring instance '{os.path.dirname(instance_file)}'")

            if i == 0:
                self.source_instance = DexterityInstance(self.gym, self.sim, asset_root,
                                                         instance_file, vhacd_resolution=vhacd_resolution)
            else:
                self.target_instances.append(
                    DexterityInstance(self.gym, self.sim, asset_root, instance_file, vhacd_resolution=vhacd_resolution))

    def acquire_deformation_fields(self, alpha: float, beta: float, max_iterations: int = 100) -> torch.Tensor:
        self.source_pc = self.source_instance.sample_pointcloud(rgb=(0, 0, 255))
        deformation_field_vectors = []
        pbar = tqdm(self.target_instances)
        for target_instance in pbar:
            pbar.set_description(f"Acquiring deformation from '{self.source_instance.name}' to '{target_instance.name}'")
            deformation_field = self.register_instance(target_instance, alpha, beta, max_iterations)
            deformation_field_vectors.append(deformation_field.flatten())
        return torch.from_numpy(np.array(deformation_field_vectors)).to(torch.float32)

    def register_instance(self, target_instance: DexterityInstance, alpha: float, beta: float, max_iterations: int = 200, show: bool = False) -> np.array:
        target_pc = target_instance.sample_pointcloud(rgb=(255, 0, 0))
        registration = DexterityDeformableRegistration(
            target_pc, deepcopy(self.source_pc), alpha, beta, max_iterations, show)
        G, W = registration.get_registration_parameters()
        return W.flatten()

    def find_principal_deformations(self, num_latents: int) -> None:
        self.deformation_mean = self.deformation_design_matrix.mean(0)
        self.deformation_std = self.deformation_design_matrix.std(0)
        self.norm_deformation_design_matrix = (self.deformation_design_matrix - self.deformation_mean) / self.deformation_std

        print("deformation_mean.shape:", self.deformation_mean.shape)
        print("self.deformation_design_matrix.shape:", self.deformation_design_matrix.shape)
        print("self.norm_deformation_design_matrix.shape:", self.norm_deformation_design_matrix.shape)
        print("self.norm_deformation_design_matrix.mean(0):", self.norm_deformation_design_matrix.mean(0))
        print("self.norm_deformation_design_matrix.std(0):",
              self.norm_deformation_design_matrix.std(0))
        print("self.norm_deformation_design_matrix:",
              self.norm_deformation_design_matrix)

        U, S, V = torch.pca_lowrank(self.norm_deformation_design_matrix, center=False)
        self.principal_deformations = V[:, :num_latents]

        self.source_instance.demo_dict = self.demo_pose
        source_deformation_vector = - self.deformation_mean / self.deformation_std

        print("source_deformation_vector:", source_deformation_vector)

        source_deformation_field = ((source_deformation_vector * self.deformation_std) + self.deformation_mean).reshape(
            -1, 3).numpy()

        print("source_deformation_field:", source_deformation_field)
        self.source_instance.latent_shape_params = torch.matmul(source_deformation_vector, self.principal_deformations)

        print("source-instance_latent_shape_params:", self.source_instance.latent_shape_params)

        print("self.source_instance.latent_shape_params:", self.source_instance.latent_shape_params)
        si_deformation_feature_vector = torch.matmul(
            self.source_instance.latent_shape_params, self.principal_deformations.T)
        si_deformation_field = ((si_deformation_feature_vector * self.deformation_std) + self.deformation_mean).reshape(-1, 3).numpy()

        print("si_deformation_feature_vector:", si_deformation_feature_vector)

        print("si_deformation_field:", si_deformation_field)
        print("si_deformation_field.mean():", si_deformation_field.mean())


        # Set latent shape params  and generalized keypoint pose of target instances.
        for deformation_vector, instance in zip(self.norm_deformation_design_matrix, self.target_instances):
            demo_dict = {}
            if self.disable_generalization:
                latent_shape_params = torch.zeros(self.num_latents)
                demo_dict['residual_actuated_dof_demo_pos'] = self.demo_pose['residual_actuated_dof_demo_pos']
                for keypoint_group in ['fingertips', 'hand_bodies']:
                    demo_dict[keypoint_group + '_demo_pos'] = self.demo_pose[keypoint_group + '_demo_pos']

            else:
                latent_shape_params = torch.matmul(deformation_vector, self.principal_deformations)
                transformed_hand_bodies = self.transform_keypoints(latent_shape_params, keypoint_group='hand_bodies')
                self.manipulator_model = DexterityRobot(
                    '/home/user/mosbach/PycharmProjects/dexterity/assets/dexterity/',
                    ['schunk_sih/right_hand.xml', 'vive_tracker/tracker.xml'])

                joint_space_optimization = DexterityJointSpaceOptimization(
                    self.manipulator_model, self.demo_pose, transformed_hand_bodies,
                    show=False)
                joint_space_optimization.optimize_joints_for_keypoints()
                demo_dict['residual_actuated_dof_demo_pos'] = joint_space_optimization.get_actuated_dof_pos()
                demo_dict['hand_bodies_demo_pos'] = joint_space_optimization.get_keypoint_pos()
                demo_dict['hand_bodies_demo_quat'] = joint_space_optimization.get_keypoint_quat()
                demo_dict['ik_body_demo_pos'] = joint_space_optimization.get_ik_body_pos()
                demo_dict['ik_body_demo_quat'] = joint_space_optimization.get_ik_body_quat()

            # Set demonstration and latent shape parameters of instance
            instance.demo_dict = demo_dict
            instance.latent_shape_params = latent_shape_params

    def transform_source_pointcloud(self, latent_space_params: torch.Tensor
                                    ) -> np.array:
        deformation_feature_vector = torch.matmul(
            latent_space_params, self.principal_deformations.T)
        #deformation_feature_vector = -self.deformation_mean / self.deformation_std
        deformation_field = ((deformation_feature_vector * self.deformation_std) + self.deformation_mean).reshape(
            -1, 3).numpy()

        deformed_source_pc = self.source_pc.points + np.dot(
            self.source_gaussian_kernel, deformation_field)
        return deformed_source_pc

    def transform_keypoints(self, latent_space_params: torch.Tensor,
                            keypoint_group: str) -> np.array:
        deformation_feature_vector = torch.matmul(
            latent_space_params, self.principal_deformations.T)
        deformation_field = ((deformation_feature_vector * self.deformation_std) + self.deformation_mean).reshape(
            -1, 3).numpy()

        gaussian_kernel = pycpd.gaussian_kernel(
            X=self.demo_pose[keypoint_group + '_demo_pos'], beta=self.beta,
            Y=self.source_pc.points)
        transformed_keypoints = self.demo_pose[keypoint_group + '_demo_pos'] + np.dot(
            gaussian_kernel, deformation_field)
        return transformed_keypoints

    def optimize_joints_for_keypoints(self, target_keypoints: np.array):
        self.manipulator_model = DexterityRobot(
            '/home/user/mosbach/PycharmProjects/dexterity/assets/dexterity/',
            ['schunk_sih/right_hand.xml', 'vive_tracker/tracker.xml'])

        joint_space_optimization = DexterityJointSpaceOptimization(
            self.manipulator_model, self.demo_pose, target_keypoints, show=False)
        joint_space_optimization.optimize_joints_for_keypoints()
        manipulator_rigid_body_poses = joint_space_optimization.get_rigid_body_poses()
        return manipulator_rigid_body_poses

    def fit_shape_to_observed(self, observed_points_list: List[torch.Tensor],
                              tool_pos: torch.Tensor, tool_quat: torch.Tensor,
                              visualize_energy_landscape: bool = True):
        def cpd_loss(def_can: torch.Tensor, obs: torch.Tensor):
            dist = torch.sum((obs[None, :, :] - def_can[:, None, :]) ** 2,
                             dim=2)  # (N, M)

            #min_dist = torch.min(dist, dim=0)[0]
            #loss = torch.sum(min_dist)
            P = torch.exp(-dist / (2 * sigma2))
            den = torch.sum(P, axis=0, keepdim=True)
            #P = torch.divide(P, den)
            loss = torch.sum(P, dim=0)
            loss = torch.log(loss)
            loss = -torch.sum(loss, dim=0)
            return loss

        self.estimated_observed_pointclouds = []

        instance_list = [self.source_instance, ] + self.target_instances

        for env_id in range(len(observed_points_list)):
            observed_points = observed_points_list[env_id]
            num_points_on_tool = observed_points.shape[0]
            curr_tool_pos = tool_pos[env_id].unsqueeze(0).repeat(num_points_on_tool, 1)
            curr_tool_quat = tool_quat[env_id].unsqueeze(0).repeat(num_points_on_tool, 1)

            observed_points_in_tool_coordinates = observed_points - curr_tool_pos
            observed_points_in_tool_coordinates = quat_rotate_inverse(
                curr_tool_quat, observed_points_in_tool_coordinates)

            latent_params = torch.zeros(self.num_latents).to("cuda:0")
            latent_params = nn.Parameter(data=latent_params)
            learning_rate = 0.05
            optimizer = torch.optim.Adam([latent_params, ], lr=learning_rate)
            sigma2 = 0.00001

            obs = observed_points_in_tool_coordinates
            canonical_points = torch.from_numpy(self.source_pc.points).to(torch.float32).to("cuda:0")
            can = canonical_points.to("cuda:0")

            def_std = self.deformation_std.to("cuda:0")
            def_mean = self.deformation_mean.to("cuda:0")
            principal_def = self.principal_deformations.T.to("cuda:0")
            source_gaussian_kernel = torch.from_numpy(self.source_gaussian_kernel).to(torch.float32).to("cuda:0")

            if visualize_energy_landscape:
                def save_pc_figure(pc, name):
                    if self.tool_category == 'drill':
                        x_range = [-0.325, 0.05]
                        y_range = [-0.1875, 0.1875]
                        z_range = [-0.325, 0.05]
                    elif self.tool_category == 'mug':
                        x_range = [-0.1, 0.1]
                        y_range = [-0.1, 0.1]
                        z_range = [-0.01, 0.19]
                    elif self.tool_category == 'hammer':
                        x_range = [-0.35, 0.05]
                        y_range = [-0.2, 0.2]
                        z_range = [-0.35, 0.05]
                    camera = dict(
                        eye=dict(x=1.5, y=0., z=0.)
                    )

                    tmp_fig = go.Figure(data=[go.Scatter3d(
                        x=pc[:, 0].detach().cpu().numpy(),
                        y=pc[:, 1].detach().cpu().numpy(),
                        z=pc[:, 2].detach().cpu().numpy(),
                        mode='markers',
                        marker=dict(size=4, color='rgba(0.5, 0.5, 0.5, 1.0)'),
                        name=name), ])

                    tmp_fig.update_layout(
                        showlegend=False,
                        scene=dict(
                            xaxis=dict(range=x_range, ),
                            yaxis=dict(range=y_range, ),
                            zaxis=dict(range=z_range, ), ),
                        height=800,
                        scene_camera=camera,
                    )

                    tmp_fig.update_scenes(aspectratio=dict(x=1, y=1, z=1))
                    tmp_fig.update_scenes(xaxis_visible=False,
                                          yaxis_visible=False,
                                          zaxis_visible=False)
                    tmp_fig.write_image(f"{name}.png")

                num_samples = 128
                min_val = -38.5
                max_val = 38.5
                latents = torch.zeros([4]).to("cuda:0")
                loss_map = np.zeros([num_samples, num_samples])
                for i, fpc in enumerate(np.linspace(min_val, max_val, num_samples)):
                    for j, spc in enumerate(np.linspace(min_val, max_val, num_samples)):
                        print("fpc:", fpc)
                        print("spc:", spc)

                        latents[0] = fpc
                        latents[1] = spc
                        deformation_feature_vector = torch.matmul(latents, principal_def)
                        deformation_field = ((deformation_feature_vector * def_std) + def_mean).reshape(-1, 3)
                        def_can = can + torch.matmul(source_gaussian_kernel, deformation_field)
                        loss = cpd_loss(def_can, obs)
                        loss_map[i, j] = loss.item()

                        if (i == 0 or i == num_samples - 1) \
                                and (j == 0 or j == num_samples - 1):
                            if env_id == 0:
                                save_pc_figure(def_can, f"env_id={env_id}; fpc={fpc}; spc={spc}")

                save_pc_figure(can, "canonical")
                save_pc_figure(obs, f"env_id={env_id}_observed")

                import matplotlib.pyplot as plt
                fig1, ax1 = plt.subplots(1, 1)
                ax1.imshow(- loss_map, cmap='RdYlGn')
                fig1.savefig(f'env_id={env_id}_energy_landscape.png')
                #plt.show()
                fig2, ax2 = plt.subplots(1, 1)
                ax2.imshow(- loss_map * np.abs(loss_map) ** 0.25, cmap='RdYlGn')
                fig2.savefig(f'env_id={env_id}_energy_landscape_log.png')
                #plt.show()

            obs_scatter = go.Scatter3d(
                x=obs[:, 0].cpu().numpy(),
                y=obs[:, 1].cpu().numpy(),
                z=obs[:, 2].cpu().numpy(),
                mode='markers',
                marker=dict(size=3),
                name='observed')
            scatters = [obs_scatter, ]

            #good_latent_params = torch.Tensor([24.01, -35.82]).to("cuda:0")
            #latent_params.data = good_latent_params

            losses = []

            for itr in range(1000):
                deformation_feature_vector = torch.matmul(
                    latent_params, principal_def)
                deformation_field = ((deformation_feature_vector * def_std) + def_mean).reshape(-1, 3)
                def_can = can + \
                          torch.matmul(source_gaussian_kernel, deformation_field)

                #def_can_scatter = go.Scatter3d(
                #    x=def_can[:, 0].detach().cpu().numpy(),
                #    y=def_can[:, 1].detach().cpu().numpy(),
                #    z=def_can[:, 2].detach().cpu().numpy(),
                #    mode='markers',
                #    marker=dict(size=3),
                #    name='deformed_canonical')

                #can_scatter = go.Scatter3d(
                #    x=can[:, 0].detach().cpu().numpy(),
                #    y=can[:, 1].detach().cpu().numpy(),
                #    z=can[:, 2].detach().cpu().numpy(),
                #    mode='markers',
                #    marker=dict(size=3),
                #    name='canonical')

                loss = cpd_loss(def_can, obs)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print("latent_params:", latent_params)
                print(f"loss: {loss:>7f}  [{itr:>5d}/{100:>5d}]")

                if itr % 250 == 0:
                    def_can_scatter = go.Scatter3d(
                        x=def_can[:, 0].detach().cpu().numpy(),
                        y=def_can[:, 1].detach().cpu().numpy(),
                        z=def_can[:, 2].detach().cpu().numpy(),
                        mode='markers',
                        marker=dict(size=3),
                        name=f"deformed_canonical_itr={itr}")
                    scatters.append(def_can_scatter)

            self.estimated_observed_pointclouds.append(def_can.detach().cpu().numpy())

            fig = go.Figure(data=scatters, )
            fig.update_layout(
                scene=dict(
                    xaxis=dict(range=[-0.75, 0.75], ),
                    yaxis=dict(range=[-0.75, 0.75], ),
                    zaxis=dict(range=[-0.05, 1.45], ), ),

            )

            #fig.show()

            loss_curve = go.Scatter(x=list(range(len(losses))),
                                    y=losses)

            loss_fig = go.Figure(data=[loss_curve, ])
            #loss_fig.show()

            # Set latent params and new demo pose for a partially observed instance
            instance_list[env_id].latent_shape_params = latent_params.detach().cpu().clone()
            demo_dict = {}
            transformed_hand_bodies = self.transform_keypoints(
                        instance_list[env_id].latent_shape_params, keypoint_group='hand_bodies')
            self.manipulator_model = DexterityRobot(
                '/home/user/mosbach/PycharmProjects/dexterity/assets/dexterity/',
                ['schunk_sih/right_hand.xml',
                 'vive_tracker/tracker.xml'])

            joint_space_optimization = DexterityJointSpaceOptimization(
                self.manipulator_model, self.demo_pose,
                transformed_hand_bodies, show=False)
            joint_space_optimization.optimize_joints_for_keypoints()
            demo_dict['residual_actuated_dof_demo_pos'] = joint_space_optimization.get_actuated_dof_pos()
            demo_dict['hand_bodies_demo_pos'] = joint_space_optimization.get_keypoint_pos()
            demo_dict['hand_bodies_demo_quat'] = joint_space_optimization.get_keypoint_quat()
            demo_dict['ik_body_demo_pos'] = joint_space_optimization.get_ik_body_pos()
            demo_dict['ik_body_demo_quat'] = joint_space_optimization.get_ik_body_quat()
            instance_list[env_id].demo_dict = demo_dict

    def fit_pose_to_observed(self, observed_points_list: List[torch.Tensor]):
        tool_pos, tool_quat = [], []

        for env_id in range(len(observed_points_list)):
            observed_points = observed_points_list[env_id].cpu().numpy()
            target_pc = DexterityPointCloud(points=observed_points,
                                            rgb=(255, 0, 0))
            source_pc = DexterityPointCloud(points=self.estimated_observed_pointclouds[env_id],
                                            rgb=(0, 0, 255))
            rigid_reg = DexterityRigidRegistration(target_pc, source_pc,
                                                   max_iterations=50)
            scale, rotation_matrix, translation_vector = rigid_reg.get_registration_parameters()
            rot = R.from_matrix(rotation_matrix)
            tool_pos.append(translation_vector)
            tool_quat.append(rot.as_quat())
        tool_pos = np.stack(tool_pos)
        tool_quat = np.stack(tool_quat)

        return tool_pos, tool_quat

    def draw(self, latent_shape_params: torch.Tensor, show: bool = True) -> go.Figure:
        fig = make_subplots(rows=1, cols=2, specs=[[{'is_3d': False}, {'is_3d': True}]])

        for latent_space_scatter in self.draw_latent_space(
                latent_shape_params, show=False):
            fig.add_trace(latent_space_scatter, row=1, col=1)

        fig.add_trace(self.draw_deformed_pointcloud(latent_shape_params, show=False), row=1, col=2)

        for mesh_component in self.draw_manipulator_mesh(latent_shape_params, show=False):
            fig.add_trace(mesh_component, row=1, col=2)

        if show:
            fig.show()
        return fig

    def draw_latent_space(self, latent_shape_params: torch.Tensor,
                          show: bool = False) -> go.Figure:

        fpc_range = [
            min(self.latent_space_df['first_principal_component']),
            max(self.latent_space_df['first_principal_component'])]
        spc_range = [
            min(self.latent_space_df['second_principal_component']),
            max(self.latent_space_df['second_principal_component'])]

        latent_space_scatter = go.Scatter(
            x=self.latent_space_df['first_principal_component'],
            y=self.latent_space_df['second_principal_component'],
            text=self.latent_space_df['name'],
            textposition="top center",
            mode='markers',
            name='Target Instances')

        latent_space_pointer = go.Scatter(
            x=[latent_shape_params[0]],
            y=[latent_shape_params[1]],
            text='pointer',
            mode='markers',
            marker=dict(size=8, symbol="x", color="red"),
            name='Current Latent Params'
        )

        feature_x = np.linspace(1.5 * fpc_range[0], 1.5 * fpc_range[1], num=100)
        feature_y = np.linspace(1.5 * spc_range[0], 1.5 * spc_range[1],num=100)

        # Creating 2-D grid of features
        [X, Y] = np.meshgrid(feature_x, feature_y)

        Z = 0. * (np.cos(X / 2) + np.sin(Y / 4))

        latent_space_background = go.Heatmap(
            x=feature_x, y=feature_y, z=Z,
        hoverinfo='none', showscale=False, showlegend=False, colorscale=[[0, 'rgb(255, 255, 255)'], [1, 'rgb(255, 255, 255)']])

        fig = go.Figure([latent_space_background, latent_space_scatter, latent_space_pointer])

        fig.update_xaxes(range=[1.5 * fpc_range[0], 1.5 * fpc_range[1]],
                         title_text='First principal component')
        fig.update_yaxes(range=[1.5 * spc_range[0], 1.5 * spc_range[1]],
                         title_text='Second principal component')

        fig.update_layout(legend=dict(
            yanchor="bottom",
            y=-0.5,
            xanchor="left",
            x=0.01
        ))

        if show:
            fig.show()
        return fig

    def draw_deformed_pointcloud(self, latent_shape_params: torch.Tensor,
                                 show_keypoints: bool,
                                 show_training_instances: List = [],
                                 show_axes: bool = False,
                                 show: bool = False) -> go.Figure:
        # Create scatter for deformed source points.
        deformed_points = self.transform_source_pointcloud(latent_shape_params)
        deformed_source_scatter = go.Scatter3d(
            x=deformed_points[:, 0],
            y=deformed_points[:, 1],
            z=deformed_points[:, 2],
            mode='markers',
            name='deformed_source_points',
            marker=dict(size=2, color='rgba(0, 0, 255, 1.0)'))

        # Create scatters for training instances.
        instance_scatters = []
        for instance_name in show_training_instances:
            instance_points = self.instance_pointclouds[instance_name]
            instance_scatters.append(go.Scatter3d(
            x=instance_points[:, 0],
            y=instance_points[:, 1],
            z=instance_points[:, 2],
            mode='markers',
            name=instance_name,
            marker=dict(size=2, color='rgba(255, 0, 0, 1.0)')))

        # Create scatter for deformed keypoints.
        deformed_keypoints = self.transform_keypoints(
            latent_shape_params, keypoint_group='hand_bodies')
        deformed_keypoints_scatter = go.Scatter3d(
            x=deformed_keypoints[:, 0],
            y=deformed_keypoints[:, 1],
            z=deformed_keypoints[:, 2],
            opacity=float(show_keypoints),
            mode='markers',
            marker=dict(size=5, color='rgba(205, 73, 0, 1.0)'),
            name='keypoints'
        )
        fig = go.Figure([deformed_source_scatter, deformed_keypoints_scatter] + instance_scatters)

        if self.tool_category == 'drill':
            x_range = [-0.325, 0.05]
            y_range = [-0.1875, 0.1875]
            z_range = [-0.325, 0.05]
        elif self.tool_category == 'mug':
            x_range = [-0.1, 0.1]
            y_range = [-0.1, 0.1]
            z_range = [-0.01, 0.19]
        elif self.tool_category == 'hammer':
            x_range = [-0.35, 0.05]
            y_range = [-0.2, 0.2]
            z_range = [-0.35, 0.05]

        camera = dict(
            eye=dict(x=0., y=1.5, z=0.)
        )

        fig.update_layout(
            showlegend=False,
            scene=dict(
                xaxis=dict(range=x_range, ),
                yaxis=dict(range=y_range, ),
                zaxis=dict(range=z_range, ),),
            height=800,
            scene_camera=camera,
        )

        fig.update_scenes(aspectratio=dict(x=1, y=1, z=1))

        if not show_axes:
            fig.update_scenes(xaxis_visible=False, yaxis_visible=False,
                              zaxis_visible=False)

        if show:
            fig.show()
        return fig

    def draw_manipulator_mesh(self, latent_shape_params: torch.Tensor,
                              show: bool = True) -> List[go.Mesh3d]:

        def transformation_matrix(pos, quat,
                                  reorder_quat: bool = True) -> np.array:
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

        target_keypoints = self.transform_keypoints(
            latent_shape_params, keypoint_group='hand_bodies')

        manipulator_rigid_body_poses = self.optimize_joints_for_keypoints(
            target_keypoints)

        self._manipulator_bodies_trimesh = \
            self.manipulator_model.model.bodies_as_trimesh()

        meshes_visible, meshes_invisible = [], []
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

                transform = transformation_matrix(
                    manipulator_rigid_body_poses[body_name][
                        'pos'] + body_offset,
                    manipulator_rigid_body_poses[body_name]['quat'])

                mesh = body_collision_mesh['trimesh'].apply_transform(transform)


                meshes_visible.append(go.Mesh3d(
                    x=mesh.vertices[:, 0],
                    y=mesh.vertices[:, 1],
                    z=mesh.vertices[:, 2],
                    i=mesh.faces[:, 0],
                    j=mesh.faces[:, 1],
                    k=mesh.faces[:, 2],
                    opacity=0.75,
                    color='rgba(200, 250, 200, 0.75)'
                ))
                meshes_invisible.append(go.Mesh3d(
                    x=mesh.vertices[:, 0],
                    y=mesh.vertices[:, 1],
                    z=mesh.vertices[:, 2],
                    i=mesh.faces[:, 0],
                    j=mesh.faces[:, 1],
                    k=mesh.faces[:, 2],
                    opacity=0.0,
                ))

        return meshes_visible, meshes_invisible

    def visualize(
            self,
            target_instance_to_show: str = 'concept_drill',
            keypoints_to_show: str = 'hand_bodies',
    ) -> None:
        def on_click(event):
            if event.button == 1:
                x_click, y_click = event.xdata, event.ydata
                if None not in [x_click, y_click]:
                    draw_latent_space(x_click, y_click)
                    draw_pointcloud(x_click, y_click)
                    plt.draw()

        def draw_latent_space(x_click, y_click):
            latent_space_ax.clear()
            latent_space_ax.set_title('Latent (shape) space')
            for instance in self.target_instances:
                latent_space_ax.scatter(
                    instance.latent_shape_params[0], instance.latent_shape_params[1])
                latent_space_ax.annotate(
                    instance.name,
                    (instance.latent_shape_params[0], instance.latent_shape_params[1]))
            latent_space_ax.scatter(x_click, y_click, color='red', marker='x')

        def draw_pointcloud(x_click, y_click):
            latent_space_params = torch.Tensor([x_click, y_click])

            deformation_feature_vector = torch.matmul(
                latent_space_params, self.principal_deformations.T)
            deformation_field = deformation_feature_vector.reshape(
                -1, 3).numpy()
            deformed_source_pc = self.source_pc.points + np.dot(
                source_gaussian_kernel, deformation_field)
            pointcloud_ax.clear()
            pointcloud_ax.set_title('Point-cloud')
            pointcloud_ax.scatter(deformed_source_pc[:, 0],
                                  deformed_source_pc[:, 1],
                                  deformed_source_pc[:, 2])

            if target_instance_to_show is not None:
                pointcloud_ax.scatter(target_pc.points[:, 0],
                                      target_pc.points[:, 1],
                                      target_pc.points[:, 2])

            if keypoints_to_show is not None:
                pointcloud_ax.scatter(canonical_keypoints[:, 0],
                                      canonical_keypoints[:, 1],
                                      canonical_keypoints[:, 2],
                                      c='yellow', s=12)

            pointcloud_ax.set_xlim(-0.3, 0.05)
            pointcloud_ax.set_ylim(-0.175, 0.175)
            pointcloud_ax.set_zlim(-0.3, 0.05)

        source_gaussian_kernel = pycpd.gaussian_kernel(
            self.source_pc.points, self.beta, self.source_pc.points)

        if target_instance_to_show is not None:
            for instance in self.target_instances:
                if instance.name == target_instance_to_show:
                    target_pc = instance.sample_pointcloud(rgb=(255, 0, 0))

        if keypoints_to_show is not None:
            canonical_keypoints = self.demo_pose[
                keypoints_to_show + '_demo_pos']

        fig = plt.figure()
        latent_space_ax = fig.add_subplot(1, 2, 1)
        pointcloud_ax = fig.add_subplot(1, 2, 2, projection='3d')
        draw_latent_space(0, 0)
        draw_pointcloud(0, 0)

        plt.connect('motion_notify_event', on_click)
        plt.connect('button_press_event', on_click)

        plt.show()


if __name__ == "__main__":
    gym = gymapi.acquire_gym()
    sim_params = gymapi.SimParams()
    sim = gym.create_sim(0, -1,
                         gymapi.SIM_PHYSX, sim_params)
    shape_space = DexterityCategory(
        gym=gym,
        sim=sim,
        demo_path='/assets/dexterity/tools/drill/black_and_decker_unknown/right_schunk_sih_hand_demo_pose.npz',
        asset_root='../../../../assets/dexterity/tools/drill/train',
        source_file='black_and_decker_unknown/rigid.urdf',
        target_files=[
            'bosch_gsb180_li/rigid.urdf',
            'bosch_psr1800_li/rigid.urdf',
            'concept_drill/rigid.urdf',
            'dewalt_dcd780/rigid.urdf',
            'dewalt_unknown/rigid.urdf',
            'makita_bhp451/rigid.urdf',
            'makita_unknown/rigid.urdf',
            'makita_xfd15/rigid.urdf',
            'porter_cable_unknown/rigid.urdf',
            'solidworks_arp/rigid.urdf',
        ]
    )

    shape_space.build()

    shape_space.to_file()

    #shape_space.draw(latent_shape_params=torch.Tensor([0., 0.]))
    #shape_space.draw_latent_space(latent_shape_params=torch.Tensor([0., 0.]))
    #shape_space.draw_deformed_pointcloud(latent_shape_params=torch.Tensor([0., 0.]))
    #shape_space.draw_manipulator_mesh(latent_shape_params=torch.Tensor([0., 0.]))
