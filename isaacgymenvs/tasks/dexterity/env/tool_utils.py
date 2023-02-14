import isaacgym
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
from tqdm import tqdm
import trimesh
from trimesh import viewer
from typing import *
from urdfpy import URDF
import plotly.graph_objects as go
import mujoco
import mujoco_viewer
import nevergrad as ng
from scipy.spatial.transform import Rotation as R
from plotly.subplots import make_subplots


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

        ctrl = self.demo_pose['residual_actuated_dof_demo_pos'] + delta_ctrl

        for i in range(sim_steps):
            self.step(ik_body_pos, ik_body_quat, ctrl)

        resulting_keypoint_pos = []
        for site_name in self._keypoint_site_names:
            resulting_keypoint_pos.append(self._mj_data.site(site_name).xpos)
        resulting_keypoint_pos = np.array(resulting_keypoint_pos)

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

    def move_to_canonical_pose(self, sim_steps: int = 256) -> None:
        for i in range(sim_steps):
            self.step(self.demo_pose['ik_body_demo_pos'],
                      self.ik_body_quat,
                      self.demo_pose['residual_actuated_dof_demo_pos'])

    def optimize_joints_for_keypoints(self, budget: int = 128) -> None:
        self.move_to_canonical_pose()


        parameterization = ng.p.Instrumentation(
            delta_ik_body_pos=ng.p.Array(shape=(3,)).set_bounds(-0.05, 0.05),
            delta_ik_body_rot=ng.p.Array(shape=(3,)).set_bounds(-0.5, 0.5),
            delta_ctrl=ng.p.Array(
                shape=(self.manipulator_robot.model.num_actions,)).set_bounds(
                -0.5, 0.5))

        optimizer = ng.optimizers.NGOpt(parameterization, budget=budget)
        print("optimizer:", optimizer)
        recommendation = optimizer.minimize(self.loss)



class DexterityInstance:
    def __init__(
            self,
            asset_root: str,
            asset_file: str
    ) -> None:
        self.urdf = URDF.load(os.path.join(asset_root, asset_file))
        self.collision_mesh = self.assemble_collision_mesh()
        self.name = os.path.dirname(asset_file)
        self._latent_shape_params = None

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


class DexterityCategory:
    def __init__(
            self,
            asset_root: str,
            source_file: str,
            target_files: List[str],
            alpha: float = 3.,
            beta: float = 1.5,
            max_iterations: int = 10,
    ) -> None:
        self.asset_root = asset_root
        self.source_file = source_file
        self.target_files = target_files
        self.alpha = alpha
        self.beta = beta
        self.max_iterations = max_iterations

        self.demo_pose_npz = np.load(
            '/home/user/mosbach/PycharmProjects/dexterity/assets/dexterity/tools/drills/train/black_and_decker_unknown/right_schunk_sih_hand_demo_pose.npz')
        self.demo_pose = {}
        for k, v in self.demo_pose_npz.items():
            self.demo_pose[k] = v[0]

    def build(self) -> None:
        self.acquire_instances(self.asset_root, self.source_file, self.target_files)
        self.deformation_design_matrix = self.acquire_deformation_fields(
            self.alpha, self.beta, self.max_iterations)
        self.principal_deformations = self.find_principal_deformations(
            num_latents=2)

        d = {
            'first_principal_component': [],
            'second_principal_component': [],
            'name': []
        }
        for instance in self.target_instances:
            d['first_principal_component'].append(
                instance.latent_shape_params[0].cpu().numpy())
            d['second_principal_component'].append(
                instance.latent_shape_params[1].cpu().numpy())
            d['name'].append(instance.name)
        self.latent_space_df = pd.DataFrame(data=d)

        self.source_gaussian_kernel = pycpd.gaussian_kernel(
            self.source_pc.points, self.beta, self.source_pc.points)

    def to_file(self, save_path: str = './category') -> None:

        # Create DataFrame from latent params of target instances
        self.latent_space_df.to_csv(save_path + '_latent_space.csv')
        np.save(save_path + '_principal_deformations.npy',
                self.principal_deformations.cpu().numpy())
        np.save(save_path + '_source_points.npy', self.source_pc.points)

    def from_file(self, load_path: str = './category'):
        self.principal_deformations = torch.from_numpy(
            np.load(load_path + '_principal_deformations.npy')).to(torch.float32)
        self.source_pc = DexterityPointCloud(
            points=np.load(load_path + '_source_points.npy'), rgb=(0, 0, 255))
        self.latent_space_df = pd.read_csv(load_path + '_latent_space.csv')
        self.source_gaussian_kernel = pycpd.gaussian_kernel(
            self.source_pc.points, self.beta, self.source_pc.points)

    def acquire_instances(self, asset_root: str, source_file: str, target_files: List[str]) -> None:
        # Create DexterityInstances from the asset files
        self.target_instances = []
        pbar = tqdm([source_file, ] + target_files)
        for i, instance_file in enumerate(pbar):
            pbar.set_description(f"Acquiring instance '{os.path.dirname(instance_file)}'")

            if i == 0:
                self.source_instance = DexterityInstance(asset_root,
                                                         instance_file)
            else:
                self.target_instances.append(
                    DexterityInstance(asset_root, instance_file))

        # Load demonstration for the canonical instance
        canonical_demo_path = os.path.join(
            asset_root, os.path.dirname(source_file),
            MANIPULATOR_MODEL + '_demo_pose.npz')
        assert os.path.isfile(canonical_demo_path), \
            f"Tried to load canonical demo pose for " \
            f"{MANIPULATOR_MODEL}, but " \
            f"{canonical_demo_path} was not found."
        self.canonical_demo_dict = np.load(canonical_demo_path)

    def acquire_deformation_fields(self, alpha: float, beta: float, max_iterations: int = 100) -> torch.Tensor:
        self.source_pc = self.source_instance.sample_pointcloud(rgb=(0, 0, 255))
        deformation_field_vectors = []
        pbar = tqdm(self.target_instances)
        for target_instance in pbar:
            pbar.set_description(f"Acquiring deformation from '{self.source_instance.name}' to '{target_instance.name}'")
            deformation_field = self.register_instance(target_instance, alpha, beta, max_iterations)
            deformation_field_vectors.append(deformation_field.flatten())
        return torch.from_numpy(np.array(deformation_field_vectors)).to(torch.float32)

    def register_instance(self, target_instance: DexterityInstance, alpha: float, beta: float, max_iterations: int = 100, show: bool = False) -> np.array:
        target_pc = target_instance.sample_pointcloud(rgb=(255, 0, 0))
        registration = DexterityDeformableRegistration(
            target_pc, deepcopy(self.source_pc), alpha, beta, max_iterations, show)
        G, W = registration.get_registration_parameters()
        return W.flatten()

    def find_principal_deformations(self, num_latents: int) -> torch.Tensor:
        U, S, V = torch.pca_lowrank(self.deformation_design_matrix)
        principal_deformations = V[:, :num_latents]

        # Set latent shape params of target instances.
        for deformation_vector, instance in zip(self.deformation_design_matrix, self.target_instances):
            instance.latent_shape_params = torch.matmul(deformation_vector, principal_deformations)
        return principal_deformations

    def find_latent_params(self, points: np.array):
        # Find parameters in shape space for a newly observed instance
        pass

    def transform_source_pointcloud(self, latent_space_params: torch.Tensor
                                    ) -> np.array:
        deformation_feature_vector = torch.matmul(
            latent_space_params, self.principal_deformations.T)
        deformation_field = deformation_feature_vector.reshape(
            -1, 3).numpy()
        deformed_source_pc = self.source_pc.points + np.dot(
            self.source_gaussian_kernel, deformation_field)
        return deformed_source_pc

    def transform_keypoints(self, latent_space_params: torch.Tensor,
                            keypoint_group: str) -> np.array:
        deformation_feature_vector = torch.matmul(
            latent_space_params, self.principal_deformations.T)
        deformation_field = deformation_feature_vector.reshape(
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
                                 show: bool = False) -> go.Figure:
        deformed_points = self.transform_source_pointcloud(latent_shape_params)
        deformed_source_scatter = go.Scatter3d(
            x=deformed_points[:, 0],
            y=deformed_points[:, 1],
            z=deformed_points[:, 2],
            mode='markers',
            name='deformed_source_points',
            marker=dict(size=3, ))

        deformed_keypoints = self.transform_keypoints(
            latent_shape_params, keypoint_group='hand_bodies')
        deformed_keypoints_scatter = go.Scatter3d(
            x=deformed_keypoints[:, 0],
            y=deformed_keypoints[:, 1],
            z=deformed_keypoints[:, 2],
            opacity=float(show_keypoints),
            mode='markers',
            marker=dict(size=6, ),
            name='keypoints'
        )
        fig = go.Figure([deformed_source_scatter, deformed_keypoints_scatter])


        fig.update_layout(
            showlegend=False,
            scene=dict(
                xaxis=dict(range=[-0.325, 0.05], ),
                yaxis=dict(range=[-0.075, 0.075], ),
                zaxis=dict(range=[-0.325, 0.05], ),
                aspectmode='data'
            ),
            height=800,
        )

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
                    opacity=1.0,
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
            canonical_keypoints = self.canonical_demo_dict[
                keypoints_to_show + '_demo_pos'][0]

        fig = plt.figure()
        latent_space_ax = fig.add_subplot(1, 2, 1)
        pointcloud_ax = fig.add_subplot(1, 2, 2, projection='3d')
        draw_latent_space(0, 0)
        draw_pointcloud(0, 0)

        plt.connect('motion_notify_event', on_click)
        plt.connect('button_press_event', on_click)

        plt.show()


if __name__ == "__main__":
    shape_space = DexterityCategory(
        asset_root='../../../../assets/dexterity/tools/drills/train',
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