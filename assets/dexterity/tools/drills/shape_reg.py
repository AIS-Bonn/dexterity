import matplotlib.pyplot as plt
import numpy as np
import pycpd
import os
import threading
import torch
from tqdm import tqdm
import trimesh
from trimesh import viewer
from typing import *
from urdfpy import URDF

MANIPULATOR_MODEL = "right_schunk_sih_hand"


def standardize_data(arr):
    '''
    This function standardize an array, its substracts mean value,
    and then divide the standard deviation.

    param 1: array
    return: standardized array
    '''
    rows, columns = arr.shape

    standardizedArray = np.zeros(shape=(rows, columns))
    tempArray = np.zeros(rows)

    for column in range(columns):

        mean = np.mean(X[:, column])
        std = np.std(X[:, column])
        tempArray = np.empty(0)

        for element in X[:, column]:
            tempArray = np.append(tempArray, ((element - mean) / std))

        standardizedArray[:, column] = tempArray

    return standardizedArray


def get_collision_mesh(urdf_path: str) -> trimesh.Trimesh:
    urdf_model = URDF.load(urdf_path)
    collision_meshes = []
    for link in urdf_model.links:
        if link.collision_mesh is not None:
            collision_meshes.append(link.collision_mesh)
    return trimesh.util.concatenate(collision_meshes)


def get_pointcloud(urdf_path: str, count: int = 512,
                   rgba: Tuple[int, int, int, int] = (0, 0, 0, 255),
                   name: str = None
                   ) -> trimesh.points.PointCloud:
    collision_mesh = get_collision_mesh(urdf_path)
    points = np.array(trimesh.sample.sample_surface(
        collision_mesh, count=count)[0]).astype(float)
    colors = np.array([rgba]).repeat(count, axis=0)
    return trimesh.points.PointCloud(points, colors=colors,
                                     metadata={'name': name})


def load_pointclouds(canonical: str):
    train_root = './train'
    train_models = [f.name for f in os.scandir(train_root) if f.is_dir()]
    train_models.sort()

    target_pointclouds = []
    for model in train_models:
        model_path = os.path.join(train_root, f"{model}/drill.urdf")
        if model == canonical:
            source_pointcloud = get_pointcloud(
                model_path, rgba=(0, 0, 255, 255), name=model)
        else:
            target_pointclouds.append(get_pointcloud(
                model_path, rgba=(255, 0, 0, 255), name=model))

    return source_pointcloud, target_pointclouds


def load_canonical_demonstration():
    # Find and load the correct demonstration
    canonical_demo_path = os.path.join(
        "./canonical", MANIPULATOR_MODEL + '_demo_pose.npz')
    assert os.path.isfile(canonical_demo_path), \
        f"Tried to load canonical demo pose for " \
        f"{MANIPULATOR_MODEL}, but " \
        f"{canonical_demo_path} was not found."
    canonical_demo_pose_dict = np.load(canonical_demo_path)
    return canonical_demo_pose_dict


class ShapeSpace:
    def __init__(self, alpha: float = 3., beta: float = 1.5) -> None:
        self.alpha = alpha
        self.beta = beta

    def build_shape_space(self) -> None:
        source_pointcloud, target_pointclouds = load_pointclouds(
            canonical='bosch_gsb180_li')
        deformation_feature_vectors = []
        for target_pointcloud in tqdm(target_pointclouds):
            deformation_feature_vectors.append(self.register_instance(
                source_pointcloud, target_pointcloud))

        self.visualize_shape_space(source_pointcloud, target_pointclouds,
                                   deformation_feature_vectors)



    def register_instance(self, source_pointcloud, target_pointcloud,
                          show: bool = False):
        print(f"Fitting {source_pointcloud.metadata['name']} to "
              f"{target_pointcloud.metadata['name']}.")
        self.target_pointcloud = target_pointcloud
        self.deformed_source_pointcloud = source_pointcloud

        if show:
            viewer_thread = threading.Thread(target=self.run_trimesh_viewer)
            viewer_thread.start()

        registration = pycpd.DeformableRegistration(
            X=np.array(self.target_pointcloud.vertices),
            Y=np.array(self.deformed_source_pointcloud.vertices),
            alpha=3,
            beta=1.5,
            max_iterations=10,
            tolerance=0.0)
        registration.register(self.registration_callback)
        G, W = registration.get_registration_parameters()

        if show:
            viewer_thread.join()

        return W.flatten()

    def update_viewer(self, scene) -> None:
        scene.delete_geometry('deformed_source')
        scene.add_geometry(self.deformed_source_pointcloud,
                           node_name='deformed_source')

    def registration_callback(self, iteration: int, error: float, X, Y):
        count = 512
        colors = np.array([[0, 0, 255, 255]]).repeat(count, axis=0)
        self.deformed_source_pointcloud = trimesh.points.PointCloud(
            Y, colors=colors)

    def run_trimesh_viewer(self):
        tm = np.array([[-1., 0., 0., -0.14],
                       [0., 0., 1., 0.3],
                       [0., 1., 0., -0.1],
                       [0., 0., 0., 1.]])

        camera = trimesh.scene.cameras.Camera(fov=(70, 70))
        scene = trimesh.scene.Scene(camera=camera, camera_transform=tm)
        # add target point cloud to scene
        scene.add_geometry(self.target_pointcloud, node_name='target')
        viewer.SceneViewer(scene, callback=self.update_viewer)

    def visualize_shape_space(self, source_pointcloud, target_pointclouds,
                              deformation_feature_vectors) -> None:
        source_points = np.array(source_pointcloud.vertices)
        G = pycpd.gaussian_kernel(source_points, self.beta, source_points)

        # Perform PCA on deformation field feature vectors.
        deformation_feature_vectors = torch.from_numpy(
            np.array(deformation_feature_vectors)).to(torch.float32)
        U, S, V = torch.pca_lowrank(deformation_feature_vectors)
        variance_explained = []
        for eigen_val in S:
            variance_explained.append((eigen_val / sum(S)) * 100)
        print("variance_explained:", variance_explained)
        V_reduced = V[:, :2]

        # Infer latent positions of the training instances.
        latent_x, latent_y = [], []
        for deformation_feature_vector in deformation_feature_vectors:
            latent_pos = torch.matmul(deformation_feature_vector, V_reduced)
            latent_x.append(latent_pos[0])
            latent_y.append(latent_pos[1])



        fig = plt.figure()
        fig.suptitle('Test title')

        latent_space_ax = fig.add_subplot(1, 2, 1)
        latent_space_ax.set_title('Latent (shape) space')
        latent_space_ax.scatter(x=latent_x,y=latent_y)
        for x, y, label in zip(
                latent_x, latent_y, [
                    tpc.metadata['name'] for tpc in target_pointclouds]):
            latent_space_ax.annotate(label, (x, y))

        pointcloud_ax = fig.add_subplot(1, 2, 2, projection='3d')
        pointcloud_ax.set_title('Point-cloud')
        pointcloud_ax.scatter(source_points[:, 0],
                              source_points[:, 1],
                              source_points[:, 2])
        pointcloud_ax.set_xlim(-0.3, 0.05)
        pointcloud_ax.set_ylim(-0.175, 0.175)
        pointcloud_ax.set_zlim(-0.3, 0.05)

        def on_click(event):
            if event.button == 1:
                x, y = event.xdata, event.ydata


                latent_space_ax.clear()
                latent_space_ax.set_title('Latent (shape) space')
                latent_space_ax.scatter(x=latent_x, y=latent_y)
                latent_space_ax.scatter(x, y, color='red', marker='x')
                
                for lat_x, lat_y, label in zip(
                        latent_x, latent_y, [
                            tpc.metadata['name'] for tpc in
                            target_pointclouds]):
                    latent_space_ax.annotate(label, (lat_x, lat_y))

                point_in_latent_space_space = torch.Tensor([x, y])
                deformation_field = torch.matmul(
                    point_in_latent_space_space, V_reduced.T)

                deformation_field = deformation_field.reshape(-1, 3).numpy()

                deformed_pointcloud = source_points + np.dot(G, deformation_field)

                pointcloud_ax.clear()
                pointcloud_ax.set_title('Point-cloud')
                pointcloud_ax.scatter(deformed_pointcloud[:, 0],
                                      deformed_pointcloud[:, 1],
                                      deformed_pointcloud[:, 2])
                pointcloud_ax.set_xlim(-0.3, 0.05)
                pointcloud_ax.set_ylim(-0.175, 0.175)
                pointcloud_ax.set_zlim(-0.3, 0.05)

                plt.draw()

        plt.connect('motion_notify_event', on_click)
        plt.connect('button_press_event', on_click)

        plt.show()





shape_space = ShapeSpace()
shape_space.build_shape_space()
