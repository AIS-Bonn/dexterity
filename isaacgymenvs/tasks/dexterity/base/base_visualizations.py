from isaacgym import gymapi, gymutil
import matplotlib.cm as cm
import torch
from typing import *


class DexterityBaseVisualizations:
    def draw_visualizations(self, visualizations: List[str]) -> None:
        for env_id in range(self.num_envs):
            for visualization in visualizations:
                split_visualization = visualization.split('_')
                # Visualize positions with sphere geom.
                if split_visualization[-1] == 'pos':
                    self.visualize_pos(visualization, env_id)
                # Visualize poses with axes geom.
                elif split_visualization[-1] == 'pose':
                    self.visualize_pose(visualization, env_id)
                # Call any other visualization functions (e.g. workspace extent, etc.).
                else:
                    getattr(self, "visualize_" + visualization)(env_id)

    def visualize_pos(self, data: Union[str, torch.Tensor], env_id: int, sphere_size: float = 0.003,
                      color: Tuple[float, float, float] = (1, 1, 0)) -> None:

        if isinstance(data, str):
            pos = getattr(self, data)[env_id]
        else:
            pos = data[env_id]

        if len(pos.shape) == 1:
            pos = pos.unsqueeze(0)

        for point_idx in range(pos.shape[0]):
            pose = gymapi.Transform(gymapi.Vec3(*pos[point_idx]))
            sphere_geom = gymutil.WireframeSphereGeometry(
                sphere_size, 12, 12, gymapi.Transform(), color=color)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer,
                               self.env_ptrs[env_id], pose)

    def visualize_pose(self, name: str, env_id: int, axis_length: float = 0.2, sphere_size: float = None):
        split_name = name.split("_")
        body_name = "_".join(split_name[:-1])
        pos = getattr(self, body_name + "_pos")[env_id]
        quat = getattr(self, body_name + "_quat")[env_id]

        if len(pos.shape) == 1:
            pos = pos.unsqueeze(0)
            quat = quat.unsqueeze(0)
        
        for point_idx in range(pos.shape[0]):
            pose = gymapi.Transform(gymapi.Vec3(*pos[point_idx]), gymapi.Quat(*quat[point_idx]))
            axes_geom = gymutil.AxesGeometry(axis_length)
            gymutil.draw_lines(axes_geom, self.gym, self.viewer, self.env_ptrs[env_id], pose)

            if sphere_size is not None:
                sphere_geom = gymutil.WireframeSphereGeometry(
                sphere_size, 12, 12, gymapi.Transform(), color=(1, 1, 0))
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer,
                                self.env_ptrs[env_id], pose)

    def visualize_polygon(self, name: str, env_id: int):
        points = getattr(self, name)[env_id]

        for start_idx in range(points.shape[0]):
            end_idx = start_idx + 1 if start_idx < points.shape[0] - 1 else 0
            start_point = gymapi.Vec3(*points[start_idx])
            end_point = gymapi.Vec3(*points[end_idx])
            gymutil.draw_line(start_point, end_point, gymapi.Vec3(1, 1, 0),
                              self.gym, self.viewer, self.env_ptrs[env_id])

    def visualize_ik_body_workspace(self, env_id: int) -> None:
        # Set extent in z-direction to 0
        extent = torch.tensor(self.cfg_base.ctrl.workspace.pos)
        bbox = gymutil.WireframeBBoxGeometry(extent, pose=gymapi.Transform(),
                                             color=(0, 1, 1))
        gymutil.draw_lines(bbox, self.gym, self.viewer, self.env_ptrs[env_id],
                           pose=gymapi.Transform())
        

    def visualize_robot_contact_forces(self, env_id: int, colormap: str = 'viridis', vmax: float = 10., arm_only: bool = False) -> None:
        """Visualize contact forces on the robot arm.
        
        Args:
            env_id (int): Environment ID.
            colormap (str, optional): Matplotlib colormap to use. Defaults to 'viridis'.
            vmax (float, optional): Maximum force to normalize by in Newton. Defaults to 10.
            arm_only (bool, optional): Whether to only visualize the arm. Defaults to False.
        """
        
        cmap = cm.get_cmap(colormap)
        contact_force = self.contact_force[env_id]
        contact_force_mag = torch.clamp(torch.norm(contact_force, dim=1) / vmax, max=1.0)
        rgb = cmap(contact_force_mag.cpu().numpy())[:, :3]
        num_rigid_bodies = self.robot_arm_rigid_body_count if arm_only else contact_force.shape[0]
        for rb_idx in range(num_rigid_bodies):
            self.gym.set_rigid_body_color(
                self.env_ptrs[env_id], self.robot_handles[env_id], rb_idx,
                gymapi.MESH_VISUAL,
                gymapi.Vec3(rgb[rb_idx, 0], rgb[rb_idx, 1], rgb[rb_idx, 2]))
