from isaacgym import gymapi, gymutil
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

    def visualize_pos(self, name: str, env_id: int, sphere_size: float = 0.003):
        pos = getattr(self, name)[env_id]

        if len(pos.shape) == 1:
            pos = pos.unsqueeze(0)

        for point_idx in range(pos.shape[0]):
            pose = gymapi.Transform(gymapi.Vec3(*pos[point_idx]))
            sphere_geom = gymutil.WireframeSphereGeometry(
                sphere_size, 12, 12, gymapi.Transform(), color=(1, 1, 0))
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer,
                               self.env_ptrs[env_id], pose)

    def visualize_pose(self, name: str, env_id: int, axis_length: float = 0.3, sphere_size: float = None):
        split_name = name.split("_")
        body_name = "_".join(split_name[:-1])
        pos = getattr(self, body_name + "_pos")[env_id]
        quat = getattr(self, body_name + "_quat")[env_id]
        assert all(obs in self.cfg['env']['observations']
                   for obs in [f'{body_name}_pos',
                               f'{body_name}_quat']), \
            f'Cannot visualize {body_name} pose if ' \
            f'{body_name}_pos and {body_name}_quat' \
            f'are not part of the observations.'

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

