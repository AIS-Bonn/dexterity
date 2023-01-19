from isaacgym import gymapi, gymutil
from typing import *


class DexterityBaseVisualizations:
    def draw_visualizations(self, visualizations: List[str]) -> None:
        for env_id in range(self.num_envs):
            for visualization in visualizations:
                # For visualization of multiple keypoint poses
                if visualization.startswith(tuple(self.keypoint_dict.keys())):
                    split_visualization = visualization.split('_')
                    assert split_visualization[-1] == 'pose'
                    keypoint_group_name = '_'.join(split_visualization[:-1])
                    self.visualize_keypoint_body_pose(
                        keypoint_group_name, env_id)

                # For none-keypoint visualizations call the regular functions
                else:
                    getattr(self, "visualize_" + visualization)(env_id)

    def visualize_body_pose(self, body_name: str, env_id: int,
                            axis_length: float = 0.3, idx: int = None,
                            sphere_size: float = None,
                            ) -> None:
        body_pos = getattr(self, body_name + "_pos")[env_id]
        body_quat = getattr(self, body_name + "_quat")[env_id]
        if idx is not None:
            body_pos = body_pos[idx]
            body_quat = body_quat[idx]
        axes_geom = gymutil.AxesGeometry(axis_length)
        pos = gymapi.Vec3(*body_pos)
        quat = gymapi.Quat(*body_quat)
        pose = gymapi.Transform(pos, quat)
        gymutil.draw_lines(
            axes_geom, self.gym, self.viewer, self.env_ptrs[env_id], pose)

        if sphere_size is not None:
            sphere_geom = gymutil.WireframeSphereGeometry(
                sphere_size, 12, 12, gymapi.Transform(), color=(1, 1, 0))
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer,
                               self.env_ptrs[env_id], pose)

    def visualize_ik_body_pose(self, env_id: int, axis_length: float = 0.3
                               ) -> None:
        self.visualize_body_pose("ik_body", env_id, axis_length)

    def visualize_arm_eef_pose(self, env_id: int, axis_length: float = 0.3
                               ) -> None:
        self.visualize_body_pose("arm_eef", env_id, axis_length)

    def visualize_tracker_pose(self, env_id: int, axis_length: float = 0.3
                               ) -> None:
        self.visualize_body_pose("tracker", env_id, axis_length)

    def visualize_ctrl_target_ik_body_pose(self, env_id: int,
                                           axis_length: float = 3) -> None:
        self.visualize_body_pose("ctrl_target_ik_body", env_id, axis_length)

    def visualize_keypoint_body_pose(
            self, keypoint_group_name: str, env_id: int,
            axis_length: float = 0.05, sphere_size: float = 0.0025) -> None:
        assert all(obs in self.cfg['env']['observations']
                   for obs in [f'{keypoint_group_name}_pos',
                               f'{keypoint_group_name}_quat']), \
            f"Cannot visualize the {keypoint_group_name} poses if " \
            f"'{keypoint_group_name}_pos' and '{keypoint_group_name}_quat' " \
            f"are not part of the observations."

        keypoint_group_pos = getattr(self, keypoint_group_name + '_pos')
        keypoint_group_quat = getattr(self, keypoint_group_name + '_quat')

        for keypoint_body_idx in range(keypoint_group_pos.shape[1]):
            self.visualize_body_pose(keypoint_group_name, env_id, axis_length,
                                     keypoint_body_idx, sphere_size)

    def visualize_recorded_keypoint_pose(self, keypoint_pos, env_id: int,
                                         sphere_size: float = 0.0025):
        for keypoint_idx in range(keypoint_pos.shape[1]):
            pos = gymapi.Vec3(*keypoint_pos[env_id, keypoint_idx])
            quat = gymapi.Quat()
            pose = gymapi.Transform(pos, quat)
            sphere_geom = gymutil.WireframeSphereGeometry(
                sphere_size, 12, 12, gymapi.Transform(), color=(1, 1, 0))
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer,
                               self.env_ptrs[env_id], pose)

    def visualize_demo_keypoint_pose(self, keypoint_group_name, env_id: int,
                                         sphere_size: float = 0.0025):
        keypoint_pos = getattr(self, keypoint_group_name + '_pos_demo')
        for keypoint_idx in range(keypoint_pos.shape[1]):
            pos = gymapi.Vec3(*keypoint_pos[env_id, keypoint_idx])
            quat = gymapi.Quat()
            pose = gymapi.Transform(pos, quat)
            sphere_geom = gymutil.WireframeSphereGeometry(
                sphere_size, 12, 12, gymapi.Transform(), color=(1, 1, 0))
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer,
                               self.env_ptrs[env_id], pose)
