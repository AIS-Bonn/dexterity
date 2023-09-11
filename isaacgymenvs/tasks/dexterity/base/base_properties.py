from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import *
from typing import *


class DexterityBaseProperties:
    @property
    def arm_eef_euler(self) -> torch.Tensor:
        return torch.stack(get_euler_xyz(self.arm_eef_quat)).transpose(0, 1)

    @property
    def base_rigid_bodies(self) -> int:
        """Returns number of rigid bodies for base components present in all
        environments, such as the robot, table and camera bodies."""
        return self.robot.rigid_body_count + \
               self.table_rigid_body_count + \
               self.camera_rigid_body_count

    @property
    def base_rigid_shapes(self) -> int:
        """Returns number of rigid shapes for base components present in all
        environments, such as the robot, table and camera bodies."""
        return self.robot.rigid_shape_count + \
               self.table_rigid_shape_count + \
               self.camera_rigid_shape_count
