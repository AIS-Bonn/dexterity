from isaacgym import gymapi, gymutil
from isaacgym.torch_utils import *
from typing import *


class DexterityBaseProperties:
    @property
    def arm_eef_euler(self) -> torch.Tensor:
        return torch.stack(get_euler_xyz(self.arm_eef_quat)).transpose(0, 1)