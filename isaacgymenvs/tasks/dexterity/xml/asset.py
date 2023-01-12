"""Dexterity: class for DexterityXMLAsset

Inherits DexterityXML class. Extends the DexterityXML class by providing
information about the IsaacGym asset represented by the XML model. Used by
DexterityRobot.
"""

from functools import partial
import os
from typing import *

from . import GYM_ASSET_API, SG_FINGER_IDX
from .xml import DexterityXML
from isaacgym import gymapi
from isaacgym.torch_utils import *


class DexterityXMLAsset(DexterityXML):
    def __init__(self, xml_path: str) -> None:
        super().__init__(xml_path)
        self._asset_created = False
        self._joint_equality_initialized = False
        self._teleop_mapping_initialized = False

    def create_asset(self, gym, sim, model_root) -> None:
        self._gym = gym
        with self.as_xml(os.path.join(model_root, "tmp_dexterity_asset.xml")):
            self.gym_asset = self._gym.load_asset(
                sim, model_root, "tmp_dexterity_asset.xml",
                gymapi.AssetOptions())
        self._asset_created = True

    def __getattr__(self, *args):
        # Return gym function run on asset for functions in the gym API
        if args[0] in GYM_ASSET_API:
            assert self._asset_created, \
                "Cannot access information about IsaacGym asset before " \
                "'create_asset' has been called."
            return partial(getattr(self._gym, args[0]), self.gym_asset)
        else:
            return self.__getattribute__(*args)

    def joint_equality(self,
                       ctrl_target_dof_pos: torch.Tensor
                       ) -> torch.Tensor:
        """Enforces joint equalities based on position targets according to
            equality tags in the XML model.

            Args:
                ctrl_target_dof_pos: (torch.Tensor) Position target of the
                    DoFs without equality (shape: [num_envs, num_dofs]).
            Returns:
                ctrl_target_dof_pos_eq: (torch.Tensor) Position targets of DoFs
                    with enforced equalities (shape: [num_envs, num_dofs]).
        """
        if len(self.equality) == 0:
            return ctrl_target_dof_pos

        if not self._joint_equality_initialized:
            self._init_joint_equality(ctrl_target_dof_pos.device)
            self._joint_equality_initialized = True

        x = ctrl_target_dof_pos[:, self.parent_joint_ids]
        ctrl_target_dof_pos[:, self.child_joint_ids] = self.c0 + self.c1 * x
        return ctrl_target_dof_pos

    def _init_joint_equality(self, device: torch.device) -> None:
        assert self._asset_created, \
            "Cannot access information about IsaacGym asset before " \
            "'create_asset' has been called."

        parent_joint_names, child_joint_names, \
            joint_equality_polycoef = [], [], []
        for joint_equality in self.equality:
            assert joint_equality.tag == 'joint'

            parent_joint_names.append(joint_equality.attrib['joint1'])
            child_joint_names.append(joint_equality.attrib['joint2'])
            joint_equality_polycoef.append(
                list(map(float, joint_equality.attrib['polycoef'].split(" "))))

        parent_joint_ids, child_joint_ids = [], []
        for parent_name, child_name in \
                zip(parent_joint_names, child_joint_names):
            parent_joint_ids.append(self.find_asset_dof_index(parent_name))
            child_joint_ids.append(self.find_asset_dof_index(child_name))

        self.parent_joint_ids = to_torch(parent_joint_ids, dtype=torch.long,
                                         device=device)
        self.child_joint_ids = to_torch(child_joint_ids, dtype=torch.long,
                                        device=device)
        self.joint_equality_polycoef = to_torch(joint_equality_polycoef,
                                                device=device)

        self.c0 = self.joint_equality_polycoef[:, 0]
        self.c1 = self.joint_equality_polycoef[:, 1]

    def teleop_mapping(self,
                       sg_sensor_data: np.array,
                       sg_flexions: np.array
                       ) -> Dict[str, Any]:

        def get_sensor_data(sensor_name, sensor_angles, flexions) -> float:
            hand, finger, sensor = sensor_name.split("_")
            finger_idx = SG_FINGER_IDX[finger]
            if sensor == "flexion":
                return flexions[finger_idx]
            else:
                assert sensor.startswith("angle")
                angle_idx = int(sensor[-1])
                return sensor_angles[finger_idx][angle_idx]

        if not self._teleop_mapping_initialized:
            self._init_teleop_mapping()
            self._teleop_mapping_initialized = True

        actuation_values = {}
        for teleop_mapping in self._teleop_mapping:
            sg_sensor = teleop_mapping['sg_sensor']
            target_joint = teleop_mapping['target_joint']
            polycoef = teleop_mapping['polycoef']
            actuation_values[target_joint] = 0.
            sensor_value = get_sensor_data(sg_sensor, sg_sensor_data,
                                           sg_flexions)
            for i in range(polycoef.shape[0]):
                actuation_values[target_joint] += \
                    polycoef[i] * pow(sensor_value, i)
        return actuation_values

    def _init_teleop_mapping(self):
        self._teleop_mapping = self.get_teleop_mapping()

