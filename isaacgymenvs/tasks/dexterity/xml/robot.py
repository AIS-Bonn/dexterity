"""Dexterity: class for DexterityRobot

Collects DexterityXMLAssets in one robot model and provides information about
its components.
"""

from copy import deepcopy
import os
import torch
from typing import *

from .asset import DexterityXMLAsset
from .schema_class_robot import DexterityABCRobot


class DexterityRobot(DexterityABCRobot):

    def __init__(self, model_root, xml_files: List[str]) -> None:
        self._model_root = model_root

        assert len(xml_files) > 0, \
            "At least one XML-file must given to construct a robot."

        # Create XMLAssets from files
        self.xml_assets = [
            DexterityXMLAsset(os.path.join(model_root, xml_files[i]))
            for i in range(len(xml_files))]

        # Infer and check robot types
        self.robot_types = [m.robot_type for m in self.xml_assets]
        assert len(set(self.robot_types)) == len(self.robot_types), \
            f"Duplicate robot types in {self.robot_types}."
        #assert "arm" in self.robot_types, \
        #    f"'arm' should be in robot types {self.robot_types}."

        # Set robot models as class attributes for easier access to properties
        for i, robot_type in enumerate(self.robot_types):
            setattr(self, robot_type, self.xml_assets[i])

        # Build complete model from components
        self.model = deepcopy(self.xml_assets[0])
        for xml_asset in self.xml_assets[1:]:
            if xml_asset.robot_type == "visual_asset":
                order = "insert"
            else:
                order = "append"
            self.model.attach(deepcopy(xml_asset), order=order)

    def as_asset(self, gym, sim, asset_options):
        self._gym = gym
        self.model.create_asset(gym, sim, self._model_root)
        with self.model.as_xml(
                os.path.join(self._model_root, "robot.xml")):
            self._asset = gym.load_asset(sim, self._model_root,
                                         "robot.xml", asset_options)
        return self._asset

    @property
    def rigid_body_count(self) -> int:
        return self._gym.get_asset_rigid_body_count(self._asset)

    @property
    def rigid_shape_count(self) -> int:
        return self._gym.get_asset_rigid_shape_count(self._asset)
