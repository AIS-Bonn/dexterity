"""Dexterity: abstract base class for DexterityRobot class

Inherits ABC class. Inherited by DexterityRobot class. Defines template for
DexterityRobot class.
"""

from abc import ABC, abstractmethod
from typing import *


class DexterityABCRobot(ABC):
    """Represents a robot model.
    """

    @abstractmethod
    def __init__(self, model_root, xml_files: List[str]) -> None:
        pass

    @abstractmethod
    def as_asset(self, gym, sim, asset_options):
        pass
