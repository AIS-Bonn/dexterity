"""Dexterity: abstract base class for DexterityXML class

Inherits ABC class. Inherited by DexterityXML class. Defines template for
DexterityXML class.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
import torch
from typing import *


class DexterityABCXML(ABC):
    """Base class that loads XML files representing MuJoCo models and provides
    basic functionalities to interact with them.

    Args:
        xml_path (str): Path to the XML file to load.
    """

    @abstractmethod
    def __init__(self, xml_path: str) -> None:
        """Create XMLModel from an XML file representing a MuJoCo model. The
        XMLModel provides functionalities to interact with them and compose
        multiple models.

        Args:
            xml_path (str): Path to the XML file to load.
        """
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def robot_type(self) -> str:
        pass

    @property
    @abstractmethod
    def default_initial_dof_pos(self) -> List[float]:
        """Return the initial state the robot DOFs should be reset to as defined
        in the MuJoCo keyframe entry 'default_initial'.
        """
        pass

    @abstractmethod
    def attach(self, other, attachment_body: str = None,
               attachment_pos: str = "0 0 0",
               attachment_quat: str = "1 0 0 0"
               ) -> None:
        """Attaches another DexterityXML to this one. The attachment body
        specifies where the new model should be added.

        Args:
            other (DexterityXML): Model that should be
                attached to this one.
            attachment_body (str): Which body of this XMLModel the new model
                should be attached to. If no attachment_body is specified,
                the new model will be attached to the worldbody.
            attachment_pos (Tuple[float, ...]): Positional offset of the model
                to be attached.
            attachment_quat (Tuple[float, ...]): Rotational offset of the model
                to be attached.
        """
        pass

    @contextmanager
    @abstractmethod
    def as_xml(self, file_path: str, collect_meshes: bool = True) -> None:
        """Creates a temporary XML file that can be used to load the model in
        MuJoCo or IsaacGym.
        """
        pass
