"""Dexterity: class for DexterityXML

Inherits DexterityABCXML class. Inherited by DexterityXMLAsset class.
"""

from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import shutil
from typing import *
import xml.etree.ElementTree as ET

from .schema_class_xml import DexterityABCXML
from . import MUJOCO_TAGS, ROBOT_TYPES


class DexterityXML(DexterityABCXML):

    def __init__(self, xml_path: str) -> None:
        self.xml_path = xml_path
        self.dir_path = os.path.dirname(xml_path)
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()

        for tag in MUJOCO_TAGS:
            setattr(self, tag, self._init_element(tag))
            
        self._resolve_assets()

    @property
    def model_name(self) -> str:
        if not hasattr(self, "_model_name"):
            self._init_model_info()
        return self._model_name

    @property
    def robot_type(self) -> str:
        if not hasattr(self, "_robot_type"):
            self._init_model_info()
        return self._robot_type

    @property
    def ros_state_topic(self) -> str:
        if not hasattr(self, "_ros_state_topic"):
            self._init_model_info()
        return self._ros_state_topic

    @property
    def ros_target_topic(self) -> str:
        if not hasattr(self, "_ros_target_topic"):
            self._init_model_info()
        return self._ros_target_topic

    @property
    def ros_joint_names(self) -> List[str]:
        if not hasattr(self, "_ros_joint_mapping"):
            self._init_model_info()
        return list(self._ros_joint_mapping.keys())

    @property
    def ros_joint_mapping(self) -> Dict[str, List[float]]:
        if not hasattr(self, "_ros_joint_mapping"):
            self._init_model_info()
        return self._ros_joint_mapping

    @property
    def home_dof_pos(self) -> List[float]:
        initial_qpos = self._findall_rec(
            node=self.keyframe, tags="key",
            attribs={"name": "home"}, return_first=True).attrib["qpos"]
        return list(map(float, initial_qpos.split(" ")))

    @property
    def default_initial_dof_pos(self) -> List[float]:
        initial_qpos = self._findall_rec(
            node=self.keyframe, tags="key",
            attribs={"name": "default_initial"}, return_first=True).attrib["qpos"]
        return list(map(float, initial_qpos.split(" ")))

    @property
    def real_robot_initial_dof_pos(self) -> List[float]:
        initial_qpos = self._findall_rec(
            node=self.keyframe, tags="key",
            attribs={"name": "real_robot_initial"}, return_first=True).attrib["qpos"]
        return list(map(float, initial_qpos.split(" ")))

    @property
    def num_actions(self) -> int:
        num_actuators = 0
        for actuator in self.actuator:
            num_actuators += 1
        return num_actuators

    @property
    def actuator_names(self) -> List[str]:
        return [actuator.attrib['name'] for actuator in self.actuator]

    @property
    def actuator_targets(self) -> List[str]:
        return [actuator.attrib['joint'] for actuator in self.actuator]

    @property
    def num_joints(self) -> int:
        joints = self._findall_rec(
            node=self.worldbody, tags="joint", return_first=False)
        if joints is None:
            return 0
        return len(joints)

    @property
    def joint_names(self) -> List[str]:
        joints = self._findall_rec(
            node=self.worldbody, tags="joint", return_first=False)
        return [j.attrib['name'] for j in joints]

    @property
    def actuated_joint_names(self) -> List[str]:
        return [actuator.attrib['joint'] for actuator in self.actuator]

    @property
    def body_names(self) -> List[str]:
        bodies = self._findall_rec(
            node=self.worldbody, tags="body", return_first=False)
        return [b.attrib['name'] for b in bodies]

    @property
    def site_names(self) -> List[str]:
        sites = self._findall_rec(
            node=self.worldbody, tags="site", return_first=False)
        return [s.attrib['name'] for s in sites]

    def bodies_as_trimesh(self) -> Dict[str, Any]:
        import trimesh
        with self.as_xml('tmp_trimesh_collision_mesh_export.xml'):
            tmp_tree = ET.parse('tmp_trimesh_collision_mesh_export.xml')
            tmp_root = tmp_tree.getroot()
            tmp_worldbody = tmp_root.find('worldbody')
            tmp_asset = tmp_root.find('asset')
            geoms = self._findall_rec(
                node=tmp_worldbody, tags="geom",
                attribs={"type": "mesh"},
                return_first=False)

            visual_geoms, collision_geoms = [], []
            for geom in geoms:
                if all(k in geom.keys() for k in ["conaffinity", "contype"]):
                    if geom.attrib["contype"] == "0" and \
                            geom.attrib["conaffinity"] == "0":
                        visual_geoms.append(geom)
                        continue

                collision_geoms.append(geom)

            body_trimeshes = {}
            for geom in visual_geoms:
                parent_body = tmp_worldbody.find(
                    f".//geom[@mesh='{geom.attrib['mesh']}']..")
                if parent_body.attrib['name'] not in body_trimeshes.keys():
                    body_trimeshes[parent_body.attrib['name']] = \
                        {"visual": [], "collision": []}
                mesh = self._findall_rec(
                    node=tmp_asset, tags="mesh",
                    attribs={"name": geom.attrib['mesh']}, return_first=True)
                body_trimesh = trimesh.load_mesh(
                        ".dexterity_xml_model_meshes/" + mesh.attrib['file'])
                if "scale" in mesh.keys():
                    body_trimesh.apply_scale(
                        [float(x) for x in mesh.attrib['scale'].split(" ")])

                if "pos" in geom.keys():
                    pos_offset = np.array(
                        [float(x) for x in geom.attrib['pos'].split(" ")])
                else:
                    pos_offset = np.array([0, 0, 0])
                body_trimeshes[parent_body.attrib['name']]['visual'].append(
                    {"trimesh": body_trimesh,
                     "pos_offset": pos_offset})

            for geom in collision_geoms:
                parent_body = tmp_worldbody.find(
                    f".//geom[@mesh='{geom.attrib['mesh']}']..")
                if parent_body.attrib['name'] not in body_trimeshes.keys():
                    body_trimeshes[parent_body.attrib['name']] = \
                        {"visual": [], "collision": []}
                mesh = self._findall_rec(
                    node=tmp_asset, tags="mesh",
                    attribs={"name": geom.attrib['mesh']}, return_first=True)
                body_trimesh = trimesh.load_mesh(
                    ".dexterity_xml_model_meshes/" + mesh.attrib['file'])
                if "scale" in mesh.keys():
                    body_trimesh.apply_scale(
                        [float(x) for x in mesh.attrib['scale'].split(" ")])

                if "pos" in geom.keys():
                    pos_offset = np.array(
                        [float(x) for x in geom.attrib['pos'].split(" ")])
                else:
                    pos_offset = np.array([0, 0, 0])
                body_trimeshes[parent_body.attrib['name']]['collision'].append(
                    {"trimesh": body_trimesh,
                     "pos_offset": pos_offset})
        return body_trimeshes

    def get_keypoints(self):
        parent_map = {c: p for p in self.worldbody.iter() for c in p}
        sites = self._findall_rec(
            node=self.worldbody, tags="site", return_first=False)

        keypoints_dict = defaultdict(dict)
        for site in sites:
            if site.attrib['name'].startswith('keypoint-'):
                _, group, name = site.attrib['name'].split('-')
                keypoints_dict[group][name] = {
                    'body_name': parent_map[site].attrib['name'],
                    'pos': np.fromstring(site.attrib['pos'], sep=' ')
                    if 'pos' in site.attrib.keys() else np.array([0., 0., 0.]),
                    # x, y, z, w convention in IsaacGym
                    'quat': np.roll(
                        np.fromstring(site.attrib['quat'], sep=' '), -1)
                    if 'quat' in site.attrib.keys() else
                    np.array([0., 0., 0., 1.])
                }
        return keypoints_dict

    def get_teleop_mapping(self):
        parent_map = {c: p for p in self.worldbody.iter() for c in p}
        sites = self._findall_rec(
            node=self.worldbody, tags="site", return_first=False)

        teleop_mapping = []
        for site in sites:
            if site.attrib['name'].startswith('teleop-'):
                _, sg_sensor = site.attrib['name'].split('-')
                teleop_mapping.append({
                    'sg_sensor': sg_sensor,
                    'target_joint': parent_map[site].find('joint').attrib['name'],
                    'polycoef': np.fromstring(site.attrib['user'], sep=' ')
                })
        return teleop_mapping

    def get_parent_names(self, body_name: str) -> str:
        """Find the name of the last joint before a given body."""
        body = self.worldbody.find(
                f".//body[@name='{body_name}']")
        assert body is not None, \
            f"body with name '{body_name}' not in worldbody/"
        parent_map = {c: p for p in self.worldbody.iter() for c in p}
        while self.worldbody.find(f".//body[@name='{body_name}']..joint") is None:
            body = parent_map[body]
            body_name = body.attrib['name']
        parent_joint = self.worldbody.find(f".//body[@name='{body_name}']..joint")
        return parent_joint.attrib['name'], body.attrib['name']

    def attach(self, other, attachment_body: str = None,
               attachment_pos: str = "0 0 0",
               attachment_quat: str = "1 0 0 0",
               order: str = "append",
               ) -> None:
        assert isinstance(other, DexterityXML), \
            f"{type(other)} is not a DexterityXML instance."

        # Attach to specified body
        if attachment_body is not None:
            attachment_body = self._findall_rec(
                node=self.worldbody, tags="body",
                attribs={"name": attachment_body}, return_first=True)

        # Find attachment_site to attach to. If none is found, attach to
        # worldbody.
        else:
            attachment_site = self.root.find(".//site[@name='attachment_site']")
            # Body with attachment_site found
            if attachment_site is not None:
                attachment_pos = attachment_site.attrib["pos"]
                attachment_quat = attachment_site.attrib["quat"]
                attachment_body = self.root.find(
                    ".//site[@name='attachment_site']..")

                # Mark attachment_site as used
                attachment_site.set('name', other.model_name + '_attached_here')

            # Attach to worldbody
            else:
                attachment_body = self.worldbody

        for tag in MUJOCO_TAGS:
            root = attachment_body if tag == "worldbody" else getattr(self, tag)
            for i, node in enumerate(getattr(other, tag)):

                if (self._findall_rec(
                        node=root, tags=node.tag,
                        attribs={"name": node.get("name")}, return_first=True)
                        is None or node.get("name") is None):
                    # Overwrite position and rotation of attached base link
                    if tag == "worldbody" and i == 0:
                        base_pos = node.attrib[
                            "pos"] if "pos" in node.attrib.keys() else "0 0 0"
                        base_quat = node.attrib[
                            "quat"] if "quat" in node.attrib.keys() else "1 0 0 0"

                        t0 = np.array(list(map(float, base_pos.split())))
                        r0_w, r0_x, r0_y, r0_z = list(
                            map(float, base_quat.split()))
                        r0 = R.from_quat([r0_x, r0_y, r0_z, r0_w])

                        t1 = np.array(list(map(float, attachment_pos.split())))
                        r1_w, r1_x, r1_y, r1_z = list(
                            map(float, attachment_quat.split()))
                        r1 = R.from_quat([r1_x, r1_y, r1_z, r1_w])

                        r_x, r_y, r_z, r_w = (r1 * r0).as_quat()
                        t = r1.as_matrix().dot(t0) + t1

                        node.set("pos", str(t)[1:-1])
                        node.set("quat", f"{r_w} {r_x} {r_y} {r_z}")
                    if order == "append":
                        root.append(node)
                    elif order == "insert":
                        root.insert(0, node)
                    else:
                        assert False, f"Unknown order argument '{order}' given."
                else:
                    # Merge default robot positions and controls
                    if tag == "keyframe":
                        base_node = self._findall_rec(
                            node=root, tags=node.tag,
                            attribs={"name": node.get("name")},
                            return_first=True)
                        for k, v in node.attrib.items():
                            if k != "name":
                                base_node.set(
                                    k, " ".join([base_node.attrib[k], v]))
                    else:
                        print(f"Skipping duplicate {node.tag} with name "
                              f"{node.get('name')}.")

    @contextmanager
    def as_xml(self, file_path: str, collect_meshes: bool = True) -> None:
        self._save_xml(file_path, collect_meshes)
        try:
            yield None
        finally:
            self._remove_xml(file_path)

    def _resolve_assets(self) -> None:
        self._resolve_relative_paths()
        self._make_material_names_unique()
        self._make_mesh_names_unique()
        self._resolve_defaults()
        
    def _resolve_relative_paths(self) -> None:
        """Convert relative paths in assets to absolute filepaths.
        """

        if "meshdir" in self.compiler.attrib.keys():
            mesh_dir = self.compiler.attrib["meshdir"]
        else:
            mesh_dir = "."

        for asset_node in self.asset.findall("./*[@file]"):
            file_path = asset_node.get("file")
            file_name = os.path.basename(file_path)

            abs_path = os.path.abspath(self.dir_path)
            if asset_node.tag == "mesh":
                abs_path = os.path.join(abs_path, mesh_dir)
            abs_path = os.path.join(abs_path, file_name)
            asset_node.set("file", abs_path)

    def _make_material_names_unique(self) -> None:
        """Add robot info the names of materials to avoid duplicates when
        other models are attached.
        """

        for asset_node in self.asset.findall("./*[@name]"):
            if asset_node.tag == "material":
                local_name = asset_node.attrib["name"]
                asset_node.set("name", self.model_name + "_" + local_name)

        for geom in self._findall_rec(self.root, "geom", return_first=False):
            if "material" in geom.attrib.keys():
                geom.set("material",
                         "_".join([self.model_name, geom.attrib["material"]]))

    def _make_mesh_names_unique(self) -> None:
        """Add robot info the names of meshes to avoid duplicates when
        other models are attached.
        """

        # Add model name to meshes
        for asset_node in self.asset.findall("./*[@file]"):
            if asset_node.tag == "mesh":
                file_path = asset_node.get("file")
                file_name = os.path.basename(file_path).split(".")[0]
                local_name = asset_node.attrib["name"] \
                    if "name" in asset_node.attrib.keys() else file_name
                asset_node.set("name", "_".join([self.model_name, local_name]))

        # Adjust mesh references in geom tags
        for geom in self._findall_rec(self.root, "geom", return_first=False):
            if "mesh" in geom.attrib.keys():
                geom.set("mesh",
                         "_".join([self.model_name, geom.attrib["mesh"]]))

    def _resolve_defaults(self) -> None:
        """Resolve default attributes on initialization, so they do not
        interfere when models are attached or exported."""

        # Resolve inheritance in default class tree
        self._resolve_default_inheritance(self.default)

        # Resolve inheritance via 'childclass' attribute
        self._resolve_childclass_inheritance(self.worldbody)

        # Update model element attributes with default classes
        self._update_by_default(self.default)

        # Remove remaining 'class' and 'childclass' elements from worldbody
        for worldbody_element in self.worldbody.iter():
            if 'class' in worldbody_element.attrib.keys():
                worldbody_element.attrib.pop('class')
            if 'childclass' in worldbody_element.attrib.keys():
                worldbody_element.attrib.pop('childclass')

        # Set default to empty element as everything should be resolved now
        self.default = ET.Element('default')

        # Fix joint elements so IsaacGym reads them correctly
        if self.num_joints > 0:
            for joint in self._findall_rec(self.worldbody, 'joint', return_first=False):
                if 'range' in joint.attrib.keys() and \
                        'limited' not in joint.attrib.keys():
                    joint.set('limited', 'true')

        # Update root elements
        for tag in MUJOCO_TAGS:
            for child in self.root:
                if child.tag == tag:
                    self.root.remove(child)
            self.root.append(getattr(self, tag))

    def _resolve_default_inheritance(self, node: ET.Element,
                                     parent: ET.Element = None) -> None:
        if parent is not None:
            # Get elements of the parent node that are not again default
            for default_element in parent:
                if default_element.tag != "default":
                    # Merge attribs if element with this tag is in node already
                    if default_element.tag in [child.tag for child in node]:
                        node_element = node.find(default_element.tag)
                        for k, v in default_element.attrib.items():
                            if k not in node_element.attrib.keys():
                                node_element.set(k, v)
                    # Append parent element to node
                    else:
                        node.append(default_element)

        # Continue recursively
        for child_node in node:
            if child_node.tag == "default":
                self._resolve_default_inheritance(child_node, parent=node)

    def _update_by_default(self, default_class: ET.Element) -> None:
        default_class_name = default_class.attrib['class'] if 'class' in default_class.attrib.keys() else None
        for default_element in default_class:
            if default_element.tag != "default":
                matches = []
                for tag in MUJOCO_TAGS:
                    if tag != "default":
                        tmp_matches = self._findall_rec(getattr(self, tag), default_element.tag, return_first=False)
                        if tmp_matches is not None:
                            matches += tmp_matches

                for matching_element in matches:
                    matching_element_class_name = matching_element.attrib['class'] if 'class' in matching_element.attrib.keys() else None
                    # If the class of the found element and default class match
                    if matching_element_class_name == default_class_name:
                        # Set missing attribs by default values
                        for k, v in default_element.attrib.items():
                            if k not in matching_element.attrib.keys():
                                matching_element.set(k, v)
                        # Remove class attrib as it is no longer needed
                        if matching_element_class_name is not None:
                            matching_element.attrib.pop("class")

        for default_element in default_class:
            if default_element.tag == "default":
                self._update_by_default(default_element)

    def _resolve_childclass_inheritance(self, node: ET.Element,
                                        parent_childclass: str = None) -> None:
        node_childclass = node.attrib[
            'childclass'] if 'childclass' in node.attrib.keys() else None

        if parent_childclass is not None:
            # Set parent_childclass as node_childclass if node does not have
            # its own childclass already
            if node_childclass is None:
                node.set('childclass', parent_childclass)

            # Set node class as parent childclass if node does not have its own
            # class already
            if 'class' not in node.attrib.keys():
                node.set('class', parent_childclass)

        # Update node childclass
        node_childclass = node.attrib[
            'childclass'] if 'childclass' in node.attrib.keys() else None

        # Continue recursively
        for child_node in node:
            # Childclass only applies to (world)body elements
            self._resolve_childclass_inheritance(
                child_node, parent_childclass=node_childclass)

    def _init_element(self, name: str, root_name: str = "root") -> ET.Element:
        """Returns the first sub-element matching 'name'. If there is none, a
        default <'name'/> tag is created.
        Args:
            name (str): Name of sub-element to be found or initialized.
            root_name (str): Name of the element to search
        Returns:
            ET.Element: Node that was found or created.
        """
        root = getattr(self, root_name)

        found = root.find(name)
        if found is None:
            found = ET.Element(name)
            root.append(found)
        return found

    def _findall_rec(self, node: ET.Element, tags: Union[str, List[str]],
                     attribs: Dict[str, Any] = {}, return_first: bool = True
                     ) -> Union[None, ET.Element, List[ET.Element]]:

        """ Return elements of the given tag(s) that match the attributes found
        by recursive search from root element.

        Args:
            node (ET.Element): Root node where the search starts.
            tags (Union[str, List[str]]): Tags being searched.
            attribs (Dict[str, Any): Attributes that should be matched.
            return_first (bool): Whether the first match is returned
                immediately.
        Returns:
            Union[None, ET.Element, List[ET.Element]]: Elements with required
                tags that match the attributes. None is returned if no match is
                found.
        """

        ret = []
        tags = [tags] if not isinstance(tags, List) else tags

        # Check whether node matches the requirements
        if node.tag in tags:
            match = True
            for k, v in attribs.items():
                if node.get(k) != v:
                    match = False
                    break

            if match:
                if return_first:
                    return node
                else:
                    ret.append(node)

        # Continue recursive search
        for child_node in node:
            if return_first:
                ret = self._findall_rec(child_node, tags, attribs, return_first)
                if ret is not None:
                    return ret
            else:
                fnd = self._findall_rec(child_node, tags, attribs, return_first)
                if fnd:
                    ret += [fnd] if not isinstance(fnd, List) else fnd

        return ret if ret else None

    def _init_model_info(self) -> None:
        # Set model name and robot type.
        model_info = self.root.attrib["model"].split("-")
        assert len(model_info) == 2, \
            f"Model attribute {model_info} of model {self.xml_path} does not " \
            f"follow naming convention model='<model_name>-<robot_type>'."
        model_name, robot_type = model_info
        assert robot_type in ROBOT_TYPES, \
            f"Robot type {robot_type} of model {self.xml_path} is not in " \
            f"allowed types {ROBOT_TYPES}."
        self._model_name = model_name
        self._robot_type = robot_type

        # Set ROS topics if they are specified.
        self._ros_state_topic = None
        self._ros_target_topic = None
        self._ros_joint_mapping = {}
        for child in self.custom:
            if child.tag == 'text' and child.attrib['name'] == 'ros_state_topic':
                self._ros_state_topic = child.attrib['data']
            elif child.tag == 'text' and child.attrib['name'] == 'ros_target_topic':
                self._ros_target_topic = child.attrib['data']
            elif child.tag == 'numeric' and child.attrib['name'].startswith('ros_joint'):
                _, joint_name = child.attrib['name'].split(" ")
                self._ros_joint_mapping[joint_name] = [float(num) for num in child.attrib['data'].split(" ")]

    def _save_xml(self, file_path: str, collect_meshes: bool = True) -> None:
        file_path = os.path.normpath(file_path)

        # Create copy of model for exporting it
        self.root_exp = deepcopy(self.root)
        for tag in MUJOCO_TAGS:
            setattr(self, tag + "_exp", self._init_element(tag, root_name="root_exp"))

        if collect_meshes:
            file_path_dir = os.path.dirname(os.path.abspath(file_path))
            meshdir = os.path.join(file_path_dir, ".dexterity_xml_model_meshes")
            self._collect_meshes(meshdir)

        with open(file_path, "w") as f:
            model_str = ET.tostring(self.root_exp, encoding="unicode")
            f.write(model_str)

    def _remove_xml(self, file_path: str) -> None:
        file_path = os.path.normpath(file_path)
        file_path_dir = os.path.dirname(os.path.abspath(file_path))
        meshdir = os.path.join(file_path_dir, ".dexterity_xml_model_meshes")
        os.remove(file_path)
        shutil.rmtree(meshdir)

    def _collect_meshes(self, meshdir: str) -> None:
        """Store meshes in temporary subdirectory, since all meshes must be
        defined in relative paths for IsaacGym to find them.
        """

        os.makedirs(meshdir, exist_ok=True)
        self.compiler_exp.set("meshdir", os.path.basename(meshdir))

        for asset_node in self.asset_exp.findall("./*[@file]"):
            file_path = asset_node.get("file")
            file_name, file_extension = os.path.basename(file_path).split(".")

            # Create unique file names for the meshes
            i = 0
            while os.path.exists(os.path.join(
                    os.path.abspath(meshdir),
                    file_name + "_" + str(i).zfill(4) + "." + file_extension)):
                i += 1
            local_file_name = file_name + "_" + str(i).zfill(4) + "." + \
                              file_extension

            # Copy file to local subdirectory
            shutil.copyfile(
                file_path,
                os.path.join(os.path.abspath(meshdir), local_file_name))

            # Set file path to local copy
            asset_node.set("file", local_file_name)

    def add_sites(self, site_pos: np.array, site_group_name: str,
                  rgba: str = '1 1 0 0.8') -> None:
        for i in range(site_pos.shape[0]):
            site = ET.SubElement(self.worldbody, 'site')
            site.set('name', site_group_name + str(i))
            site.set('rgba', rgba)
            with np.printoptions(precision=8):
                site.set('pos', str(site_pos[i])[1:-1])

    def add_mocap(self, pos, quat, name: str = '__mocap__', target_body: str = 'tracker') -> None:
        tracker_body = self._findall_rec(
            node=self.worldbody, tags="body",
            attribs={"name": "tracker"}, return_first=True)

        mocap_body = ET.SubElement(self.worldbody, 'body')
        mocap_body.set('name', name)
        mocap_body.set('mocap', 'true')
        with np.printoptions(precision=8):
            mocap_body.set('pos', str(pos)[1:-1])
            mocap_body.set('quat', str(quat)[1:-1])

        mocap_visual = ET.SubElement(mocap_body, 'geom')
        mocap_visual.set('type', 'box')
        mocap_visual.set('size', '0.02 0.02 0.04')
        mocap_visual.set('group', '2')
        mocap_visual.set('rgba', '1 0.7 0 0.8')
        mocap_visual.set('contype', '0')
        mocap_visual.set('conaffinity', '0')

        weld_equality = ET.SubElement(self.equality, 'weld')
        weld_equality.set('body1', name)
        weld_equality.set('body2', target_body)
        weld_equality.set('solimp', '0.9 0.95 0.001')
        weld_equality.set('solref', '0.02 1')

    def add_freejoint(self) -> None:
        bodies = [body for body in self.worldbody.iter('body')]
        root_body = bodies[0]
        freejoint = ET.SubElement(root_body, 'freejoint')

        home_qpos = self._findall_rec(
            node=self.keyframe, tags="key",
            attribs={"name": "home"}, return_first=True)
        home_qpos.set('qpos', "0 0 0 1 0 0 0 " + home_qpos.attrib["qpos"])

        initial_qpos = self._findall_rec(
            node=self.keyframe, tags="key",
            attribs={"name": "initial"}, return_first=True)
        initial_qpos.set('qpos', "0 0 0 1 0 0 0 " + initial_qpos.attrib["qpos"])
