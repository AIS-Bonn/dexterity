import isaacgym
from isaacgymenvs.tasks.dexterity.xml.xml import DexterityXML
import mujoco
import os
import pytest


cwd = os.path.dirname(os.path.abspath(__file__))
assets_dir = os.path.join(cwd, "../../../../assets/dexterity")

MODEL_XML_PATHS = ["franka_emika_panda/panda.xml",
                   #"kuka_allegro/right_hand.xml",
                   "schunk_sih/right_hand_modified.xml",
                   "shadow_hand/right_hand.xml",
                   "ur5e/ur5e.xml",
                   "vive_tracker/tracker.xml"]

for i, xml_path in enumerate(MODEL_XML_PATHS):
    MODEL_XML_PATHS[i] = os.path.normpath(os.path.join(assets_dir, xml_path))


#@pytest.mark.skip(reason="Test when new model is added.")
def test_xml_files(show_model: bool = True):
    for xml_path in MODEL_XML_PATHS:
        model = mujoco.MjModel.from_xml_path(xml_path)

        if show_model:
            import mujoco_viewer
            data = mujoco.MjData(model)
            viewer = mujoco_viewer.MujocoViewer(model, data,)

            while True:
                if viewer.is_alive:
                    mujoco.mj_step(model, data)
                    viewer.render()
                else:
                    break

            viewer.close()

@pytest.mark.skip()
def test_xml_export(show_model: bool = True):
    for xml_path in MODEL_XML_PATHS:
        model = DexterityXML(xml_path)

        with model.as_xml("dexterity_xml_model.xml"):
            model = mujoco.MjModel.from_xml_path("dexterity_xml_model.xml")

        if show_model:
            import mujoco_viewer
            data = mujoco.MjData(model)
            viewer = mujoco_viewer.MujocoViewer(model, data)

            while True:
                if viewer.is_alive:
                    mujoco.mj_step(model, data)
                    viewer.render()
                else:
                    break

            viewer.close()

@pytest.mark.skip()
def test_model_attaching(show_model: bool = True):
    arm_model = DexterityXML(os.path.normpath(os.path.join(assets_dir, "franka_emika_panda/panda.xml")))
    hand_model = DexterityXML(os.path.normpath(os.path.join(assets_dir, "shadow_hand/right_hand.xml")))
    tracker_model = DexterityXML(os.path.normpath(os.path.join(assets_dir, "vive_tracker/tracker.xml")))
    arm_model.attach(hand_model)
    arm_model.attach(tracker_model)
    with arm_model.as_xml("dexterity_xml_model.xml"):
        model = mujoco.MjModel.from_xml_path("dexterity_xml_model.xml")

    if show_model:
        import mujoco_viewer
        data = mujoco.MjData(model)
        viewer = mujoco_viewer.MujocoViewer(model, data)

        while True:
            if viewer.is_alive:
                mujoco.mj_step(model, data)
                viewer.render()
            else:
                break

        viewer.close()


if __name__ == '__main__':
    test_xml_files(show_model=True)
