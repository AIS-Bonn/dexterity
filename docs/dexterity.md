Dexterity
=======

This document explains the environments and functionalities added to IsaacGymEnvs by dexterity.

Overview
--------
Dexterity environments make use of the same underlying structure as the factory tasks. Specifically, dexterity uses an underlying [base](../isaacgymenvs/tasks/dexterity/base/base.py) class. [environment](../isaacgymenvs/tasks/dexterity/env/) classes (e.g. hammer) are derived from the base class to create different scenes. From each environment, multiple [tasks](../isaacgymenvs/tasks/dexterity/task/) can be created that specify the objective for this setting.

The tasks included so far are: **DexterityTaskBinPick**, **DexterityTaskDrillPickAndPlace**, **DexterityTaskHammerDriveNail**, and **DexterityTaskObjectLift**.

Assets and Robot Models
------

dexterity adds a variety of assets required for the tasks, as well as robot models and a new way of composing them. All robot models are specified as XML files. Instead of building separate models for each combination of robot arm and end-effector [DexterityXML](../isaacgymenvs/tasks/dexterity/xml/xml.py) provides functionalities to attach different models. Therefore, the desired robot is specified simply through the individual files, for example as:
```
robot: [ 'franka_emika_panda/panda.xml', 'schunk_sih/right_hand.xml', 'vive_tracker/tracker.xml']
```
in the [DexterityBase.yaml](../isaacgymenvs/cfg/task/DexterityBase.yaml) config file.

### Dexterity robot models
| Robot                                            | Preview                                                                         |
|--------------------------------------------------|---------------------------------------------------------------------------------|
| [UR5e](../assets/dexterity/ur5e)                 | <img src="images/dexterity/ur5e.png" align="center" width="300"/>               |
| [Panda](../assets/dexterity/franka_emika_panda)  | <img src="images/dexterity/franka_emika_panda.png" align="center" width="300"/> |
| [KUKA Allegro](../assets/dexterity/kuka_allegro) | <img src="images/dexterity/kuka_allegro.png" align="center" width="300"/>       |
| [Schunk SIH](../assets/dexterity/schunk_sih)     | <img src="images/dexterity/schunk_sih.png" align="center" width="300"/>         |
| [Shadow Hand](../assets/dexterity/shadow_hand)   | <img src="images/dexterity/shadow_hand.png" align="center" width="300"/>        |

Teleoperation and Imitation Learning
------
dexterity provides an interface to teleoperate the Isaac Gym environments in virtual reality. Related code can be found in the [demo](../isaacgymenvs/tasks/dexterity/demo) directory. The teleoperation interface is restricted to specific hardware (Vive VR Headset and Tracker and SenseGlove) and can be used to operate any combination of the robot arm and hand models.

Contact and Citation
------
dexterity is developed by [Malte Mosbach](https://maltemosbach.github.io/).
