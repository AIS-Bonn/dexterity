"""Installation script for the 'isaacgymenvs' python package."""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from setuptools import setup, find_packages

import os


root_dir = os.path.dirname(os.path.realpath(__file__))


# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # RL
    "gym==0.24.1",
    "torch",
    "omegaconf",
    "termcolor",
    "hydra-core>=1.1",
    "rl-games==1.5.2",
    "pyvirtualdisplay",
    ]


# Build extension to create Cython interface to OpenVR and SGCore and SGConnect required for teleoperation

# Find libraries
openvr_lib_dir = os.path.expanduser("~/.steam/steam/steamapps/common/SteamVR/bin/linux64")
sgcore_lib_dir = os.path.abspath("./isaacgymenvs/tasks/dexterity/demo/contrib/SenseGlove-API/Core/SGCoreCpp/lib/linux/Release")
sgconnect_lib_dir = os.path.abspath("./isaacgymenvs/tasks/dexterity/demo/contrib/SenseGlove-API/Core/SGConnect/lib/linux/Release")

# If OpenVR is found: install with teleoperation functionality
if os.path.isdir(openvr_lib_dir):
    from Cython.Build import cythonize
    from distutils.core import Extension
    EXTENSIONS = cythonize(
        [Extension(
            "dexterityvr",
            ["./isaacgymenvs/tasks/dexterity/demo/src/dexterityvr.pyx"],
            libraries=["openvr_api", "SGCoreCpp", "SGConnect", "GL", "GLEW", "glut"],
            library_dirs=[openvr_lib_dir, sgcore_lib_dir, sgconnect_lib_dir],
            runtime_library_dirs=[openvr_lib_dir, sgcore_lib_dir, sgconnect_lib_dir],
        )])
else:
    EXTENSIONS = []
    warnings.warn("Path to OpenVR not found. Installing dexterity without teleoperation functionality.")


# Installation operation
setup(
    name="isaacgymenvs",
    author="NVIDIA",
    version="1.3.2",
    description="Benchmark environments for high-speed robot learning in NVIDIA IsaacGym.",
    keywords=["robotics", "rl"],
    include_package_data=True,
    python_requires=">=3.6.*",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages("."),
    classifiers=["Natural Language :: English", "Programming Language :: Python :: 3.7, 3.8"],
    zip_safe=False,
    ext_modules=EXTENSIONS,
)

# EOF
