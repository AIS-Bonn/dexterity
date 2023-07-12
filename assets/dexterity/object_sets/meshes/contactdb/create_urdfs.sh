#!/bin/bash
for file in ./*
  do
      file=${file##*/}
      name=${file%.*}

      echo '<?xml version="1.0"?>' > "./visual/${name}.urdf"
      echo "<robot name=\"${name}\">" >> "./visual/${name}.urdf"
      echo '  <link name="base_link">' >> "./visual/${name}.urdf"
      echo '    <inertial>' >> "./visual/${name}.urdf"
      echo '      <mass value="0.1" />' >> "./visual/${name}.urdf"
      echo '      <origin xyz="0 0 0" />' >> "./visual/${name}.urdf"
      echo '      <inertia  ixx="0.0002" ixy="0.0"  ixz="0.0"  iyy="0.0002"  iyz="0.0" izz="0.0002" />' >> "./visual/${name}.urdf"
      echo '    </inertial>' >> "./visual/${name}.urdf"
      echo '    <visual>' >> "./visual/${name}.urdf"
      echo '      <geometry>' >> "./visual/${name}.urdf"
      mesh="<mesh filename=\"../../meshes/contactdb/${name}.stl\" scale=\"0.001 0.001 0.001\" />"
      echo "        $mesh" >> "./visual/${name}.urdf"
      echo '      </geometry>' >> "./visual/${name}.urdf"
      echo '    </visual>' >> "./visual/${name}.urdf"
      echo '    <collision>' >> "./visual/${name}.urdf"
      echo '      <geometry>' >> "./visual/${name}.urdf"
      echo "        $mesh" >> "./visual/${name}.urdf"
      echo '      </geometry>' >> "./visual/${name}.urdf"
      echo '    </collision>' >> "./visual/${name}.urdf"
      echo '  </link>' >> "./visual/${name}.urdf"
      echo '</robot>' >> "./visual/${name}.urdf"
  done

