# P8LH7Grounding








__For simulation__
1. Install webots (https://cyberbotics.com/doc/guide/installation-procedure)

2. Install UR-ROS drivers (https://github.com/ros-industrial/universal_robot)
	-Remember to install for the correct ROS distro


__Usage__

Once you have started a simulation with a UR robot and set its controller to `<extern>`, you can use the following launch file to setup all the required ROS parameters and start the simulated UR robot to ROS interface:

```
roslaunch ur_e_webots ur5e_joint_limited.launch
```

You can then control the robot with MoveIt!, use the following launch file (from the `universal_robot` ROS package) to start MoveIt!:

```
roslaunch ur5_e_moveit_config ur5_e_moveit_planning_execution.launch
```

For starting up RViz with a configuration including the MoveIt! Motion Planning plugin, run the following launch file (from the `universal_robot` ROS package):

```
roslaunch ur5_e_moveit_config moveit_rviz.launch config:=true
```

