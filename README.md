# P8LH7Grounding

## Compiling the Catkin Workspace
There is a name conflict between the em and empy packages on python3, so run `pip3 uninstall em` and then `pip3 install empy`
Python 3 is used for the Python modules. To ensure that Catkin uses Python 3 for compiling the packages, use 
`catkin_make -DPYTHON_EXECUTABLE=[path to your python 3 executable]` in the root of the Catkin workspace.

## Running a node
If you have problems running a Python 3 node, make sure to source in the following order: ROS Melodic -> (Anaconda if you use that for python 3 - not sure this is required) -> workspace setup.bash.
If you get errors about the node not being an executable or something like that, add executable permission (`chmod +x [path_to_node]`). If you get Python errors when trying to run the node, make sure `rospkg` is installed for Python 3 using `pip install rospkg`. Both Melodic Python 2.7 and your Python 3 has to be in PYTHONPATH environment variable, since rospy and other modules are located without Melodic's Python 2.7.

### cv_bridge problem
If there is a problem with cv_bridge, it needs to be compiled for python 3. See [here](https://medium.com/@beta_b0t/how-to-setup-ros-with-python-3-44a69ca36674).

---

## For simulation
This project contains a *webots* and *ur_e_webots* folder. The *webots* folder contains the files related to WeBots, including the world file. The *ur_e_wobots* is a catkin package obtained from [here](https://github.com/cyberbotics/webots/tree/master/projects/robots/universal_robots/resources/ros_package/ur_e_webots), which acts as a bridge between WeBots and ROS, such as publishing transforms and handling MoveIt commands.

To get up and running:

1. Install webots (https://cyberbotics.com/doc/guide/installation-procedure)

2. Install webots_ros (I think this is actually optional, just includes example files): `sudo apt install ros-melodic-webots-ros`

3. Install ROS MoveIt - just run `sudo apt install ros-melodic-moveit` or look [here](http://docs.ros.org/en/melodic/api/moveit_tutorials/html/doc/getting_started/getting_started.html) if you are having problems.

4. Install UR-ROS drivers (https://github.com/ros-industrial/universal_robot/tree/melodic-devel)
   - You have to build from source. Make a new catkin workspace and build it there.
   - Remember to install for the correct ROS distro

5. Compile cv_bridge to work with Python 3 (https://github.com/ros-perception/vision_opencv/tree/melodic)

6. Make sure that the catkin workspace that **this** project/package is inside is built and sourced.

7. (Re)build the main workspace with the UR-ROS and cv_bridge ws sourced.

### Test that you got it right / it is working
1. Start roscore
2. Start webots and open the world (it is under the *webots/worlds* folder)
3. In all terminals from now on, make sure to source ROS, your universal_robots ROS Industrial workspace and source the workspace that this project is in
4. Run `roslaunch ur_e_webots ur5e_joint_limited.launch` - this acts as the WeBots controller and publishes transforms and so on
5. Run `roslaunch ur5_e_moveit_config ur5_e_moveit_planning_execution.launch`
6. Run `roslaunch ur5_e_moveit_config moveit_rviz.launch config:=true`
7. The last command should open up RViz with MoveIt! The robot in WeBots and RViz should have the same pose, if everything worked as expected. You can now play around with planning trajectories in RViz, which should move the "real" WeBots simulation.

Note that the controller in WeBots have to be set to `<extern>`. There is also a `ros` controller which enables interfacing directly with WeBots objects using topics / services. 

## Speech to text
To get speech to text working, several libraries need to be installed - some compiled from source.

For Azure Cognitive Services (what we are using now):
1. `pip install azure-cognitiveservices-speech`

## Running the code
Make sure that ROS, cv_bridge, UR-ROS and this workspace is sourced when running the code. It is very important that cv_bridge is BEFORE ROS in PYTHONPATH environment variable (ROS should preferably be the last).
1. `roslaunch dialog_flow hri_backbone_launch.launch azure_key:=[api key to Azure]`
2. `rosrun bin_picking moveit_interface_node.py` - IMPORTANT: Needs to be python 2!
3. `Run dialog_flow.py` NOTE: The import paths for the dialog_flow.py does not allow it to be run from terminal, without the correct PYTHONPATH set. 
To be able to run it without updating the import paths, source the ROS, cv_bridge and this workspace and start PyCharm and set all packages as "Sources root". Then run dialog_flow.py from PyCharm.

If you want to visualize the robot in RViz: `roslaunch ur5_moveit_config moveit_rviz.launch config:=true`

### Examples of controlling WeBots directly using ROS
These examples needs to have the controller in WeBots set to the `ros` controller.

__Get image from webots__
1. Subscribe to /camera/image and/or /range_finder/image
2. Publish std\_msgs/Bool = 1 to /publish_images


__Control gripper/suction__
Topics: /gripper/set_state
		/suction/set_state

Type: std_msgs/Bool

Value: 1 = close gripper/start suction
	   0 = open gripper/stop suction
