# P8LH7Grounding

## Compiling the Catkin Workspace
Python 3 is used for the Python modules. To ensure that Catkin uses Python 3 for compiling the packages, use 
`catkin_make -DPYTHON_EXECUTABLE=[path to your python 3 executable]` in the root of the Catkin workspace.

## Running a node
If you have problems running a Python 3 node, make sure to source in the following order: ROS Melodic -> (Anaconda if you use that for python 3 - not sure this is required) -> workspace setup.bash.
If you get errors about the node not being an executable or something like that, add executable permission (`chmod +x [path_to_node]`). If you get Python errors when trying to run the node, make sure `rospkg` is installed for Python 3 using `pip install rospkg`. Both Melodic Python 2.7 and your Python 3 has to be in PYTHONPATH environment variable, since rospy and other modules are located without Melodic's Python 2.7.

### cv_bridge problem
If there is a problem with cv_bridge, it needs to be compiled for python 3. See [here](https://medium.com/@beta_b0t/how-to-setup-ros-with-python-3-44a69ca36674).

__For simulation__
1. Install webots (https://cyberbotics.com/doc/guide/installation-procedure)

2. Install UR-ROS drivers (https://github.com/ros-industrial/universal_robot)
	-Remember to install for the correct ROS distro
	-I installed it into the project catkin workspace 

__Run the simulation__
1. Start roscore
2. Start webots and open the world


__Get image from webots__
1. Subscribe to /camera/image and/or /range_finder/image
2. Publish std\_msgs/Bool = 1 to /publish_images


__Control gripper/suction__
Topics: /gripper/set_state
		/suction/set_state

Type: std_msgs/Bool

Value: 1 = close gripper/start suction
	   0 = open gripper/stop suction

