# P8LH7Grounding

__For simulation__
1. Install webots (https://cyberbotics.com/doc/guide/installation-procedure)

2. Install UR-ROS drivers (https://github.com/ros-industrial/universal_robot)
	-Remember to install for the correct ROS distro

__Run the simulation__
1. Start roscore
2. Start webots and open the world

__Get image from webots__
1. Subscribe to /camera/image and/or /range_finder/image
2. Publish std\_msgs/Bool = 1 to /publish_images
