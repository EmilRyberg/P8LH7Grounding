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

5. Make sure that the catkin workspace that **this** project/package is inside is built and sourced.

6. (Re)build the main workspace with the UR-ROS ws sourced.

### Test that you got it right / it is working
1. Start roscore
2. Start webots and open the world (it is under the *webots/worlds* folder)
3. In all terminals from now on, make sure to source ROS, your universal_robots ROS Industrial workspace and source the workspace that this project is in
4. Run `roslaunch ur_e_webots ur5e_joint_limited.launch` - this acts as the WeBots controller and publishes transforms and so on
5. Run `roslaunch ur5_e_moveit_config ur5_e_moveit_planning_execution.launch`
6. Run `roslaunch ur5_e_moveit_config moveit_rviz.launch config:=true`
7. The last command should open up RViz with MoveIt! The robot in WeBots and RViz should have the same pose, if everything worked as expected. You can now play around with planning trajectories in RViz, which should move the "real" WeBots simulation.

Note that the controller in WeBots have to be set to `<extern>`. There is also a `ros` controller which enables interfacing directly with WeBots objects using topics / services. 

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

## Speech to text
To get speech to text working, several libraries need to be installed - some compiled from source.

For Azure Cognitive Services (what we are using now):
1. `pip install azure-cognitiveservices-speech`

Below is for using SpeechRecognition python library:

To setup using Sphinx:
1. Download PCRE from here: https://ftp.pcre.org/pub/pcre/ - select version pcre-8.44 (**make sure you dont take pcre2-xxx!!!**)
   1. Run `./configure --prefix=/usr` in the pcre folder
   2. Run `make` then `sudo make install`
2. Download SWIG from here: http://www.swig.org/download.html
   1. Run `./configure`, then `make` and finally `(sudo) make install`
   2. Test that it works by running `swig` in a new terminal. If installed correctly it should say "*Must specify an input file. Use -help for available options.*". If it complains about libpcre.so.1, you did something wrong in the previous step.
3. Run `sudo apt install libpulse-dev portaudio19-dev` - these dependencies are need for PyAudio and PocketSphinx.
4. Run `pip install PyAudio`
5. Run `pip install --upgrade pocketsphinx`
6. Run `pip install SpeechRecognition`

To setup for Google Speech API:
1. Run `sudo apt install libpulse-dev portaudio19-dev` - these dependencies are need for PyAudio and PocketSphinx.
2. Run `pip install PyAudio`
3. Run `pip install SpeechRecognition`
4. Run `pip install oauth2client`
5. Run `pip install google-api-python-client`

Note: You will probably get a lot of warnings when running the code, in the form of *ALSA lib pcm.c:2495:(snd_pcm_open_noupdate) Unknown PCM cards.pcm.rear*. This is a common issue, but are just warnings, and can be disabled by uncommenting lines in some config files.
