cmake_minimum_required(VERSION 2.8.3)

project(ur_e_webots)

find_package(catkin REQUIRED genmsg actionlib_msgs message_generation)

catkin_python_setup()

add_action_files(
  DIRECTORY action
  FILES
  Gripper.action
)

generate_messages(
  DEPENDENCIES
  actionlib_msgs
)

catkin_package(CATKIN_DEPENDS message_runtime)

if (CATKIN_ENABLE_TESTING)
  find_package(roslaunch REQUIRED)
  roslaunch_add_file_check(tests/roslaunch_test.xml)
endif()

install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

catkin_install_python(PROGRAMS scripts/universal_robots_ros.py
        DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
