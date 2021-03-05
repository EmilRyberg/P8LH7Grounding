"""ur_controller controller."""
import rospy
from controller import Robot
from joint_state_publisher import JointStatePublisher
from trajectory_follower import TrajectoryFollower
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Float32MultiArray, Bool
from controller import Connector
import numpy as np
import time

def get_joint_angles():
    angles = np.zeros(6)
    for i in range(6):
        angles[i] = motor_sensors[i].getValue()
    return angles

def gripper_cb(msg):
    if msg.data == True:
        finger_motors[1].setVelocity(5)
        finger_motors[1].setPosition(FINGER_CLOSED_POSITION[1])
        finger_motors[0].setVelocity(5)
        finger_motors[0].setPosition(FINGER_CLOSED_POSITION[0])
        """ TODO
            if lock: #should be depending on the task (i.e. gripping cover part), probably make a custom msg
                gripper_connector.lock()
            if gripping_box: #task box shaking
                gripper_connector_for_box.lock()
        """
    elif msg.data == False:
        gripper_connector.unlock()
        gripper_connector_for_box.unlock()
        finger_motors[1].setVelocity(5)
        finger_motors[1].setPosition(FINGER_OPEN_POSITION[1])
        finger_motors[0].setVelocity(5)
        finger_motors[0].setPosition(FINGER_OPEN_POSITION[0])

def suction_cb(msg):
    if msg.data == True:
        suction.lock()
        print("succer on")
    elif msg.data == False:
        suction.unlock
        print("succer off")

rospy.init_node('ur_sim_driver', disable_signals=True)
jointPrefix = rospy.get_param('prefix', '')
if jointPrefix:
    print('Setting prefix to %s' % jointPrefix)


robot = Robot()
suction = Connector("suction")
gripper_connector = Connector("gripper_connector")
gripper_connector_for_box = Connector("gripper_connector_for_box")


jointStatePublisher = JointStatePublisher(robot, jointPrefix)
trajectoryFollower = TrajectoryFollower(robot, jointStatePublisher, jointPrefix)
trajectoryFollower.start()

# we want to use simulation time for ROS
clockPublisher = rospy.Publisher('clock', Clock, queue_size=1)
if not rospy.get_param('use_sim_time', False):
    rospy.logwarn('use_sim_time is not set!')


timestep = int(robot.getBasicTimeStep())


finger_motors = [robot.getDevice("right_finger_motor"), robot.getDevice("left_finger_motor")]
finger_sensors = [robot.getDevice("right_finger_sensor"), robot.getDevice("left_finger_sensor")]
FINGER_CLOSED_POSITION = [0.005, 0.005]  # right, left
FINGER_OPEN_POSITION = [-0.035, 0.045]  # right, left
MAX_FINGER_DISTANCE = 80  # mm abs open pos
CLOSED_FINGER_DISTANCE = 10 # mm

#for sensor in motor_sensors:
#    sensor.enable(10)
for sensor in finger_sensors:
    sensor.enable(10)

first_run = True
can_run = False
time_passed = 0
flag = True
wait_time = 2000

# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1 and not rospy.is_shutdown():
    if flag == True:
        rospy.Subscriber("/gripper/set_state/", Bool, gripper_cb)
        rospy.Subscriber("/suction/set_state", Bool, suction_cb)
        flag = False
    jointStatePublisher.publish()
    trajectoryFollower.update()

    # pulish simulation clock
    msg = Clock()
    time = robot.getTime()
    msg.clock.secs = int(time)
    # round prevents precision issues that can cause problems with ROS timers
    msg.clock.nsecs = round(1000 * (time - msg.clock.secs)) * 1.0e+6
    clockPublisher.publish(msg)

#conn.close()
print("Robot controller ended")