"""ur_controller controller."""
import rospy
from controller import Robot
from joint_state_publisher import JointStatePublisher
from trajectory_follower import TrajectoryFollower
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Float32MultiArray

from controller import Robot
from controller import Connector
import numpy as np
#from kinematics.inverse import InverseKinematics
#from kinematics.forward import ForwardKinematics
#from trajectory import Trajectory
#from ur_utils import Utils
#from scipy.spatial.transform import Rotation
#import socket
import time
#import struct
#import pickle
#from PIL import Image as pimg
#from P6BinPicking.vision.segmentation.detector import InstanceDetector
#np.set_printoptions(precision=4, suppress=True)

"""
def send_msg(msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    conn.setblocking(True)
    conn.sendall(msg)
    conn.setblocking(False)

def recv_msg():
    # Read message length and unpack it into an integer
    raw_msglen = recvall(4)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
    # Read the message data
    return recvall(msglen)

def recvall(n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = None
        try:
            packet = conn.recv(n - len(data))
        except socket.error:
            pass
        if not packet:
            return None
        data.extend(packet)
    return data
"""
def get_joint_angles():
    angles = np.zeros(6)
    for i in range(6):
        angles[i] = motor_sensors[i].getValue()
    return angles
"""
def respond(result, data = None):
    cmd = {}
    cmd["result"] = result
    cmd["data"] = data
    send_msg(pickle.dumps(cmd))
"""


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


finger_motors = [robot.getMotor("right_finger_motor"), robot.getMotor("left_finger_motor")]
finger_sensors = [robot.getPositionSensor("right_finger_sensor"), robot.getPositionSensor("left_finger_sensor")]
FINGER_CLOSED_POSITION = [0.005, 0.005]  # right, left
FINGER_OPEN_POSITION = [-0.035, 0.045]  # right, left
MAX_FINGER_DISTANCE = 80  # mm abs open pos
CLOSED_FINGER_DISTANCE = 10 # mm

#for sensor in motor_sensors:
#    sensor.enable(10)
for sensor in finger_sensors:
    sensor.enable(10)


"""
motors[0].setPosition(1.57)
motors[1].setPosition(-2.14)
motors[2].setPosition(-1.57)
motors[3].setPosition(-1.01)
motors[4].setPosition(1.57)
motors[5].setPosition(1.05)
motors[6].setPosition(0.35)"""
first_run = True
can_run = False
time_passed = 0

wait_time = 2000


current_task = "idle"
args = None
command_is_executing = False
print_once_flag = True
rgb_enabled = False
depth_enabled = False
gripper_timeout_timer = "paused"
#instance_detector = InstanceDetector("model_final_sim.pth")


#Init loop
# while robot.step(timestep) != -1:
#     time_passed += timestep
#     if time_passed > wait_time:
#         break

#trajectory.generate_trajectory([-0.13, 0.16, 0.7, 0.5, -1.5, 0.5], 0.1)



# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1 and not rospy.is_shutdown():
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






















"""
    if current_task == "idle":
        if print_once_flag:
            print("Waiting for command")
            robot.step(1)
            print_once_flag = False
        msg = recv_msg()
        if msg != None:
            cmd = pickle.loads(msg)
            #print(cmd)
            current_task = cmd["name"]
            args = cmd["args"]
            print("Executing: " + current_task + " " + str(args))
            print_once_flag = True
    elif current_task == "movej":
        if not command_is_executing:
            for i in range(6):
                motors[i].setVelocity(args["speed"])
                motors[i].setAcceleration(args["acc"])
                motors[i].setPosition(args["angles"][i])
            command_is_executing = True
        else:
            diff = 0
            for i in range(6):
                diff += abs(args["angles"][i] - motor_sensors[i].getValue())
            if diff < 0.1:
                command_is_executing = False
                current_task = "idle"
                respond("done")
    elif current_task == "suction_on":
        suction.lock()
        current_task = "idle"
        respond("done")
    elif current_task == "suction_off":
        suction.unlock()
        current_task = "idle"
        respond("done")
    elif current_task == "close_gripper":
        if gripper_timeout_timer == "paused":
            gripper_timeout_timer = 600
        gripper_timeout_timer -= timestep
        width = args["width"]
        speed = args["speed"]
        lock = args["lock"]
        gripping_box = args["gripping_box"]
        if width > MAX_FINGER_DISTANCE:
            print(f'WARNING: Width over max distance of {MAX_FINGER_DISTANCE}mm. Clamping value')
            width = MAX_FINGER_DISTANCE
        elif width < 0:
            print('WARNING: Width under min distance of 0mm. Clamping value')
            width = 0
        percent_closed = (MAX_FINGER_DISTANCE - width) / MAX_FINGER_DISTANCE
        right_finger_position = FINGER_OPEN_POSITION[0] - (FINGER_OPEN_POSITION[0] - FINGER_CLOSED_POSITION[0]) * percent_closed
        left_finger_position = FINGER_OPEN_POSITION[1] - (FINGER_OPEN_POSITION[1] - FINGER_CLOSED_POSITION[1]) * percent_closed
        if not command_is_executing:
            command_is_executing = True
            finger_motors[1].setVelocity(speed)
            finger_motors[1].setPosition(left_finger_position)
            finger_motors[0].setVelocity(speed)
            finger_motors[0].setPosition(right_finger_position)
        else:
            if finger_sensors[0].getValue() - 0.0001 <= right_finger_position <= finger_sensors[0].getValue() + 0.0001 or gripper_timeout_timer < 0:
                if lock:
                    gripper_connector.lock()
                if gripping_box:
                    gripper_connector_for_box.lock()
                command_is_executing = False
                current_task = "idle"
                respond("done")
                gripper_timeout_timer = "paused"
    elif current_task == "open_gripper":
        gripper_connector.unlock()
        gripper_connector_for_box.unlock()
        if gripper_timeout_timer == "paused":
            gripper_timeout_timer = 1000
        gripper_timeout_timer -= timestep
        if not command_is_executing:
            command_is_executing = True
            finger_motors[1].setVelocity(5)
            finger_motors[1].setPosition(FINGER_OPEN_POSITION[1])
            finger_motors[0].setVelocity(5)
            finger_motors[0].setPosition(FINGER_OPEN_POSITION[0])
        else:
            if finger_sensors[0].getValue() - 0.0001 <= FINGER_OPEN_POSITION[1] <= finger_sensors[0].getValue() + 0.0001 or gripper_timeout_timer < 0:
                command_is_executing = False
                current_task = "idle"
                respond("done")
    elif current_task == "movel":
        for i in range(6):
            motors[i].setVelocity(3)
            motors[i].setAcceleration(-1)
            #motors[i].setControlPID(50, 3, 5)
        if not command_is_executing:
            trajectory.generate_trajectory(args["coords"], args["speed"])
            command_is_executing = True
        else:
            angles = trajectory.calculate_step()
            for i in range(6):
                motors[i].setPosition(angles[i])
            if trajectory.is_done:
                diff = 0
                for i in range(6):
                    diff += abs(angles[i] - motor_sensors[i].getValue())
                if diff < 0.1:
                    for i in range(200):
                        robot.step(timestep)
                    command_is_executing = False
                    current_task = "idle"
                    respond("done")
    elif current_task == "getl":
        thetas = [0]*6
        for i in range(6):
            thetas[i] = motor_sensors[i].getValue()
        tmat = fkin.compute_TBT(thetas)
        trans, rot = Utils.tmat_to_trans_and_rot(tmat)
        rotvec = rot.as_rotvec()
        pose = [trans[0], trans[1], trans[2], rotvec[0], rotvec[1], rotvec[2]]
        respond("done", pose)
        current_task = "idle"
    elif current_task == "set_tcp":
        pose = args["pose"]
        trans = [pose[0], pose[1], pose[2]]
        rotvec = [pose[3], pose[4], pose[5]]
        rot = Rotation.from_rotvec(rotvec)
        tmat = Utils.trans_and_rot_to_tmat(trans, rot)
        fkin.T6T = tmat
        current_task = "idle"
        respond("done")
    elif current_task == "get_image":
        if not rgb_enabled:
            cameraRGB.enable(timestep)
            rgb_enabled = True
        else:
            np_img = np.array(cameraRGB.getImageArray(), dtype=np.uint8)
            respond("done", np_img)
            current_task = "idle"
            cameraRGB.disable()
            rgb_enabled = False
    elif current_task == "get_depth":
        if not depth_enabled:
            cameraDepth.enable(timestep)
            depth_enabled = True
        else:
            np_dep = np.array(cameraDepth.getRangeImageArray())
            respond("done", np_dep)
            current_task = "idle"
            cameraDepth.disable()
            depth_enabled = False
    elif current_task == "print":
        print(f"remote message: {args['content']}")
        respond("done")
        current_task = "idle"
    elif current_task == "inst_seg":
        if not rgb_enabled:
            cameraRGB.enable(timestep)
            rgb_enabled = True
        else:
            np_img = np.array(cameraRGB.getImageArray(), dtype=np.uint8)
            np_img = np_img.transpose((1, 0, 2))
            np_img = np_img[:, :, ::-1]  # BGR ordering
            results = instance_detector.predict(np_img)
            respond("done", results)
            current_task = "idle"
            cameraRGB.disable()
            rgb_enabled = False
    elif current_task == "finger_displacement":
        right_finger_position = finger_sensors[0].getValue()
        left_finger_position = finger_sensors[1].getValue()
        distance = np.abs(right_finger_position - left_finger_position)
        print(f"Right finger: {right_finger_position}, left finger: {left_finger_position}, distance: {distance}m")
        respond("done", distance)
        current_task = "idle"
    else:
        respond("Unknown command: " + current_task)
        raise Exception("Received unknown command: " + current_task)
"""