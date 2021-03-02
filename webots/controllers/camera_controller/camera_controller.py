"""camera controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor

import rospy
from controller import Camera
from controller import RangeFinder
from controller import Robot
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rosgraph_msgs.msg import Clock
import cv2
import numpy as np



def camera_CB(msg):
    global rscam
    global depthcam
    global camPub
    global rangePub
    rospy.sleep(1)
    cameraImgToROS(rscam, camPub)
    rospy.sleep(1)
    rangeImgToROS(depthcam, rangePub)

def cameraImgToROS(camera, pub):
    Img = camera.getImageArray()
    Img = np.array(Img, dtype=np.uint8)
    Img = np.fliplr(Img)
    rotatedImg = np.rot90(Img)
    test = np.zeros((camera.getHeight(), camera.getWidth(), 3), np.uint8)
    bridge = CvBridge()
    rosmsg = bridge.cv2_to_imgmsg(rotatedImg, 'rgb8')

    pub.publish(rosmsg)

def rangeImgToROS(camera, pub):
    Img = camera.getRangeImageArray()
    Img = np.array(Img)
    Img = np.fliplr(Img)
    rotatedImg = np.rot90(Img)
    #norm_img = cv2.normalize(rotatedImg, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    #norm_img.astype(np.uint8)
    bridge = CvBridge()
    rosmsg = bridge.cv2_to_imgmsg(rotatedImg, encoding = "passthrough")

    pub.publish(rosmsg)

rospy.init_node('camera_test_node')
robot = Robot()
global camPub
global rangePub
camPub = rospy.Publisher('/camera/image', Image, queue_size=20)
rangePub = rospy.Publisher('/range_finder/image', Image, queue_size=20)
# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())
SAMPLE_TIME = 100
camera = robot.getDevice('camera')
depth = robot.getDevice('range-finder')
global rscam
global depthcam
rscam = Camera('camera')
depthcam = RangeFinder('range-finder')
depthcam.enable(SAMPLE_TIME)
rscam.enable(SAMPLE_TIME)
rospy.Subscriber('publish_images', Bool, camera_CB)

clockPublisher = rospy.Publisher('clock', Clock, queue_size=1)
if not rospy.get_param('use_sim_time', False):
    rospy.logwarn('use_sim_time is not set!')


# Main loop:
# - perform simulation steps until Webots is stopping the controller
while robot.step(timestep) != -1 and not rospy.is_shutdown():
    msg = Clock()
    time = robot.getTime()
    msg.clock.secs = int(time)
    # round prevents precision issues that can cause problems with ROS timers
    msg.clock.nsecs = round(1000 * (time - msg.clock.secs)) * 1.0e+6
    clockPublisher.publish(msg)

