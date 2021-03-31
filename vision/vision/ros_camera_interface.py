import numpy as np
import rospy
from PIL import Image as pimg
import actionlib
from ur_e_webots.msg import GripperAction, GripperGoal
from cv_bridge import CvBridge
import cv2


class ROSCamera:
    def __init__(self, create_node=False):
        if create_node:
            rospy.init_node("test", anonymous=True)
        self.client = actionlib.SimpleActionClient("gripper_server", GripperAction)
        self.client.wait_for_server()
        self.bridge = CvBridge()

    def get_image(self):
        goal = GripperGoal()
        goal.action = "get_image"
        self.client.send_goal(goal)
        self.client.wait_for_result()
        result = self.client.get_result()
        np_img = self.bridge.compressed_imgmsg_to_cv2(result.rgb_compressed, desired_encoding="bgr8")
        return np_img

    def get_depth(self):
        goal = GripperGoal()
        goal.action = "get_depth"
        self.client.send_goal(goal)
        self.client.wait_for_result()
        result = self.client.get_result()
        np_img = self.bridge.imgmsg_to_cv2(result.depth, desired_encoding="passthrough")
        return np_img


if __name__ == "__main__":
    camera = ROSCamera(True)
    img = camera.get_image()
    cv2.imshow("rgb", img)
    cv2.imwrite("background.png", img)
    #depth = camera.get_depth()
    #cv2.imshow("depth", depth)
    cv2.waitKey(0)