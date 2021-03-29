from vision.vision_controller import ObjectInfo
import rospy
import actionlib
from bin_picking.msg import PickObjectAction, PickObjectGoal
import cv2
from cv_bridge import CvBridge
import numpy as np 
import cv_bridge

class RobotController():
    __init__(self):
    rospy.init_node("robot_controller")
    self.bridge = CvBridge()
    self.is_home
    self.client = actionlib.SimpleActionClient("pick_object", PickObjectAction)
    self.goal = PickObjectGoal()
    
    def pick_up(self, object_info: ObjectInfo, rgb, depth):
        print("Waiting for server")
        self.client.wait_for_server()

        self.goal.mask = self.bridge.cv2_to_imgmsg(object_info.mask_full)
        self.goal.reference_img = self.bridge.cv2_to_imgmsg(rgb)
        self.goal.depth_img = self.bridge.cv2_to_imgmsg(depth)
        self.goal.goal_msg = "pick_up"

        print("sending goal")
        self.client.send_goal(self.goal)
        self.client.wait_for_result()
        result = self.client.get_result()

        return result

    def find(self, object_info: ObjectInfo, rgb):
        print("Waiting for server")
        self.client.wait_for_server()

        self.goal.mask = self.bridge.cv2_to_imgmsg(object_info.mask_full)
        self.goal.reference_img = self.bridge.cv2_to_imgmsg(rgb)
        self.goal.goal_msg = "find"

        print("sending goal")
        self.client.send_goal(self.goal)
        self.client.wait_for_result()
        result = self.client.get_result()

        return result

    def move_out_of_view(self):
        print("Waiting for server")
        self.client.wait_for_server()

        self.goal.goal_msg = "move_out_of_view"

        print("sending goal")
        self.client.send_goal(self.goal)
        self.client.wait_for_result()
        result = self.client.get_result()

        self.is_home = result

        return result

    def is_out_of_view(self):
        if self.is_home == True:
            return True
        if self.is_home == False:
            return False