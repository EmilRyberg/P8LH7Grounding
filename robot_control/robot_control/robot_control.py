from vision_lib.vision_controller import ObjectInfo
import rospy
import actionlib
from bin_picking.msg import PickObjectAction, PickObjectGoal
import cv2
from cv_bridge import CvBridge
import numpy as np 
import cv_bridge


class RobotController:
    def __init__(self):
        self.bridge = CvBridge()
        self.is_home = True
        self.client = actionlib.SimpleActionClient("pick_object", PickObjectAction)
    
    def pick_up(self, object_info: ObjectInfo, rgb, depth):
        rospy.loginfo("Waiting for server")
        self.client.wait_for_server()

        goal = PickObjectGoal()
        goal.mask = self.bridge.cv2_to_imgmsg(object_info.mask_full)
        goal.reference_img = self.bridge.cv2_to_imgmsg(rgb)
        goal.depth_img = self.bridge.cv2_to_imgmsg(depth)
        goal.command = "pick_object"

        rospy.loginfo("sending goal")
        self.client.send_goal(goal)
        self.client.wait_for_result()
        result = self.client.get_result()

        return result

    def find(self, object_info: ObjectInfo, rgb):
        rospy.loginfo("Waiting for server")
        self.client.wait_for_server()

        goal = PickObjectGoal()
        goal.mask = self.bridge.cv2_to_imgmsg(object_info.mask_full)
        goal.reference_img = self.bridge.cv2_to_imgmsg(rgb)
        goal.command = "find"

        rospy.loginfo("sending goal")
        self.client.send_goal(goal)
        self.client.wait_for_result()
        result = self.client.get_result()

        return result

    def move_out_of_view(self):
        rospy.loginfo("Waiting for server")
        self.client.wait_for_server()

        goal = PickObjectGoal()
        goal.command = "move_out_of_view"

        rospy.loginfo("sending goal")
        self.client.send_goal(goal)
        self.client.wait_for_result()
        result = self.client.get_result()

        self.is_home = result

        return result

    def is_out_of_view(self):
        return not self.is_home