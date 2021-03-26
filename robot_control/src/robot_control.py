from utility.find_objets.find_objects import ObjectInfo
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

    
    def pick_up(self, object_info: ObjectInfo, rgb, depth):
        self.client = actionlib.SimpleActionClient("pick_object", PickObjectAction)
        print("Waiting for server")
        client.wait_for_server()

        goal = PickObjectGoal()

        goal.mask = bridge.cv2_to_imgmsg(object_info.mask_full)
        goal.reference_img = bridge.cv2_to_imgmsg(rgb)
        goal.depth_img = bridge.cv2_to_imgmsg(depth)
        goal.goal_msg = "pick_up"

        print("sending goal")
        client.send_goal(goal)
        client.wait_for_result()
        result = client.get_result()

        return result